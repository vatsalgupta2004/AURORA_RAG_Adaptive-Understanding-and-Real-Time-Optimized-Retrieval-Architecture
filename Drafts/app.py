import io
import os
import time
import numpy as np
import streamlit as st

# ---------- Ingestion ----------
import fitz  # PyMuPDF
from docx import Document

# ---------- Embeddings & FAISS ----------
from sentence_transformers import SentenceTransformer
import faiss

# ---------- LLM (Ollama first; optional llama-cpp fallback) ----------
USE_LLAMACPP_FALLBACK = False  # Set True to try llama-cpp if Ollama not available
try:
    import ollama  # pip install ollama
except Exception:
    ollama = None
import subprocess


# ==========================
# Utility: Text Extraction
# ==========================
def extract_pdf(file_bytes: bytes, name: str):
    doc = fitz.open(stream=file_bytes, filetype="pdf")
    pages = []
    for i, page in enumerate(doc):
        text = page.get_text("text")
        if text:
            pages.append({"doc": name, "page": i + 1, "text": text})
    return pages


def extract_docx(file_bytes: bytes, name: str):
    file = io.BytesIO(file_bytes)
    doc = Document(file)
    text = "\n".join([p.text for p in doc.paragraphs])
    return [{"doc": name, "page": 1, "text": text}]


def extract_txt(file_bytes: bytes, name: str):
    text = file_bytes.decode("utf-8", errors="ignore")
    return [{"doc": name, "page": 1, "text": text}]


def extract_any(uploaded_file):
    name = uploaded_file.name
    data = uploaded_file.read()
    if name.lower().endswith(".pdf"):
        return extract_pdf(data, name)
    elif name.lower().endswith(".docx"):
        return extract_docx(data, name)
    elif name.lower().endswith(".txt"):
        return extract_txt(data, name)
    else:
        raise ValueError("Unsupported file type. Use PDF, DOCX, or TXT.")


# ==========================
# Chunking
# ==========================
def chunk_text(pages, target_chars=1200, overlap_chars=200):
    chunks = []
    for p in pages:
        text = p["text"].strip()
        start = 0
        n = len(text)
        while start < n:
            end = min(n, start + target_chars)
            chunk = text[start:end]
            if chunk.strip():
                chunks.append({
                    "doc": p["doc"],
                    "page": p["page"],
                    "text": chunk
                })
            if end == n:
                break
            start = end - overlap_chars
            start = max(0, start)
    return chunks


# ==========================
# Embeddings & FAISS
# ==========================
@st.cache_resource(show_spinner=False)
def load_embedder(model_name="sentence-transformers/all-MiniLM-L6-v2"):
    return SentenceTransformer(model_name)


def l2_normalize(mat: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(mat, axis=1, keepdims=True) + 1e-12
    return mat / norms


class VectorIndex:
    def __init__(self, dim):
        self.index = faiss.IndexFlatIP(dim)
        self.meta = []

    def add(self, embeddings: np.ndarray, metas: list):
        assert embeddings.shape[0] == len(metas)
        self.index.add(embeddings)
        self.meta.extend(metas)

    def search(self, query_vec: np.ndarray, top_k: int = 5):
        D, I = self.index.search(query_vec, top_k)
        results = []
        for idx, score in zip(I[0], D[0]):
            if idx == -1:
                continue
            m = self.meta[idx]
            results.append((m, float(score)))
        return results


# ==========================
# Retriever (MMR)
# ==========================
def mmr_rerank(query_vec, doc_vecs, candidates, k=5, lambda_mult=0.5):
    if not candidates:
        return []
    selected = []
    candidate_set = set(candidates)
    sims = doc_vecs[candidates] @ query_vec.T
    sims = sims.ravel()

    while len(selected) < min(k, len(candidates)):
        if not selected:
            best_idx = candidates[int(np.argmax(sims))]
            selected.append(best_idx)
            candidate_set.remove(best_idx)
            continue

        gains = []
        for c in list(candidate_set):
            rel = float(doc_vecs[c] @ query_vec.ravel())
            if selected:
                sim_to_sel = np.max(doc_vecs[c] @ doc_vecs[selected].T)
            else:
                sim_to_sel = 0.0
            score = lambda_mult * rel - (1 - lambda_mult) * sim_to_sel
            gains.append((score, c))

        if not gains:
            break

        gains.sort(reverse=True, key=lambda x: x[0])
        best = gains[0][1]
        selected.append(best)
        candidate_set.remove(best)

    return selected


# ==========================
# LLM Calls
# ==========================
def call_llm_ollama(model_name: str, prompt: str, temperature: float = 0.2, max_tokens: int = 512):
    if ollama is None:
        raise RuntimeError("Ollama Python package not installed.")
    resp = ollama.chat(
        model=model_name,
        messages=[{"role": "user", "content": prompt}],
        options={"temperature": temperature, "num_predict": max_tokens}
    )
    return resp["message"]["content"]


def call_llm_llamacpp(model_path: str, prompt: str, temperature: float = 0.2, max_tokens: int = 512):
    from llama_cpp import Llama
    llm = Llama(model_path=model_path, n_ctx=4096, logits_all=False)
    out = llm(
        prompt,
        max_tokens=max_tokens,
        temperature=temperature,
        stop=["</s>"]
    )
    return out["choices"][0]["text"]


# ==========================
# RAG Assembly
# ==========================
def build_prompt(query: str, retrieved_chunks: list, max_context_chars: int = 4000):
    ctx_parts = []
    char_budget = max_context_chars
    for ch in retrieved_chunks:
        piece = f"[Source: {ch['doc']} | Page {ch['page']}]\n{ch['text'].strip()}\n\n"
        if len(piece) <= char_budget:
            ctx_parts.append(piece)
            char_budget -= len(piece)
        else:
            ctx_parts.append(piece[:char_budget])
            break
    context = "".join(ctx_parts)

    system = (
        "You are a helpful assistant. Answer the user's question ONLY using the provided sources.\n"
        "If the answer is not contained in the sources, say you cannot find it.\n"
        "Cite sources as [doc-name, page] inline when relevant."
    )
    prompt = (
        f"{system}\n\n"
        f"### Sources:\n{context}\n"
        f"### Question:\n{query}\n"
        f"### Answer:"
    )
    return prompt


# ==========================
# Helper: get Ollama models
# ==========================
def get_ollama_models():
    try:
        result = subprocess.run(["ollama", "list"], capture_output=True, text=True)
        models = []
        for line in result.stdout.strip().split("\n")[1:]:
            if line:
                models.append(line.split()[0])
        return models if models else ["llama3.2:3b"]
    except Exception:
        return ["llama3.2:3b"]


# ==========================
# Streamlit UI
# ==========================
st.set_page_config(page_title="Smart Document Assistant", layout="wide")
st.title("🧠 Smart Document Assistant — RAG with FAISS + LLaMA (Local)")

with st.sidebar:
    st.header("Settings")
    available_models = get_ollama_models()
    model_choice = st.selectbox("Ollama model", available_models, index=0, key="model_choice")

    top_k = st.slider("Top-K chunks", 1, 10, 5, key="top_k_slider")
    use_mmr = st.checkbox("Use MMR (diversity)", value=True, key="mmr_checkbox")
    temperature = st.slider("Temperature", 0.0, 1.0, 0.2, 0.05, key="temp_slider")
    max_tokens = st.slider("Max tokens (LLM output)", 64, 1024, 512, 32, key="max_tokens_slider")

    st.markdown("---")
    st.caption("Optional (fallback if not using Ollama):")
    model_path = st.text_input("llama.cpp .gguf path (optional)", value="", key="model_path_input")

    st.markdown("---")
    st.header("Index Builder")
    uploaded_files = st.file_uploader("Upload PDF / DOCX / TXT (multiple allowed)",
                                      type=["pdf", "docx", "txt"], accept_multiple_files=True, key="file_uploader")
    build_index_btn = st.button("📚 Build/Update Index", key="build_index_btn")


# ==========================
# Session State
# ==========================
if "doc_chunks" not in st.session_state:
    st.session_state.doc_chunks = []
if "embedder" not in st.session_state:
    with st.spinner("Loading embedding model..."):
        st.session_state.embedder = load_embedder()
if "index" not in st.session_state:
    st.session_state.index = None
if "embeddings" not in st.session_state:
    st.session_state.embeddings = None


# ==========================
# Build / Update Index
# ==========================
if build_index_btn:
    if not uploaded_files:
        st.warning("Please upload at least one document.")
    else:
        with st.spinner("Extracting text..."):
            pages_all = []
            for f in uploaded_files:
                try:
                    pages_all.extend(extract_any(f))
                except Exception as e:
                    st.error(f"Failed to read {f.name}: {e}")

        with st.spinner("Chunking..."):
            chunks = chunk_text(pages_all, target_chars=1200, overlap_chars=200)

        with st.spinner("Embedding chunks..."):
            texts = [c["text"] for c in chunks]
            emb = st.session_state.embedder.encode(texts, convert_to_numpy=True, batch_size=64, show_progress_bar=True)
            emb = emb.astype("float32")
            emb = l2_normalize(emb)

        with st.spinner("Building FAISS index..."):
            dim = emb.shape[1]
            index = VectorIndex(dim)
            index.add(emb, [{"doc": c["doc"], "page": c["page"], "text": c["text"]} for c in chunks])

        st.session_state.doc_chunks = [{"doc": c["doc"], "page": c["page"], "text": c["text"]} for c in chunks]
        st.session_state.embeddings = emb
        st.session_state.index = index

        st.success(f"Index built ✅  |  Documents: {len(uploaded_files)}  |  Chunks: {len(chunks)}")


# ==========================
# Main interaction
# ==========================
col_q, col_a = st.columns([1, 2])
with col_q:
    st.subheader("Ask a Question")
    query = st.text_input("Your question", placeholder="e.g., What are the key points about RAG?", key="query_input")
    ask_btn = st.button("🔎 Retrieve & Answer", key="ask_btn")

with col_a:
    st.subheader("Answer")
    answer_area = st.empty()
    citations_area = st.container()
    latency_area = st.empty()


def retrieve_top_chunks(query_text: str, k: int, use_mmr_flag: bool = True):
    if st.session_state.index is None or st.session_state.embeddings is None:
        st.warning("Build the index first from the sidebar.")
        return []
    q_vec = st.session_state.embedder.encode([query_text], convert_to_numpy=True)
    q_vec = l2_normalize(q_vec.astype("float32"))
    k_base = max(k * 3, k)
    D, I = st.session_state.index.index.search(q_vec, k_base)
    candidates = I[0].tolist()

    if not use_mmr_flag:
        results = []
        for idx, score in zip(I[0][:k], D[0][:k]):
            meta = st.session_state.index.meta[idx]
            results.append({**meta, "score": float(score)})
        return results
    else:
        doc_vecs = st.session_state.embeddings
        selected = mmr_rerank(q_vec, doc_vecs, candidates, k=k, lambda_mult=0.5)
        results = []
        for idx in selected:
            score = float(doc_vecs[idx] @ q_vec.ravel())
            meta = st.session_state.index.meta[idx]
            results.append({**meta, "score": score})
        return results


def rag_answer(query_text: str, top_k: int, model_name: str, temperature: float, max_tokens: int):
    hits = retrieve_top_chunks(query_text, k=top_k, use_mmr_flag=use_mmr)
    prompt = build_prompt(query_text, hits, max_context_chars=4000)

    t0 = time.time()
    try:
        if ollama is not None and model_name:
            output = call_llm_ollama(model_name, prompt, temperature=temperature, max_tokens=max_tokens)
        elif USE_LLAMACPP_FALLBACK and model_path:
            output = call_llm_llamacpp(model_path, prompt, temperature=temperature, max_tokens=max_tokens)
        else:
            raise RuntimeError("No LLM backend available. Install/run Ollama or set llama.cpp model path.")
    except Exception as e:
        output = f"LLM error: {e}"
    latency_ms = int((time.time() - t0) * 1000)
    return output, hits, latency_ms


if ask_btn:
    if not query.strip():
        st.warning("Please enter a question.")
    else:
        with st.spinner("Thinking..."):
            ans, sources, took = rag_answer(query, top_k, model_choice, temperature, max_tokens)

        answer_area.write(ans)
        latency_area.caption(f"⏱️ Latency: {took} ms")

        st.markdown("**Citations**")
        with citations_area:
            for s in sources:
                with st.expander(f"{s['doc']} — page {s['page']} (score: {s['score']:.3f})"):
                    snippet = s["text"]
                    snippet = snippet[:600] + ("..." if len(snippet) > 600 else "")
                    st.write(snippet)
