# Smart Document Reader: A RAG-Based QA Assistant Using LLaMA and FAISS

![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)
![Streamlit](https://img.shields.io/badge/Streamlit-1.28+-red.svg)
![License](https://img.shields.io/badge/License-MIT-green.svg)

## 📌 Project Overview

Smart Document Reader is an **advanced AI-powered Question Answering (QA) assistant** that leverages state-of-the-art **Retrieval-Augmented Generation (RAG)** with Meta's LLaMA 3.2 model and FAISS vector search to deliver precise, document-grounded responses.

Unlike generic chatbots such as ChatGPT, this system **retrieves relevant information directly from user-uploaded documents** (PDF, DOCX, TXT), ensuring context-aware, accurate, and **privacy-focused answers** without relying on external APIs.

This project is developed as part of a **Final Year B.Tech (CSE) Minor Project**, demonstrating how open-source LLMs combined with retrieval systems can power domain-specific, real-world applications.

## 🚀 Key Features

### 📂 **Document Processing**
- **Multi-format support**: PDF, DOCX, TXT files
- **Smart text chunking** with sentence boundary detection
- **Real-time processing statistics** and quality metrics
- **Comprehensive document analysis dashboard**

### 🧠 **AI & RAG Pipeline**
- **Advanced RAG architecture** combining retrieval + generation
- **FAISS vector search** for semantic similarity
- **LLaMA 3.2 integration** via Ollama
- **Sentence Transformers** for embedding generation

### 🎤 **Voice & Audio Features**
- **Browser-based voice recording** with automatic permission requests
- **Real-time audio frequency analysis** and visualization
- **Speech-to-text transcription** using Google Speech API
- **Interactive audio quality assessment**

### 🖥️ **User Interface**
- **Professional Streamlit UI** with tabbed interface
- **Real-time performance monitoring**
- **Interactive charts and visualizations** (Plotly)
- **Export capabilities** (CSV, JSON, reports)

### 🔒 **Privacy & Performance**
- **Completely offline operation** (no external API calls for LLM)
- **Local document processing** ensuring data privacy
- **Optimized caching** for fast response times
- **Graceful fallbacks** for missing dependencies

---

## 🛠️ Tech Stack

| Component | Technology |
|-----------|------------|
| **LLM** | Meta LLaMA 3.2 (via Ollama) |
| **Vector Search** | FAISS (Facebook AI Similarity Search) |
| **Embeddings** | Sentence Transformers (all-MiniLM-L6-v2) |
| **Framework** | LangChain, Hugging Face |
| **UI/Frontend** | Streamlit |
| **Document Processing** | PyMuPDF, python-docx |
| **Audio Processing** | SciPy, SpeechRecognition, pyttsx3 |
| **Visualization** | Plotly, Pandas |
| **Optional Extensions** | Whisper (future), Weaviate/Milvus |

---

## 🔄 System Workflow

📤 Upload Documents → PDF/DOCX/TXT files
🔧 Preprocessing → Text extraction + smart chunking
🧮 Embedding Generation → Sentence Transformers
📊 FAISS Indexing → Vector storage for fast retrieval
🎤 User Input → Text or voice question
🔍 Retrieval → FAISS semantic search (top-k chunks)
🧠 Answer Generation → LLaMA 3.2 contextual response
📋 Result Display → Answer + sources + citations

## 📊 New Features Added

### 🎤 **Voice Interface**
- **Automatic microphone permission** requests via JavaScript
- **Real-time audio analysis** with frequency spectrum visualization
- **Speech-to-text conversion** with quality assessment
- **Audio export** and transcript download options

### 📄 **Document Analysis Dashboard**
- **Processing statistics** per document (time, accuracy, chunks)
- **Performance metrics** and efficiency analysis
- **Interactive charts** showing processing trends
- **Export options** for analysis reports (CSV, JSON, TXT)
- **Quality assessment** and error rate tracking

### 🔧 **System Improvements**
- **Optimized loading** with smart caching
- **Graceful dependency management** with fallbacks
- **Professional UI design** with gradient headers
- **Comprehensive error handling** and debugging


## 🚀 Installation & Setup

### Prerequisites
Install Python 3.8+
pip install streamlit numpy pandas

Core features
pip install scipy SpeechRecognition PyMuPDF python-docx

Advanced features (optional)
pip install plotly sentence-transformers faiss-cpu pyttsx3

LLM support
pip install ollama

### Running the Application

Clone the repository
git clone https://github.com/yourusername/smart-document-reader
cd smart-document-reader

Install dependencies
pip install -r requirements.txt

Start Ollama (for LLM)
ollama pull llama3.2:3b

Run the application
streamlit run app.py

---

## 📱 Usage

1. **🎤 Enable Microphone**: Grant browser permissions for voice features
2. **📚 Upload Documents**: Add PDF/DOCX/TXT files via sidebar
3. **🔨 Build Index**: Process documents and create vector embeddings
4. **❓ Ask Questions**: Use text input or voice recording
5. **📊 View Analysis**: Check document processing statistics
6. **📥 Export Results**: Download answers, transcripts, and reports

---

## 🔮 Future Scope

### 📈 **Immediate Enhancements**
- **Multi-document querying** across entire knowledge base
- **Advanced audio features** with noise reduction
- **Real-time collaboration** and document sharing

### 🌐 **Research & Development**
- **Performance benchmarking** against commercial solutions
- **Domain-specific fine-tuning** (legal, medical, academic)
- **Integration with institutional systems**

### ☁️ **Scalability**
- **Hybrid deployment** (local + cloud processing)
- **Enterprise features** (user management, analytics)
- **API development** for third-party integrations

---

## 🎓 Academic Context

This project serves as a **comprehensive demonstration** of modern NLP techniques:

- **Retrieval-Augmented Generation (RAG)** implementation
- **Vector database** integration and optimization  
- **Multi-modal AI** combining text and voice interfaces
- **Real-world application** of academic research

**Suitable for**: Final year projects, research publications, industry demonstrations

---

## 📚 References & Resources

- **Meta LLaMA 3.2**: [ai.meta.com/llama](https://ai.meta.com/llama/)
- **RAG Paper**: [arxiv.org/abs/2005.11401](https://arxiv.org/abs/2005.11401)
- **FAISS Library**: [github.com/facebookresearch/faiss](https://github.com/facebookresearch/faiss)
- **Sentence Transformers**: [huggingface.co/sentence-transformers](https://huggingface.co/sentence-transformers)
- **Streamlit Framework**: [streamlit.io](https://streamlit.io/)
- **Ollama Local LLM**: [ollama.ai](https://ollama.ai/)

---

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## 📧 Contact

For questions or collaboration opportunities, please reach out via [gupta.vatsal2004@gmail.com](mailto:your-email@domain.com)


**⭐ If you found this project helpful, please consider giving it a star!**
