import streamlit as st
import tempfile
import os

st.set_page_config(page_title="🎤 Microphone Test", layout="wide")

st.title("🎤 Streamlit Microphone Permission & Recording Test")

st.markdown("""
### This app will automatically request microphone access when you click record!
**Instructions:**
1. Click the recording button below
2. **ALLOW** microphone access when your browser asks
3. Speak into your microphone
4. The audio will be saved and played back
""")

# Built-in Streamlit audio input (automatically requests permissions)
st.header("🎙️ Built-in Audio Recording")
st.write("Click below to start recording (browser will ask for microphone permission)")

audio_bytes = st.audio_input("🔴 Click to record your message")

if audio_bytes is not None:
    # Display the recorded audio
    st.success("✅ Recording successful!")
    st.audio(audio_bytes, format='audio/wav')
    
    # Show audio details
    audio_size = len(audio_bytes.getvalue())
    st.info(f"📊 Audio size: {audio_size:,} bytes")
    
    # Save to temporary file
    with tempfile.NamedTemporaryFile(delete=False, suffix='.wav') as temp_file:
        temp_file.write(audio_bytes.getvalue())
        temp_path = temp_file.name
    
    st.success(f"💾 Audio saved to: `{temp_path}`")
    
    # Option to download
    st.download_button(
        label="📥 Download Recording",
        data=audio_bytes.getvalue(),
        file_name="my_recording.wav",
        mime="audio/wav"
    )

# Permission troubleshooting
st.divider()
st.header("🔧 Troubleshooting")

with st.expander("🚨 If microphone doesn't work"):
    st.markdown("""
    ### Windows Settings:
    1. Press `Win + I` → **Privacy & Security** → **Microphone** 
    2. Enable **"Microphone access"**
    3. Enable **"Let apps access your microphone"** 
    4. Enable **"Let desktop apps access your microphone"**
    
    ### Browser Settings (Chrome):
    1. Go to: `chrome://settings/content/microphone`
    2. Enable **"Sites can ask to use your microphone"**
    3. When this site asks, click **"Allow"**
    
    ### Quick Test:
    - Try Windows **Voice Recorder** app first
    - Check if other apps (Teams, Zoom) can use microphone
    """)

with st.expander("🌐 Browser Permission Status"):
    st.markdown("""
    **Current URL permissions:**
    - Check your browser's address bar for 🎤 microphone icon
    - Click it to manage permissions for this site
    - Make sure microphone is set to "Allow"
    """)

# System info
st.divider()
st.subheader("💻 System Information")
col1, col2 = st.columns(2)

with col1:
    st.write("**Environment:**")
    st.write(f"- Running on: {'localhost' if 'localhost' in st.get_option('server.baseUrlPath') or True else 'deployed'}")
    st.write("- Streamlit version:", st.__version__)

with col2:
    st.write("**Browser Support:**")
    st.write("✅ Chrome, Edge, Firefox")
    st.write("✅ Safari (desktop)")
    st.write("❌ Internet Explorer")
