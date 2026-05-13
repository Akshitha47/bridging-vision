import streamlit as st
import torch
import os
from PIL import Image

# MUST set page config first, before any other st calls


st.title("🖼️ AI Image Caption Generator with Voice")
st.write("Upload an image and get AI-generated captions with multi-language voice output")

# Initialize session state
if 'language' not in st.session_state:
    st.session_state.language = 'en'
if 'voice_speed' not in st.session_state:
    st.session_state.voice_speed = 1.0
if 'history' not in st.session_state:
    st.session_state.history = []

# Set device
device = "cuda" if torch.cuda.is_available() else "cpu"

# Load the pre-trained model and processor
MODEL_DIRECTORY = "blip-image-captioning-base"
MODEL_PATH = os.path.join(MODEL_DIRECTORY)
PROCESSOR_PATH = os.path.join(MODEL_DIRECTORY)

processor = None
model = None
model_loaded = False

# Try to load model
# In app2.py - Replace your current try/except block with this:
# Inside app2.py, find the model loading block and replace with:
try:
    from transformers import BlipProcessor, BlipForConditionalGeneration
    with st.spinner("Loading AI Model..."):
        MODEL_ID = "Salesforce/blip-image-captioning-base"
        processor = BlipProcessor.from_pretrained(MODEL_ID)
        model = BlipForConditionalGeneration.from_pretrained(MODEL_ID)
        model = model.to(device)
        model_loaded = True
        st.success("✅ Model loaded successfully!")
except Exception as e:
    st.warning(f"⚠️ Model loading issue: {str(e)}")
    model_loaded = False

def create_audio_player(text):
    """Generate audio from text"""
    try:
        from gtts import gTTS
        import tempfile
        import base64
        
        tts = gTTS(text=text, lang=st.session_state.language)
        
        with tempfile.NamedTemporaryFile(delete=False, suffix='.mp3') as fp:
            tts.save(fp.name)
            with open(fp.name, 'rb') as audio_file:
                audio_bytes = audio_file.read()
            audio_base64 = base64.b64encode(audio_bytes).decode()
        
        audio_player = f'''
            <audio controls style="width: 100%;">
                <source src="data:audio/mp3;base64,{audio_base64}" type="audio/mp3">
                Your browser does not support the audio element.
            </audio>
        '''
        return audio_player
    except Exception as e:
        return f"<p>Error generating audio: {str(e)}</p>"

def update_history(caption):
    """Update caption history"""
    from datetime import datetime
    timestamp = datetime.now().strftime("%H:%M:%S")
    st.session_state.history.append((caption, timestamp))
    st.session_state.history = st.session_state.history[-10:]

# Sidebar
with st.sidebar:
    st.markdown("## ⚙️ Settings")
    
    st.session_state.language = st.selectbox(
        "Select Voice Language",
        options=['en', 'es', 'fr', 'de', 'it', 'pt', 'hi', 'ja', 'ko', 'zh-CN'],
        format_func=lambda x: {
            'en': 'English 🇬🇧',
            'es': 'Spanish 🇪🇸',
            'fr': 'French 🇫🇷',
            'de': 'German 🇩🇪',
            'it': 'Italian 🇮🇹',
            'pt': 'Portuguese 🇵🇹',
            'hi': 'Hindi 🇮🇳',
            'ja': 'Japanese 🇯🇵',
            'ko': 'Korean 🇰🇷',
            'zh-CN': 'Chinese 🇨🇳'
        }[x]
    )
    
    st.markdown("## 📜 Caption History")
    if st.session_state.history:
        for i, (caption, time) in enumerate(reversed(st.session_state.history)):
            with st.expander(f"#{len(st.session_state.history)-i} ({time})"):
                st.write(caption)
    else:
        st.info("No captions yet")
    
    if st.button("🗑️ Clear History", type="secondary"):
        st.session_state.history = []
        st.rerun()

# Main content
if model_loaded:
    st.markdown("---")
    st.write("Upload an image to generate captions with voice output")
    
    col1, col2 = st.columns([1, 1.5])
    
    with col1:
        st.markdown("### 📸 Upload Image")
        uploaded_file = st.file_uploader("Choose an image...", type=["jpg","jpeg", "png"])
    
    if uploaded_file is not None:
        image = Image.open(uploaded_file).convert("RGB")
        with col1:
            st.image(image, caption="Uploaded Image", use_column_width=True)
        
        with col2:
            st.markdown("### Generation Options")
            
            tab1, tab2 = st.tabs(["Conditional", "Unconditional"])
            
            with tab1:
                text_input = st.text_input(
                    "Enter text prefix:",
                    value="a photography of",
                    help="This will guide the caption generation"
                )
                if st.button("🎯 Generate Conditional Caption", use_container_width=True):
                    try:
                        with st.spinner("Generating caption..."):
                            inputs = processor(image, text_input, return_tensors="pt").to(device)
                            with torch.no_grad():
                                out = model.generate(**inputs, max_length=50)
                            caption = processor.decode(out[0], skip_special_tokens=True)
                            
                            update_history(caption)
                            
                            st.success("✅ Caption Generated!")
                            st.info(f"📝 {caption}")
                            
                            audio_html = create_audio_player(caption)
                            st.markdown("🔊 **Voice Output:**", unsafe_allow_html=True)
                            st.markdown(audio_html, unsafe_allow_html=True)
                    except Exception as e:
                        st.error(f"Error: {str(e)}")
            
            with tab2:
                if st.button("🎲 Generate Unconditional Caption", use_container_width=True):
                    try:
                        with st.spinner("Generating caption..."):
                            inputs = processor(image, return_tensors="pt").to(device)
                            with torch.no_grad():
                                out = model.generate(**inputs, max_length=50)
                            caption = processor.decode(out[0], skip_special_tokens=True)
                            
                            update_history(caption)
                            
                            st.success("✅ Caption Generated!")
                            st.info(f"📝 {caption}")
                            
                            audio_html = create_audio_player(caption)
                            st.markdown("🔊 **Voice Output:**", unsafe_allow_html=True)
                            st.markdown(audio_html, unsafe_allow_html=True)
                    except Exception as e:
                        st.error(f"Error: {str(e)}")
else:
    st.markdown("---")
    st.info("📝 Demo Mode - Model Not Available")
    st.write("""
    The AI model is not currently loaded. The app is in demo mode.
    
    **To use the full feature:**
    - Ensure the 'blip-image-captioning-base' directory exists
    - Check that all model files are present
    - Restart the app after placing the model
    """)
    
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("About")
        st.write("""
        This app uses BLIP (Bootstrapping Language-Image Pre-training) 
        for image captioning with multi-language voice support.
        """)
    
    with col2:
        st.subheader("Features")
        st.write("""
        ✓ Image Captioning (Conditional & Unconditional)
        ✓ Multi-language voice output via gTTS
        ✓ Caption history tracking
        ✓ Language selection (10+ languages)
        """)