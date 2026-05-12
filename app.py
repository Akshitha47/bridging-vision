import streamlit as st
import torch
import sys
import os
from PIL import Image
from gtts import gTTS # Added missing import
import io # Added missing import

# FIX 1: Move page_config to the very top before any other st. commands
st.set_page_config(page_title="AI Image Captioning", layout="wide")

# 1. Add the subfolder to the Python path
sys.path.append(os.path.join(os.getcwd(), "imagecaption"))

st.sidebar.title("Navigation")
selection = st.sidebar.radio("Go to", ["Caption Generation", "Voice Generation"])

if selection == "Voice Generation":
    try:
        import imagecaption.app2 as app2
        st.stop() # Stop execution here so it doesn't run the rest of the file
    except Exception as e:
        st.error(f"Could not load Voice Generation: {e}")

# Main UI setup
st.title("🖼️ AI Image Captioning WebApp")
st.write("Generate captions for your images using AI")

device = "cuda" if torch.cuda.is_available() else "cpu"

# Load the pre-trained model and processor
MODEL_ID = "Salesforce/blip-image-captioning-base"
processor = None
model = None
model_loaded = False

try:
    from transformers import BlipProcessor, BlipForConditionalGeneration
    with st.spinner("Downloading and loading model..."):
        processor = BlipProcessor.from_pretrained(MODEL_ID)
        model = BlipForConditionalGeneration.from_pretrained(MODEL_ID)
        model = model.to(device)
        model_loaded = True
        st.success("✅ Model loaded successfully!")
except Exception as e:
    st.warning(f"⚠️ Could not load model: {str(e)}")
    model_loaded = False

if model_loaded:
    st.markdown("---")
    st.write("Upload an image to generate captions")
    
    uploaded_file = st.sidebar.file_uploader("Choose an image...", type=["jpg","jpeg","png"])

    if uploaded_file is not None:
        image = Image.open(uploaded_file).convert("RGB")
        col1, col2 = st.columns(2)
        
        with col1:
            st.image(image, caption="Uploaded Image", use_container_width=True)
        
        with col2:
            st.subheader("Caption Options")
            text_input = st.text_input("Enter text prefix for captioning:", "a photography of")
            
            if st.button("🎯 Generate Conditional Caption", use_container_width=True):
                with st.spinner("Generating caption..."):
                    try:
                        inputs = processor(image, text_input, return_tensors="pt").to(device)
                        # FIX 2: Added missing model.generate call
                        out = model.generate(**inputs, max_length=50) 
                        caption = processor.decode(out[0], skip_special_tokens=True)
                        st.success("✅ Caption Generated!")
                        st.info(f"Caption: {caption}")
                        
                        tts = gTTS(text=caption, lang='en')
                        audio_fp = io.BytesIO()
                        tts.write_to_fp(audio_fp)
                        st.audio(audio_fp, format='audio/mp3')
                    except Exception as e:
                        st.error(f"Error generating caption: {str(e)}")

            if st.button("📸 Generate Unconditional Caption", use_container_width=True):
                with st.spinner("Generating caption..."):
                    try:
                        inputs = processor(image, return_tensors="pt").to(device)
                        # FIX 3: Added missing model.generate call
                        out = model.generate(**inputs, max_length=50) 
                        caption = processor.decode(out[0], skip_special_tokens=True)
                        st.success("✅ Caption Generated!")
                        st.info(f"Caption: {caption}")
                        
                        tts = gTTS(text=caption, lang='en')
                        audio_fp = io.BytesIO()
                        tts.write_to_fp(audio_fp)
                        st.audio(audio_fp, format='audio/mp3')
                    except Exception as e:
                        st.error(f"Error generating caption: {str(e)}")
else:
    st.info("📝 Demo Mode - Model Not Available")
    # ... rest of your demo UI code ...