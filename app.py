import streamlit as st
import torch
import sys
import os
from PIL import Image


# 1. Add the subfolder to the Python path so app.py can "see" app2.py
sys.path.append(os.path.join(os.getcwd(), "imagecaption"))

st.sidebar.title("Navigation")
selection = st.sidebar.radio("Go to", ["Caption Generation", "Voice Generation"])

if selection == "Caption Generation":
    # --- Put your existing app.py code logic here ---
    st.title("🖼️ Image Captioning")
    # [Insert your current app.py model loading and UI code here]

elif selection == "Voice Generation":
    # --- Run the code from imagecaption/app2.py ---
    try:
        import imagecaption.app2 as app2
        # This will execute the code inside your app2.py file
    except Exception as e:
        st.error(f"Could not load Voice Generation: {e}")

# Streamlit page config MUST be first
st.set_page_config(page_title="AI Image Captioning", layout="wide")

st.title("🖼️ AI Image Captioning WebApp")
st.write("Generate captions for your images using AI")

# Set device
device = "cuda" if torch.cuda.is_available() else "cpu"

# Load the pre-trained model and processor
MODEL_DIRECTORY = "blip-image-captioning-base"
MODEL_PATH = os.path.join(MODEL_DIRECTORY)
PROCESSOR_PATH = os.path.join(MODEL_DIRECTORY)

# Initialize model variables
processor = None
model = None
model_loaded = False

# Try to load model
try:
    from transformers import BlipProcessor, BlipForConditionalGeneration
    
    # MODIFIED: Remove the 'if os.path.exists' check and load directly from the Hub
    with st.spinner("Downloading and loading model..."):
        # We use the official model ID instead of a local path
        MODEL_ID = "Salesforce/blip-image-captioning-base"
        
        processor = BlipProcessor.from_pretrained(MODEL_ID)
        model = BlipForConditionalGeneration.from_pretrained(MODEL_ID)
        model = model.to(device)
        model_loaded = True
        st.success("✅ Model loaded successfully!")
except Exception as e:
    st.warning(f"⚠️ Could not load model: {str(e)}")
    model_loaded = False

# Main app content
if model_loaded:
    st.markdown("---")
    st.write("Upload an image to generate captions")
    
    # Upload image through Streamlit sidebar
    uploaded_file = st.sidebar.file_uploader("Choose an image...", type=["jpg","jpeg","png"])

    if uploaded_file is not None:
        # Display the uploaded image
        image = Image.open(uploaded_file).convert("RGB")
        col1, col2 = st.columns(2)
        
        with col1:
            st.image(image, caption="Uploaded Image", use_column_width=True)
        
        with col2:
            st.subheader("Caption Options")
            
            # Perform image captioning
            text_input = st.text_input("Enter text prefix for captioning:", "a photography of")
            
            if st.button("🎯 Generate Conditional Caption", use_container_width=True):
                with st.spinner("Generating caption..."):
                    try:
                        inputs = processor(image, text_input, return_tensors="pt").to(device)
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
    st.markdown("---")
    st.info("📝 Demo Mode - Model Not Available")
    st.write("""
    The model is not currently loaded. This could be because:
    - The model directory 'blip-image-captioning-base' is not found
    - The model files are still downloading
    - There was an error loading the model
    
    To use the full app, ensure the BLIP model is properly set up.
    """)
    
    # Show demo information
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("About This App")
        st.write("""
        This app uses the BLIP (Bootstrapping Language-Image Pre-training) 
        model to generate captions for images.
        """)
    
    with col2:
        st.subheader("Features")
        st.write("""
        - Conditional Caption Generation
        - Unconditional Caption Generation
        - Support for JPG, JPEG, PNG formats
        """)
