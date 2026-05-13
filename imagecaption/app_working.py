import streamlit as st
import torch
import os
from PIL import Image
from transformers import BlipProcessor, BlipForConditionalGeneration

# Set device
device = "cuda" if torch.cuda.is_available() else "cpu"

# Load the pre-trained model and processor
MODEL_DIRECTORY = "blip-image-captioning-base"
MODEL_PATH = os.path.join(MODEL_DIRECTORY)
PROCESSOR_PATH = os.path.join(MODEL_DIRECTORY)

# Streamlit app
st.set_page_config(page_title="AI Image Captioning", layout="wide")

# Check if the model and processor are downloaded
if not os.path.exists(MODEL_PATH) or not os.path.exists(PROCESSOR_PATH):
    st.error("❌ Error: Model and processor not found.")
    st.info("Please ensure the 'blip-image-captioning-base' directory exists with all model files.")
else:
    try:
        processor = BlipProcessor.from_pretrained(PROCESSOR_PATH)
        model = BlipForConditionalGeneration.from_pretrained(MODEL_PATH)
        model = model.to(device)
        
        st.title("🖼️ AI Image Captioning WebApp")
        st.write("Upload an image to generate captions using BLIP model")

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
                        inputs = processor(image, text_input, return_tensors="pt").to(device)
                        with torch.no_grad():
                            out = model.generate(**inputs, max_length=50)
                        caption = processor.decode(out[0], skip_special_tokens=True)
                        st.success("✅ Conditional Caption Generated!")
                        st.info(f"Caption: {caption}")

                if st.button("📸 Generate Unconditional Caption", use_container_width=True):
                    with st.spinner("Generating caption..."):
                        inputs = processor(image, return_tensors="pt").to(device)
                        with torch.no_grad():
                            out = model.generate(**inputs, max_length=50)
                        caption = processor.decode(out[0], skip_special_tokens=True)
                        st.success("✅ Unconditional Caption Generated!")
                        st.info(f"Caption: {caption}")
    except Exception as e:
        st.error(f"Error loading model: {str(e)}")
        st.info("Make sure all dependencies are installed and the model directory is correct.")
