import streamlit as st
import torch
from PIL import Image
from transformers import BlipProcessor, BlipForConditionalGeneration

# 1. Page Config MUST be the very first Streamlit command
st.set_page_config(page_title="AI Image Captioning", layout="wide")

st.title("🖼️ AI Image Captioning WebApp")
st.write("Generate captions for your images using AI")

device = "cuda" if torch.cuda.is_available() else "cpu"

# 2. Use st.cache_resource to prevent re-downloading on every click
@st.cache_resource
def load_model():
    # Direct ID for Hugging Face Hub - No local folder needed
    model_id = "Salesforce/blip-image-captioning-base"
    try:
        processor = BlipProcessor.from_pretrained(model_id)
        model = BlipForConditionalGeneration.from_pretrained(model_id)
        model = model.to(device)
        return processor, model, True
    except Exception as e:
        st.error(f"Failed to load model from Hub: {e}")
        return None, None, False

# 3. Trigger the load
processor, model, model_loaded = load_model()

# 4. Interface Logic
if model_loaded:
    st.success("✅ Model loaded successfully from Hugging Face Hub!")
    st.markdown("---")
    
    uploaded_file = st.sidebar.file_uploader("Choose an image...", type=["jpg","jpeg","png"])

    if uploaded_file is not None:
        image = Image.open(uploaded_file).convert("RGB")
        col1, col2 = st.columns(2)
        
        with col1:
            st.image(image, caption="Uploaded Image", use_container_width=True)
        
        with col2:
            st.subheader("Caption Options")
            text_input = st.text_input("Enter text prefix:", "a photography of")
            
            if st.button("🎯 Generate Conditional Caption", use_container_width=True):
                with st.spinner("Generating..."):
                    inputs = processor(image, text_input, return_tensors="pt").to(device)
                    out = model.generate(**inputs, max_length=50)
                    caption = processor.decode(out[0], skip_special_tokens=True)
                    st.info(f"**Result:** {caption}")

            if st.button("📸 Generate Unconditional Caption", use_container_width=True):
                with st.spinner("Generating..."):
                    inputs = processor(image, return_tensors="pt").to(device)
                    out = model.generate(**inputs, max_length=50)
                    caption = processor.decode(out[0], skip_special_tokens=True)
                    st.info(f"**Result:** {caption}")
else:
    st.error("❌ Model could not be initialized.")