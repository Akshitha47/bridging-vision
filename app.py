import streamlit as st
import torch
from PIL import Image
from transformers import BlipProcessor, BlipForConditionalGeneration

# 1. Streamlit page config MUST be first
st.set_page_config(page_title="AI Image Captioning", layout="wide")

st.title("🖼️ AI Image Captioning WebApp")
st.write("Generate captions for your images using AI")

# 2. Set device (Automatically use GPU if available)
device = "cuda" if torch.cuda.is_available() else "cpu"

# 3. Load the model directly from Hugging Face Hub
@st.cache_resource
def load_model():
    model_id = "Salesforce/blip-image-captioning-base"
    try:
        # This downloads the model from the internet instead of looking for a local folder
        processor = BlipProcessor.from_pretrained(model_id)
        model = BlipForConditionalGeneration.from_pretrained(model_id)
        model = model.to(device)
        return processor, model, True
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None, None, False

processor, model, model_loaded = load_model()

# 4. Main App Interface
if model_loaded:
    st.success("✅ Model loaded successfully!")
    st.markdown("---")
    
    # Sidebar for uploading
    uploaded_file = st.sidebar.file_uploader("Choose an image...", type=["jpg","jpeg","png"])

    if uploaded_file is not None:
        image = Image.open(uploaded_file).convert("RGB")
        col1, col2 = st.columns(2)
        
        with col1:
            st.image(image, caption="Uploaded Image", use_container_width=True)
        
        with col2:
            st.subheader("Caption Options")
            text_input = st.text_input("Enter text prefix:", "a photography of")
            
            # Conditional Caption Logic
            if st.button("🎯 Generate Conditional Caption", use_container_width=True):
                with st.spinner("Generating..."):
                    inputs = processor(image, text_input, return_tensors="pt").to(device)
                    out = model.generate(**inputs, max_length=50)
                    caption = processor.decode(out[0], skip_special_tokens=True)
                    st.info(f"**Result:** {caption}")

            # Unconditional Caption Logic
            if st.button("📸 Generate Unconditional Caption", use_container_width=True):
                with st.spinner("Generating..."):
                    inputs = processor(image, return_tensors="pt").to(device)
                    out = model.generate(**inputs, max_length=50)
                    caption = processor.decode(out[0], skip_special_tokens=True)
                    st.info(f"**Result:** {caption}")
else:
    st.error("❌ The app could not load the AI model. Check your internet connection or logs.")