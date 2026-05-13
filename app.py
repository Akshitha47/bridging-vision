import streamlit as st
import torch
from PIL import Image

def run_app1():
    st.title("🖼️ AI Image Captioning WebApp")
    st.write("Generate captions for your images using AI")

    device = "cuda" if torch.cuda.is_available() else "cpu"

    processor = None
    model = None
    model_loaded = False

    try:
        from transformers import BlipProcessor, BlipForConditionalGeneration

        with st.spinner("Downloading and loading model..."):
            MODEL_ID = "Salesforce/blip-image-captioning-base"
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

        uploaded_file = st.sidebar.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"], key="app1_uploader")

        if uploaded_file is not None:
            image = Image.open(uploaded_file).convert("RGB")
            col1, col2 = st.columns(2)

            with col1:
                st.image(image, caption="Uploaded Image", use_container_width=True)

            with col2:
                st.subheader("Caption Options")
                text_input = st.text_input("Enter text prefix for captioning:", "a photography of", key="app1_text")

                if st.button("🎯 Generate Conditional Caption", use_container_width=True, key="app1_cond"):
                    try:
                        with st.spinner("Generating caption..."):
                            inputs = processor(image, text_input, return_tensors="pt").to(device)
                            with torch.no_grad():
                                out = model.generate(**inputs, max_length=50)
                                out = model.generate(
                                     **inputs, 
                                      max_length=50, 
                                      num_beams=5, 
                                      repetition_penalty=1.5,
                                      length_penalty=1.0,
                                      temperature=0.7
                                    )
                            caption = processor.decode(out[0], skip_special_tokens=True)
                            st.success("✅ Caption Generated!")
                            st.info(f"Caption: {caption}")
                    except Exception as e:
                        st.error(f"Error generating caption: {str(e)}")

                if st.button("📸 Generate Unconditional Caption", use_container_width=True, key="app1_uncond"):
                    try:
                        with st.spinner("Generating caption..."):
                            inputs = processor(image, return_tensors="pt").to(device)
                            with torch.no_grad():
                                out = model.generate(**inputs, max_length=50)
                            caption = processor.decode(out[0], skip_special_tokens=True)
                            st.success("✅ Caption Generated!")
                            st.info(f"Caption: {caption}")
                    except Exception as e:
                        st.error(f"Error generating caption: {str(e)}")
    else:
        st.markdown("---")
        st.info("📝 Demo Mode - Model Not Available")