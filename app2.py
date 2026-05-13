import streamlit as st
import torch
from PIL import Image
from datetime import datetime
import tempfile
import base64

def create_audio_player(text, language):
    try:
        from gtts import gTTS
        tts = gTTS(text=text, lang=language)

        with tempfile.NamedTemporaryFile(delete=False, suffix=".mp3") as fp:
            tts.save(fp.name)
            with open(fp.name, "rb") as audio_file:
                audio_bytes = audio_file.read()

        audio_base64 = base64.b64encode(audio_bytes).decode()
        return f"""
            <audio controls style="width: 100%;">
                <source src="data:audio/mp3;base64,{audio_base64}" type="audio/mp3">
                Your browser does not support the audio element.
            </audio>
        """
    except Exception as e:
        return f"<p>Error generating audio: {str(e)}</p>"

def run_app2():
    st.title("🖼️ AI Image Caption Generator with Voice")
    st.write("Upload an image and get AI-generated captions with multi-language voice output")

    if "language" not in st.session_state:
        st.session_state.language = "en"
    if "history" not in st.session_state:
        st.session_state.history = []

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
        st.warning(f"⚠️ Model loading issue: {str(e)}")
        model_loaded = False

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
            }[x],
            key="app2_language"
        )

        st.markdown("## 📜 Caption History")
        if st.session_state.history:
            for i, (caption, time) in enumerate(reversed(st.session_state.history)):
                with st.expander(f"#{len(st.session_state.history)-i} ({time})"):
                    st.write(caption)
        else:
            st.info("No captions yet")

        if st.button("🗑️ Clear History", key="app2_clear"):
            st.session_state.history = []
            st.rerun()

    if model_loaded:
        st.markdown("---")
        col1, col2 = st.columns([1, 1.5])

        with col1:
            uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"], key="app2_uploader")

        if uploaded_file is not None:
            image = Image.open(uploaded_file).convert("RGB")

            with col1:
                st.image(image, caption="Uploaded Image", use_container_width=True)

            with col2:
                tab1, tab2 = st.tabs(["Conditional", "Unconditional"])

                with tab1:
                    text_input = st.text_input("Enter text prefix:", value="a photography of", key="app2_text")

                    if st.button("🎯 Generate Conditional Caption", use_container_width=True, key="app2_cond"):
                        try:
                            with st.spinner("Generating caption..."):
                                inputs = processor(image, text_input, return_tensors="pt").to(device)
                                with torch.no_grad():
                                    out = model.generate(**inputs, max_length=50)
                                caption = processor.decode(out[0], skip_special_tokens=True)
                                st.session_state.history.append((caption, datetime.now().strftime("%H:%M:%S")))
                                st.session_state.history = st.session_state.history[-10:]
                                st.success("✅ Caption Generated!")
                                st.info(f"📝 {caption}")
                                st.markdown(create_audio_player(caption, st.session_state.language), unsafe_allow_html=True)
                        except Exception as e:
                            st.error(f"Error: {str(e)}")

                with tab2:
                    if st.button("🎲 Generate Unconditional Caption", use_container_width=True, key="app2_uncond"):
                        try:
                            with st.spinner("Generating caption..."):
                                inputs = processor(image, return_tensors="pt").to(device)
                                with torch.no_grad():
                                    out = model.generate(**inputs, max_length=50)
                                caption = processor.decode(out[0], skip_special_tokens=True)
                                st.session_state.history.append((caption, datetime.now().strftime("%H:%M:%S")))
                                st.session_state.history = st.session_state.history[-10:]
                                st.success("✅ Caption Generated!")
                                st.info(f"📝 {caption}")
                                st.markdown(create_audio_player(caption, st.session_state.language), unsafe_allow_html=True)
                        except Exception as e:
                            st.error(f"Error: {str(e)}")
    else:
        st.info("📝 Demo Mode - Model Not Available")