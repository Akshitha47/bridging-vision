import streamlit as st
import torch
from PIL import Image
from transformers import BlipProcessor, BlipForConditionalGeneration

# 1. Page Config (MUST be first)
st.set_page_config(page_title="AI Image Captioning", layout="wide")

# --- LOGIN LOGIC ---
def login():
    st.sidebar.title("🔐 User Login")
    username = st.sidebar.text_input("Username")
    password = st.sidebar.text_input("Password", type="password")
    
    if st.sidebar.button("Login"):
        # You can change these credentials for your presentation
        if username == "admin" and password == "bca2026":
            st.session_state['logged_in'] = True
            st.sidebar.success(f"Welcome, {username}!")
            st.rerun()
        else:
            st.sidebar.error("Invalid Username or Password")

# Initialize session state for login
if 'logged_in' not in st.session_state:
    st.session_state['logged_in'] = False

# Show Login Page if not logged in
if not st.session_state['logged_in']:
    st.title("🖼️ Bridging Vision & Language")
    st.info("Please login from the sidebar to access the AI Image Captioning Tool.")
    login()
    st.stop() # Prevents the rest of the code from running

# --- MAIN APP CODE (Only runs after login) ---
st.title("🖼️ AI Image Captioning WebApp")
st.sidebar.button("Logout", on_click=lambda: st.session_state.update({"logged_in": False}))

device = "cuda" if torch.cuda.is_available() else "cpu"

@st.cache_resource
def load_model():
    model_id = "Salesforce/blip-image-captioning-base"
    try:
        processor = BlipProcessor.from_pretrained(model_id)
        model = BlipForConditionalGeneration.from_pretrained(model_id)
        model = model.to(device)
        return processor, model, True
    except Exception as e:
        return None, None, False

processor, model, model_loaded = load_model()

if model_loaded:
    st.success("✅ Model loaded and Authenticated!")
    # ... (Your existing image upload and generation logic goes here) ...
    uploaded_file = st.sidebar.file_uploader("Choose an image...", type=["jpg","jpeg","png"])
    
    if uploaded_file:
        image = Image.open(uploaded_file).convert("RGB")
        st.image(image, caption="Uploaded Image", width=500)
        
        if st.button("Generate Caption"):
            inputs = processor(image, return_tensors="pt").to(device)
            out = model.generate(**inputs, max_length=50)
            caption = processor.decode(out[0], skip_special_tokens=True)
            st.info(f"**Generated Caption:** {caption}")