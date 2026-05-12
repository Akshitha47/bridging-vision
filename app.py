import streamlit as st
import torch
from PIL import Image
from transformers import BlipProcessor, BlipForConditionalGeneration

# 1. Page Config (MUST be first)
st.set_page_config(page_title="AI Image Captioning", layout="wide")

# --- INITIALIZE SESSION STATE ---
if 'logged_in' not in st.session_state:
    st.session_state['logged_in'] = False

# --- LOGIN UI FUNCTION ---
def show_login_page():
    st.title("🔐 Project Login")
    st.write("Please enter your credentials to access the AI Image Captioning Tool.")
    
    # Using columns to center the login box
    col1, col2, col3 = st.columns([1,2,1])
    with col2:
        username = st.text_input("Username")
        password = st.text_input("Password", type="password")
        if st.button("Login", use_container_width=True):
            if username == "admin" and password == "bca2026":
                st.session_state['logged_in'] = True
                st.rerun()
            else:
                st.error("Invalid credentials. Please try again.")

# --- CHECK AUTHENTICATION ---
if not st.session_state['logged_in']:
    show_login_page()
    st.stop()  # Everything below this line is hidden until logged_in is True

# --- MAIN APP CODE (Only runs if logged_in is True) ---
st.title("🖼️ AI Image Captioning WebApp")
st.sidebar.button("Logout", on_click=lambda: st.session_state.update({"logged_in": False}))

# Load model logic
@st.cache_resource
def load_model():
    model_id = "Salesforce/blip-image-captioning-base"
    processor = BlipProcessor.from_pretrained(model_id)
    model = BlipForConditionalGeneration.from_pretrained(model_id)
    return processor, model.to("cpu")

processor, model = load_model()
st.success("Authenticated & Model Ready!")

# ... rest of your upload/generation code ...