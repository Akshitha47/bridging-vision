import streamlit as st
import torch
import os
from PIL import Image

# --- PRE-CONFIG & UI STYLE ---
st.set_page_config(page_title="VisionAI - Premium Captioning", layout="wide", initial_sidebar_state="collapsed")

# Futuristic AI-Themed CSS
st.markdown("""
    <style>
    /* Main Background */
    .stApp {
        background: radial-gradient(circle at top right, #1a1a2e, #16213e, #0f3460);
        color: #ffffff;
    }
    
    /* Glassmorphism Cards */
    .main-card {
        background: rgba(255, 255, 255, 0.05);
        backdrop-filter: blur(10px);
        border-radius: 20px;
        padding: 30px;
        border: 1px solid rgba(255, 255, 255, 0.1);
        box-shadow: 0 8px 32px 0 rgba(0, 0, 0, 0.37);
        margin-bottom: 20px;
    }
    
    /* Premium Buttons */
    .stButton>button {
        background: linear-gradient(90deg, #4e54c8, #8f94fb);
        color: white;
        border: none;
        border-radius: 10px;
        padding: 10px 25px;
        transition: all 0.3s ease;
        font-weight: bold;
        text-transform: uppercase;
        letter-spacing: 1px;
    }
    .stButton>button:hover {
        transform: translateY(-2px);
        box-shadow: 0 5px 15px rgba(143, 148, 251, 0.4);
    }
    
    /* Typography */
    h1 {
        background: -webkit-linear-gradient(#fff, #8f94fb);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-weight: 800 !important;
    }
    </style>
    """, unsafe_allow_html=True)

# --- SESSION STATE INITIALIZATION ---
if 'users' not in st.session_state:
    st.session_state['users'] = {"admin@bca.com": "bca2026"} # Initial admin
if 'authenticated' not in st.session_state:
    st.session_state['authenticated'] = False

# --- AUTHENTICATION UI ---
def show_auth_page():
    st.markdown("<h1 style='text-align: center;'>VisionAI</h1>", unsafe_allow_html=True)
    st.markdown("<p style='text-align: center; color: #8f94fb;'>Bridging Human Perception with Artificial Intelligence</p>", unsafe_allow_html=True)
    
    auth_mode = st.tabs(["🔒 Login", "📝 Register"])
    
    with auth_mode[0]:
        with st.container():
            st.markdown('<div class="main-card">', unsafe_allow_html=True)
            login_email = st.text_input("Email Address", key="l_email")
            login_pass = st.text_input("Password", type="password", key="l_pass")
            if st.button("Access Dashboard", use_container_width=True):
                if login_email in st.session_state['users'] and st.session_state['users'][login_email] == login_pass:
                    st.session_state['authenticated'] = True
                    st.rerun()
                else:
                    st.error("Invalid credentials.")
            st.markdown('</div>', unsafe_allow_html=True)
            
    with auth_mode[1]:
        with st.container():
            st.markdown('<div class="main-card">', unsafe_allow_html=True)
            reg_email = st.text_input("Choose Email", key="r_email")
            reg_pass = st.text_input("Create Password", type="password", key="r_pass")
            confirm_pass = st.text_input("Confirm Password", type="password", key="r_confirm")
            
            if st.button("Create Account", use_container_width=True):
                if reg_pass != confirm_pass:
                    st.error("Passwords do not match.")
                elif reg_email in st.session_state['users']:
                    st.warning("Email already registered.")
                elif reg_email and reg_pass:
                    st.session_state['users'][reg_email] = reg_pass
                    st.success("Registration successful! Please login.")
            st.markdown('</div>', unsafe_allow_html=True)

# --- APP ROUTING ---
if not st.session_state['authenticated']:
    show_auth_page()
    st.stop()

# --- PREMIUM DASHBOARD ---
st.markdown("""
    <div style='display: flex; justify-content: space-between; align-items: center;'>
        <h1>VisionAI Dashboard</h1>
    </div>
    """, unsafe_allow_html=True)

if st.sidebar.button("Logout"):
    st.session_state['authenticated'] = False
    st.rerun()

# --- CORE LOGIC START (REMAINS UNCHANGED) ---
device = "cuda" if torch.cuda.is_available() else "cpu"

@st.cache_resource
def load_model():
    # Modified loading to use the Hub for cloud reliability
    model_id = "Salesforce/blip-image-captioning-base"
    try:
        from transformers import BlipProcessor, BlipForConditionalGeneration
        processor = BlipProcessor.from_pretrained(model_id)
        model = BlipForConditionalGeneration.from_pretrained(model_id)
        return processor, model.to(device), True
    except Exception as e:
        return None, None, False

# Premium Status Indicator
with st.container():
    st.markdown('<div class="main-card" style="padding: 15px; border-left: 5px solid #8f94fb;">', unsafe_allow_html=True)
    cols = st.columns([0.1, 0.9])
    processor, model, model_loaded = load_model()
    if model_loaded:
        cols[0].markdown("🟢")
        cols[1].write("**System Status:** AI Engine Online & Model Ready")
    else:
        cols[0].markdown("🔴")
        cols[1].write("**System Status:** Engine Offline - Check Logs")
    st.markdown('</div>', unsafe_allow_html=True)

# ... (Continue with your existing upload and generation code below)