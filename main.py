import streamlit as st
from auth import register_user, login_user, logout
from app import run_app1
from app2 import run_app2

st.set_page_config(page_title="Bridging Vision", layout="wide")

if "authenticated" not in st.session_state:
    st.session_state.authenticated = False
if "user_email" not in st.session_state:
    st.session_state.user_email = None
if "page" not in st.session_state:
    st.session_state.page = "home"

def auth_page():
    st.markdown("<h1 style='text-align:center;'>🌉 Bridging Vision</h1>", unsafe_allow_html=True)
    st.markdown("<p style='text-align:center;'>AI Image Captioning and Voice Assistant</p>", unsafe_allow_html=True)

    tab1, tab2 = st.tabs(["Sign In", "Sign Up"])

    with tab1:
        with st.form("login_form"):
            email = st.text_input("Email")
            password = st.text_input("Password", type="password")
            submitted = st.form_submit_button("Sign In")
            if submitted:
                if login_user(email, password):
                    st.session_state.authenticated = True
                    st.session_state.user_email = email
                    st.success("Login successful")
                    st.rerun()
                else:
                    st.error("Invalid email or password")

    with tab2:
        with st.form("signup_form"):
            email = st.text_input("Register Email")
            password = st.text_input("Register Password", type="password")
            submitted = st.form_submit_button("Sign Up")
            if submitted:
                ok, msg = register_user(email, password)
                if ok:
                    st.success(msg)
                else:
                    st.error(msg)

def dashboard():
    st.sidebar.success(f"Logged in as {st.session_state.user_email}")
    if st.sidebar.button("Logout"):
        logout()
        st.rerun()

    st.title("🚀 AI Vision Dashboard")
    choice = st.radio("Choose Feature", ["Caption Generation", "Caption + Voice"])

    if choice == "Caption Generation":
        run_app1()
    else:
        run_app2()

if not st.session_state.authenticated:
    auth_page()
else:
    dashboard()