import streamlit as st
import os

# MUST be the first Streamlit command
st.set_page_config(page_title="AI Vision & Voice Hub", layout="wide")

st.sidebar.title("🚀 Navigation")
selection = st.sidebar.radio("Go to", ["Caption Generation (App 1)", "Voice Generation (App 2)"])

def run_app(file_path):
    # This reads the file and executes it in the current context
    with open(file_path, encoding='utf-8') as f:
        code = f.read()
    exec(code, globals())

if selection == "Caption Generation (App 1)":
    run_app("app.py")

elif selection == "Voice Generation (App 2)":
    run_app("app2.py")