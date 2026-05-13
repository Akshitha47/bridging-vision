import streamlit as st
import importlib.util
import sys
import os

# Set page config once at the very top of the main file
st.set_page_config(page_title="AI Vision & Voice Hub", layout="wide")

st.sidebar.title("🚀 Navigation")
selection = st.sidebar.radio("Go to", ["Caption Generation (App 1)", "Voice Generation (App 2)"])

# Function to dynamically load and run your other files
def run_app(file_path):
    spec = importlib.util.spec_from_file_location("module.name", file_path)
    module = importlib.util.module_open(spec)
    spec.loader.exec_module(module)

if selection == "Caption Generation (App 1)":
    # Run original app.py logic
    # Note: You MUST remove st.set_page_config from app.py
    import app 

elif selection == "Voice Generation (App 2)":
    # Run app2.py logic
    # Note: You MUST remove st.set_page_config from app2.py
    import app2