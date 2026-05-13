import streamlit as st
from app import run_app1
from app2 import run_app2

st.set_page_config(page_title="AI Vision & Voice Hub", layout="wide")

st.sidebar.title("🚀 Navigation")
selection = st.sidebar.radio(
    "Go to",
    ["Caption Generation (App 1)", "Voice Generation (App 2)"]
)

if selection == "Caption Generation (App 1)":
    run_app1()
else:
    run_app2()