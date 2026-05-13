import streamlit as st
import json
import os
import hashlib

USERS_FILE = "users.json"

def hash_password(password):
    return hashlib.sha256(password.encode()).hexdigest()

def load_users():
    if os.path.exists(USERS_FILE):
        with open(USERS_FILE, "r") as f:
            return json.load(f)
    return {}

def save_users(users):
    with open(USERS_FILE, "w") as f:
        json.dump(users, f)

def register_user(email, password):
    users = load_users()
    if email in users:
        return False, "User already exists"
    users[email] = hash_password(password)
    save_users(users)
    return True, "Registration successful"

def login_user(email, password):
    users = load_users()
    if email in users and users[email] == hash_password(password):
        return True
    return False

def logout():
    st.session_state.authenticated = False
    st.session_state.user_email = None