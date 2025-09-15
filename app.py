# app.py

import os
import streamlit as st
from src.embeddings import build_character_index

# ✅ Step 1: Debug — list folder contents
def debug_check_faiss_files():
    st.subheader("🔍 FAISS File Debug Info")
    folder_path = "memory/character_index"
    if os.path.exists(folder_path):
        files = os.listdir(folder_path)
        st.write("📁 Found in memory/character_index:", files)
        if "index.faiss" in files and "index.pkl" in files:
            st.success("✅ FAISS index is present.")
        else:
            st.error("❌ FAISS files are missing. Attempting to rebuild...")
            build_character_index()
    else:
        st.warning("⚠️ Folder does not exist. Creating and building index...")
        build_character_index()

# Run debug function BEFORE anything else
debug_check_faiss_files()

# ✅ Step 2: Now load the chatbot pipeline
from src.rag_pipeline import load_rag_pipeline, ask_character
from src.memory_faiss import SemanticMemory

st.set_page_config(page_title="Roleplay Chatbot", page_icon="✨", layout="centered")

if "qa_chain" not in st.session_state:
    st.session_state.qa_chain = load_rag_pipeline()

if "memory" not in st.session_state:
    st.session_state.memory = SemanticMemory()

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

st.title("🧙 Roleplay Chatbot - Seraphina Nightbloom")

user_input = st.chat_input("Say something to Seraphina...")

if user_input:
    st.session_state.memory.add_message("user", user_input)
    past_context = st.session_state.memory.retrieve_relevant(user_input, k=3)
    context_text = "\n".join(past_context)

    query_with_context = f"{context_text}\nUser: {user_input}\nCharacter:"
    response = ask_character(query_with_context, st.session_state.qa_chain)

    st.session_state.memory.add_message("character", response)
    st.session_state.chat_history.append(("user", user_input))
    st.session_state.chat_history.append(("character", response))

for role, text in st.session_state.chat_history:
    if role == "user":
        with st.chat_message("user"):
            st.write(text)
    else:
        with st.chat_message("assistant"):
            st.write(text)
