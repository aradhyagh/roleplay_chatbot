# app.py

import os
import streamlit as st

# 👇 Build index first if it doesn't exist
from src.embeddings import build_character_index
if not os.path.exists("memory/character_index"):
    st.write("🛠️ Building character index...")
    build_character_index()

# 👇 Now import everything that depends on the index
from src.rag_pipeline import load_rag_pipeline, ask_character
from src.memory_faiss import SemanticMemory

import os

# Debug: Check if FAISS files exist
def debug_check_faiss_files():
    folder_path = "memory/character_index"
    st.write("🧾 Checking FAISS files in memory/character_index...")
    if os.path.exists(folder_path):
        files = os.listdir(folder_path)
        st.write("📁 Files found:", files)
        if "index.faiss" in files and "index.pkl" in files:
            st.success("✅ FAISS index is present and ready!")
        else:
            st.error("❌ FAISS index files missing! App will crash.")
    else:
        st.warning("⚠️ Folder memory/character_index does not exist.")
debug_check_faiss_files()


# Streamlit settings
st.set_page_config(page_title="Roleplay Chatbot", page_icon="✨", layout="centered")

# Load RAG pipeline + memory once
if "qa_chain" not in st.session_state:
    st.session_state.qa_chain = load_rag_pipeline()

if "memory" not in st.session_state:
    st.session_state.memory = SemanticMemory()

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# UI title
st.title("🧙 Roleplay Chatbot - Seraphina Nightbloom")

# Get user input
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

# Display conversation
for role, text in st.session_state.chat_history:
    if role == "user":
        with st.chat_message("user"):
            st.write(text)
    else:
        with st.chat_message("assistant"):
            st.write(text)
