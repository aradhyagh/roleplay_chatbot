# app.py

import os
import streamlit as st
from src.rag_pipeline import load_rag_pipeline, ask_character
from src.memory_faiss import SemanticMemory

st.set_page_config(page_title="Roleplay Chatbot", page_icon="✨", layout="centered")

# -----------------------------
# DEBUG: confirm memory_faiss path
import src.memory_faiss
st.write("✅ Using memory_faiss.py from:", src.memory_faiss.__file__)
# -----------------------------

# Load or build QA chain
if "qa_chain" not in st.session_state:
    try:
        st.session_state.qa_chain = load_rag_pipeline()
        st.success("✅ QA Chain loaded successfully")
    except Exception as e:
        st.error(f"❌ Error loading QA chain: {e}")
        st.stop()

# Load or initialize SemanticMemory
if "memory" not in st.session_state:
    try:
        st.session_state.memory = SemanticMemory()
        st.success("✅ SemanticMemory initialized successfully")
    except Exception as e:
        st.error(f"❌ Error initializing SemanticMemory: {e}")
        st.stop()  # Stop app to prevent cascading errors

# Initialize chat history
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

st.title("🧙 Roleplay Chatbot - Seraphina Nightbloom")

# Chat input
user_input = st.chat_input("Say something to Seraphina...")

if user_input:
    # Save user message
    st.session_state.memory.add_message("user", user_input)
    # Retrieve past context
    past_context = st.session_state.memory.retrieve_relevant(user_input, k=3)
    context_text = "\n".join(past_context)
    # Generate query with context
    query_with_context = f"{context_text}\nUser: {user_input}\nCharacter:"
    # Get response from QA chain
    response = ask_character(query_with_context, st.session_state.qa_chain)
    # Save character response
    st.session_state.memory.add_message("character", response)
    # Append to chat history
    st.session_state.chat_history.append(("user", user_input))
    st.session_state.chat_history.append(("character", response))

# Render chat history
for role, text in st.session_state.chat_history:
    if role == "user":
        with st.chat_message("user"):
            st.write(text)
    else:
        with st.chat_message("assistant"):
            st.write(text)
