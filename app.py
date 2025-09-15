# app.py
import os
import streamlit as st

# Ensure we can build index BEFORE importing modules that may load it
from src.embeddings import build_character_index

st.set_page_config(page_title="Roleplay Chatbot", page_icon="✨", layout="centered")

INDEX_PATH = "memory/character_index"

# Build index if missing (this avoids load-time crashes on Streamlit)
if not os.path.exists(INDEX_PATH) or not os.path.exists(os.path.join(INDEX_PATH, "index.faiss")):
    st.info("⚠️ 'character_index' not found. Building it now (this may take a few seconds)...")
    build_character_index(yaml_path="data/character.yaml", save_path=INDEX_PATH)
    st.success("✅ Done building index!")

# debug — show what's in the folder (helpful for logs)
st.subheader("🔍 FAISS debug info")
if os.path.exists(INDEX_PATH):
    st.write("Files in memory/character_index:", os.listdir(INDEX_PATH))
else:
    st.write("memory/character_index folder does not exist (unexpected).")

# Now import and initialize pipeline + memory
from src.rag_pipeline import load_rag_pipeline, ask_character
from src.memory_faiss import SemanticMemory

if "qa_chain" not in st.session_state:
    try:
        st.session_state.qa_chain = load_rag_pipeline(index_path=INDEX_PATH)
    except Exception as e:
        st.error("Failed to initialize QA pipeline. Check the app logs (Manage app → Logs).")
        # Print to logs for debugging
        print("Error initializing QA pipeline:", e)

if "memory" not in st.session_state:
    try:
        st.session_state.memory = SemanticMemory()
    except Exception as e:
        st.error("Failed to initialize semantic memory. Check the app logs.")
        print("Error initializing SemanticMemory:", e)

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
