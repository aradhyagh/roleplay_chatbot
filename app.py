# app.py

import streamlit as st
from src.rag_pipeline import load_rag_pipeline, ask_character
from src.memory_faiss import SemanticMemory

# ✅ Streamlit page config
st.set_page_config(
    page_title="Roleplay Chatbot",
    page_icon="✨",
    layout="centered"
)

# ✅ Initialize pipeline + memory
if "qa_chain" not in st.session_state:
    st.session_state.qa_chain = load_rag_pipeline()

if "memory" not in st.session_state:
    st.session_state.memory = SemanticMemory()

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# ✅ App Title
st.title("🧙 Roleplay Chatbot - Seraphina Nightbloom")

# ✅ Chat input
user_input = st.chat_input("Say something to Seraphina...")

if user_input:
    # Save user message
    st.session_state.memory.add_message("user", user_input)

    # Retrieve past relevant context
    past_context = st.session_state.memory.retrieve_relevant(user_input, k=3)
    context_text = "\n".join(past_context)

    # Ask character with context
    query_with_context = f"{context_text}\nUser: {user_input}\nCharacter:"
    response = ask_character(query_with_context, st.session_state.qa_chain)

    # Save response
    st.session_state.memory.add_message("character", response)
    st.session_state.chat_history.append(("user", user_input))
    st.session_state.chat_history.append(("character", response))

# ✅ Display conversation history
for role, text in st.session_state.chat_history:
    if role == "user":
        with st.chat_message("user"):
            st.write(text)
    else:
        with st.chat_message("assistant"):
            st.write(text)
