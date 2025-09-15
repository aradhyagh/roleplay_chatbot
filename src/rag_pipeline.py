# src/rag_pipeline.py
import os
from langchain.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.chains import RetrievalQA
from src.embeddings import build_character_index

def load_llm():
    try:
        from langchain.chat_models import ChatOpenAI
        if os.getenv("OPENAI_API_KEY"):
            print("[INFO] Using OpenAI GPT-3.5")
            return ChatOpenAI(
                openai_api_key=os.getenv("OPENAI_API_KEY"),
                model="gpt-3.5-turbo",
                temperature=0.7,
                max_tokens=256
            )
    except Exception:
        pass

    # fallback to HuggingFace
    from langchain import HuggingFacePipeline
    from transformers import pipeline
    print("[INFO] Using HuggingFace flan-t5-base (local)")
    generator = pipeline("text2text-generation", model="google/flan-t5-base")
    return HuggingFacePipeline(pipeline=generator)

from src.embeddings import build_character_index

def load_rag_pipeline(index_path="memory/character_index"):
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

    # Always rebuild FAISS index (avoids missing file errors on Streamlit Cloud)
    print("[INFO] Rebuilding FAISS index on startup...")
    build_character_index(yaml_path="data/character.yaml", save_path=index_path)

    # Now load the rebuilt index
    print("[INFO] Loading FAISS index...")
    vectorstore = FAISS.load_local(index_path, embeddings, allow_dangerous_deserialization=True)
    print("[INFO] FAISS index loaded successfully!")

    llm = load_llm()
    qa = RetrievalQA.from_chain_type(
        llm=llm,
        retriever=vectorstore.as_retriever(),
        return_source_documents=True
    )
    return qa
