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

# rag_pipeline.py

import os
import yaml
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings
from langchain.chains import RetrievalQA
from langchain_openai import ChatOpenAI
from src.ingest import load_documents, split_documents

def load_rag_pipeline():
    index_path = "data/faiss_index"
    embeddings = OpenAIEmbeddings()

    try:
        # ✅ Try loading existing index
        vectorstore = FAISS.load_local(index_path, embeddings, allow_dangerous_deserialization=True)
    except Exception as e:
        print(f"[INFO] Could not load FAISS index ({e}). Rebuilding...")

        # ✅ Rebuild index from YAML
        docs = load_documents("data/character.yaml")
        chunks = split_documents(docs)
        vectorstore = FAISS.from_documents(chunks, embeddings)

        # ✅ Save new index
        os.makedirs(index_path, exist_ok=True)
        vectorstore.save_local(index_path)

    retriever = vectorstore.as_retriever(search_kwargs={"k": 3})
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.7)

    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        retriever=retriever,
        return_source_documents=False
    )

    return qa_chain
