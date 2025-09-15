# src/rag_pipeline.py

import os
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
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
    except Exception as e:
        print("[WARN] ChatOpenAI not available or no API key:", e)

    from langchain import HuggingFacePipeline
    from transformers import pipeline
    print("[INFO] Using HuggingFace flan-t5-base (local)")
    generator = pipeline("text2text-generation", model="google/flan-t5-base")
    return HuggingFacePipeline(pipeline=generator)

def load_rag_pipeline(index_path="memory/character_index"):
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

    # Always rebuild the index from YAML to avoid missing/corrupted files
    print("[INFO] Rebuilding FAISS index from character.yaml")
    vectorstore = build_character_index(yaml_path="data/character.yaml", save_path=index_path)

    # Optionally, you could try loading saved one, but skip to avoid error
    # If you prefer load_local fallback instead:
    # try:
    #     vectorstore = FAISS.load_local(index_path, embeddings, allow_dangerous_deserialization=True)
    # except Exception as e:
    #     print("[WARN] load_local failed:", e)
    #     vectorstore = build_character_index(yaml_path="data/character.yaml", save_path=index_path)

    qa = RetrievalQA.from_chain_type(
        llm=load_llm(),
        retriever=vectorstore.as_retriever(),
        return_source_documents=True
    )
    return qa

def ask_character(query, qa_chain):
    result = qa_chain({"query": query})
    return result["result"]
