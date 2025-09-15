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

    # Try to build (or re-build) the vectorstore and use it directly
    print("[INFO] Building/refreshing FAISS vectorstore from YAML (avoids loading pickles).")
    vectorstore = build_character_index(yaml_path="data/character.yaml", save_path=index_path)

    # If build_character_index returns a vectorstore, use it directly
    if vectorstore is None:
        # fallback: try to load local index (with allow flag)
        print("[WARN] build_character_index returned None, trying to load saved index.")
        vectorstore = FAISS.load_local(index_path, embeddings, allow_dangerous_deserialization=True)

    llm = load_llm()

    qa = RetrievalQA.from_chain_type(
        llm=llm,
        retriever=vectorstore.as_retriever(),
        return_source_documents=True
    )
    return qa

