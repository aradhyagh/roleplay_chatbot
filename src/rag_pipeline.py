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

def load_rag_pipeline(index_path="memory/character_index"):
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

    index_file = os.path.join(index_path, "index.faiss")
    pkl_file = os.path.join(index_path, "index.pkl")

    print("[DEBUG] Checking FAISS index at:", index_path)
    print("[DEBUG] index.faiss exists?", os.path.exists(index_file))
    print("[DEBUG] index.pkl exists?", os.path.exists(pkl_file))

    if not (os.path.exists(index_file) and os.path.exists(pkl_file)):
        print("[INFO] No FAISS index found. Building a new one...")
        build_character_index(yaml_path="data/character.yaml", save_path=index_path)
        print("[INFO] Finished building index.")

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
