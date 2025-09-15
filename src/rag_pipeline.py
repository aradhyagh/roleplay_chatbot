# src/rag_pipeline.py
from langchain.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.chains import RetrievalQA
import os
from src.embeddings import build_character_index

# OpenAI or HuggingFace fallback
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

    # Fallback to HuggingFace
    from langchain import HuggingFacePipeline
    from transformers import pipeline
    print("[INFO] Using HuggingFace flan-t5-base (local)")
    generator = pipeline("text2text-generation", model="google/flan-t5-base")
    return HuggingFacePipeline(pipeline=generator)

def load_rag_pipeline(index_path="memory/character_index"):
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

    # Ensure index exists. If not, build it.
    if not os.path.exists(index_path) or not os.path.exists(os.path.join(index_path, "index.faiss")):
        print(f"[INFO] FAISS index not found at {index_path}. Building it now.")
        build_character_index(yaml_path="data/character.yaml", save_path=index_path)

    try:
        vectorstore = FAISS.load_local(index_path, embeddings, allow_dangerous_deserialization=True)
    except Exception as e:
        # If load fails, attempt to rebuild once and try again
        print("[ERROR] Failed to load FAISS index. Attempting rebuild. Error:", e)
        build_character_index(yaml_path="data/character.yaml", save_path=index_path)
        vectorstore = FAISS.load_local(index_path, embeddings, allow_dangerous_deserialization=True)

    llm = load_llm()

    qa = RetrievalQA.from_chain_type(
        llm=llm,
        retriever=vectorstore.as_retriever(),
        return_source_documents=True
    )
    return qa

def ask_character(query, qa_chain):
    result = qa_chain({"query": query})
    return result["result"]
