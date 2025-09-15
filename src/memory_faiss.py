# src/memory_faiss.py
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
import datetime
import os

class SemanticMemory:
    def __init__(self, path="memory/faiss_memory"):
        os.makedirs("memory", exist_ok=True)
        self.embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
        self.path = path

        if os.path.exists(path):
            try:
                # use the safe flag so pickled data can be deserialized
                self.db = FAISS.load_local(path, self.embeddings, allow_dangerous_deserialization=True)
            except Exception as e:
                print("[WARN] Could not load existing FAISS memory at", path, "— creating empty DB. Error:", e)
                self.db = FAISS.from_texts([], self.embeddings)
        else:
            self.db = FAISS.from_texts([], self.embeddings)

    def add_message(self, role, content):
        timestamp = datetime.datetime.now().isoformat()
        metadata = {"role": role, "timestamp": timestamp}
        self.db.add_texts([content], metadatas=[metadata])
        self.db.save_local(self.path)

    def retrieve_relevant(self, query, k=3):
        try:
            if not getattr(self.db, "index", None) or not getattr(self.db.index, "ntotal", 0):
                return []
        except Exception:
            # fallback safe behavior
            return []

        docs = self.db.similarity_search(query, k=k)
        return [d.page_content for d in docs]
