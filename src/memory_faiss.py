# src/memory_faiss.py

import os
import datetime
import faiss
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

class SemanticMemory:
    def __init__(self, path="memory/faiss_memory"):
        os.makedirs("memory", exist_ok=True)
        self.embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
        self.path = path

        if os.path.exists(path):
            try:
                # ✅ Load existing FAISS memory
                self.db = FAISS.load_local(path, self.embeddings, allow_dangerous_deserialization=True)
            except Exception:
                # ✅ If corrupted, rebuild a clean empty index
                self.db = self._create_empty_index()
        else:
            # ✅ First run → start with clean empty index
            self.db = self._create_empty_index()

    def _create_empty_index(self):
        """Create an empty FAISS index with correct embedding dimension (384 for MiniLM-L6-v2)."""
        dim = 384  # embedding size for all-MiniLM-L6-v2
        index = faiss.IndexFlatL2(dim)
        return FAISS(embedding_function=self.embeddings, index=index, docstore={}, index_to_docstore_id={})

    def add_message(self, role, content):
        timestamp = datetime.datetime.now().isoformat()
        text = f"[{role.upper()} @ {timestamp}] {content}"
        self.db.add_texts([text])
        self.db.save_local(self.path)

    def retrieve_relevant(self, query, k=3):
        if not self.db.index.ntotal:
            return []
        docs = self.db.similarity_search(query, k=k)
        return [doc.page_content for doc in docs]
