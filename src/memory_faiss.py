# src/memory_faiss.py
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
import datetime
import os

class SemanticMemory:
    def __init__(self):
        self.embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

        # ✅ Create empty FAISS index with correct dimension (384 for MiniLM-L6-v2)
        dim = 384  
        index = faiss.IndexFlatL2(dim)
        self.db = FAISS(embedding_function=self.embeddings, index=index, docstore={}, index_to_docstore_id={})

    def add_message(self, role, content):
        text = f"{role}: {content}"
        self.db.add_texts([text])

    def retrieve_relevant(self, query, k=3):
        results = self.db.similarity_search(query, k=k)
        return [doc.page_content for doc in results]
