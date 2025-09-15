from langchain.vectorstores import FAISS
from langchain.embeddings.openai import OpenAIEmbeddings

class SemanticMemory:
    def __init__(self):
        # Initialize the embeddings object
        self.embeddings = OpenAIEmbeddings()

        # Define some sample texts for your roleplay character memory
        sample_texts = [
            "I am a friendly wizard who loves helping people with magical advice.",
            "I enjoy telling stories of ancient kingdoms and magical creatures.",
            "My goal is to assist you in solving problems using my mystical knowledge."
        ]

        # Initialize FAISS vector store with sample texts and embeddings
        self.db = FAISS.from_texts(sample_texts, self.embeddings)

    def add_text(self, text):
        # Add new roleplay dialogue or context to the FAISS memory
        self.db.add_texts([text])

    def retrieve_similar_texts(self, query, k=3):
        # Retrieve the top-k similar texts from memory
        docs = self.db.similarity_search(query, k=k)
        return [doc.page_content for doc in docs]
