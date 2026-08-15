from components.vector_store import VectorStore
from dotenv import load_dotenv
import os

load_dotenv()

def main():
    # 1. Initializes vector store
    vector_store = VectorStore()
    vector_store.create()

    document_count = vector_store.count()
    if document_count == 0:
        vector_store.load()

    # 2. Build retrieval of a document
    retriever = vector_store.get_retriever()
    retriever.invoke("Garantia do produto")
    
if __name__ == "__main__":
    main()