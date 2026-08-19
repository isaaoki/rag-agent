from components.loaders import Loader
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEndpointEmbeddings
import os

class VectorStore:
    EMBEDDING_MODEL = "sentence-transformers/paraphrase-multilingual-mpnet-base-v2"
    COLLECTION_NAME = "company_docs"
    PERSIST_DIR = "data/chroma"

    def __init__(self):
        if not os.getenv("HF_TOKEN"):
            raise ValueError("Missing value of HF_TOKEN")

        self.embeddings = HuggingFaceEndpointEmbeddings(
            model=self.EMBEDDING_MODEL,
            huggingfacehub_api_token=os.getenv("HF_TOKEN")
        )

    def create(self):
        self.loader = Loader()

        self.vectorstore = Chroma.from_documents(
            documents=self.loader.chunks,
            embedding=self.embeddings,
            collection_name=self.COLLECTION_NAME,
            persist_directory=self.PERSIST_DIR,
            collection_metadata={"hnsw:space": "cosine"}
        )

    def load(self):
        self.vectorstore = Chroma(
            collection_name=self.COLLECTION_NAME,
            embedding_function=self.embeddings,
            persist_directory=self.PERSIST_DIR,
            collection_metadata={"hnsw:space": "cosine"}
        )

    def get_retriever(self):
        return self.vectorstore.as_retriever(
            search_kwargs={"k": 3}
        )

    def count(self):
        return len(self.vectorstore.get()['documents'])

    def similarity_search_with_relevance_scores(self, query, k=5, filter=None):
        return self.vectorstore.similarity_search_with_relevance_scores(
            query, k=k, filter=filter
        )