from langchain_community.document_loaders import DirectoryLoader, PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

class Loader:
    def __init__(self):
        DOCS_DIR = "data/raw"

        # 1. Upload documents
        self.docs = DirectoryLoader(DOCS_DIR, glob="*.pdf", loader_cls=PyPDFLoader).load()

        # 2. Split documents into chunks
        self.splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=100
        )
        self.chunks = self.splitter.split_documents(self.docs)