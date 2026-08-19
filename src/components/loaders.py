from langchain_community.document_loaders import DirectoryLoader, PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from pathlib import Path

class Loader:
    def __init__(self):
        PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
        DOCS_DIR = PROJECT_ROOT / "data" / "raw"

        # 1. Upload documents
        if not DOCS_DIR.exists():
            raise FileNotFoundError(f"Diretório de documentos não encontrado: {DOCS_DIR}")

        self.docs = DirectoryLoader(str(DOCS_DIR), glob="*.pdf", loader_cls=PyPDFLoader).load()

        if not self.docs:
            raise ValueError(f"Nenhum PDF encontrado em {DOCS_DIR}")

        for doc in self.docs:
            filename = Path(doc.metadata["source"]).stem
            doc.metadata["categoria"] = filename
            doc.metadata["documento"] = filename

        # 2. Split documents into chunks
        self.splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=100
        )
        self.chunks = self.splitter.split_documents(self.docs)