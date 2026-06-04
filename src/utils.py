from dotenv import load_dotenv

from langchain_community.document_loaders import (
    DirectoryLoader,
    PyPDFLoader
)

def setup_environment():
    load_dotenv()

def load_documents():
    loader = DirectoryLoader(
        "../data",
        glob="*.pdf",
        loader_cls=PyPDFLoader
    )

    return loader.load()