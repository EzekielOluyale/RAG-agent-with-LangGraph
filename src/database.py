from langchain_text_splitters import RecursiveCharacterTextSplitter

from langchain_google_genai import GoogleGenerativeAIEmbeddings

from langchain_pinecone import PineconeVectorStore

from pinecone import Pinecone

from src.utils import load_documents

def split_documents(documents):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=20,
        add_start_index=True,
        separators=["\n\n", "\n", " ", ""]
    )

    return text_splitter.split_documents(documents)

def get_embeddings():
    return GoogleGenerativeAIEmbeddings(
        model="text-embedding-004",
        vertexai=True
    )

def get_vector_store():
    embeddings = get_embeddings()

    pc = Pinecone()

    index = pc.Index("rag")

    return PineconeVectorStore(
        embedding=embeddings,
        index=index
    )

def ingest_documents():
    docs = load_documents()

    all_splits = split_documents(docs)

    vector_store = get_vector_store()

    vector_store.add_documents(all_splits, batch_size=1)

    return vector_store