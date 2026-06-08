import os
import logging
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_pinecone import PineconeVectorStore
from pinecone import Pinecone
from langgraph.checkpoint.postgres import PostgresSaver

logger = logging.getLogger(__name__)

def get_embeddings():
    return GoogleGenerativeAIEmbeddings(
        model="text-embedding-004",
        vertexai=True
    )

def get_vector_store():
    logger.info("Initializing Pinecone connection...")
    embeddings = get_embeddings()
    pc = Pinecone()
    index = pc.Index("rag")
    return PineconeVectorStore(
        embedding=embeddings,
        index=index
    )

def get_checkpointer():
    DB_URI = os.getenv("DATABASE_URL")
    if not DB_URI:
        logger.error("DATABASE_URL environment variable is missing!")
        raise ValueError("DATABASE_URL environment variable is missing!")
    
    logger.info("Connecting to Supabase checkpointer database...")
    checkpointer = PostgresSaver.from_conn_string(DB_URI)
    checkpointer.setup() 
    logger.info("Database checkpointer tables verified successfully.")
    return checkpointer