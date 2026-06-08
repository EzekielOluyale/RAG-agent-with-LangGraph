import os
import logging
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
    index_name = os.getenv("PINECONE_INDEX")
    index = pc.Index(index_name)

    return PineconeVectorStore(
        embedding=embeddings,
        index=index
    )