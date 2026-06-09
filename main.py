import src.logger  
import logging
import os

from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException, Request
from pydantic import BaseModel, Field
from fastapi.middleware.cors import CORSMiddleware
from langchain_core.messages import HumanMessage

from src.utils import setup_environment
from src.database import get_vector_store
from src.agent import build_agent

from langgraph.checkpoint.postgres import PostgresSaver
from fastapi.concurrency import run_in_threadpool
from psycopg_pool import ConnectionPool

logger = logging.getLogger(__name__)

@asynccontextmanager
async def lifespan(app: FastAPI):
    setup_environment()
    
    logger.info("Connecting to Pinecone vector store...")
    vector_store = get_vector_store()
    
    DB_URI = os.getenv("DATABASE_URL")
    if not DB_URI:
        logger.error("DATABASE_URL environment variable is missing!")
        raise ValueError("DATABASE_URL environment variable is missing!")

    pool = ConnectionPool(
        DB_URI, 
        kwargs={"autocommit": True, "prepare_threshold": None}
    )
    checkpointer = PostgresSaver(pool)
    checkpointer.setup() 
    logger.info("Database checkpointer tables verified successfully.")

    app.state.agent = build_agent(vector_store=vector_store, checkpointer=checkpointer)
    app.state.db_pool = pool
    
    logger.info("Application started successfully and listening for traffic.")
    yield
    logger.info("Application shutting down...")
    app.state.db_pool.close()

app = FastAPI(
    title="RAG Agent API",
    description="LangChain + Gemini + Pinecone",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class ChatRequest(BaseModel):
    message: str = Field(..., description="The message sent by the user.")
    thread_id: str = Field(..., description="Unique identifier tracking this specific conversation session.")

class ChatResponse(BaseModel):
    answer: str = Field(..., description="The final verified response from the agent.")

@app.get("/")
async def root():
    return {"message": "RAG Agent Running"}

@app.get("/health")
async def health_check():
    agent_ready = hasattr(app.state, "agent") and app.state.agent is not None
    
    db_ready = hasattr(app.state, "db_pool") and not app.state.db_pool.closed
    
    if not agent_ready or not db_ready:
        raise HTTPException(
            status_code=503, 
            detail={"status": "unhealthy", "pinecone": agent_ready, "database": db_ready}
        )

    return {
        "status": "healthy",
        "pinecone": True,
        "agent": True
    }

@app.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    if not request.message.strip():
        raise HTTPException(status_code=400, detail="Message cannot be empty.")
        
    log_prefix = f"[Thread: {request.thread_id}]"

    try:
        agent = app.state.agent 
        config = {"configurable": {"thread_id": request.thread_id}}

        logger.info(f"{log_prefix} Processing chat request.")
        
        response = await run_in_threadpool(
            agent.invoke,
            {"messages": [HumanMessage(content=request.message)]},
            config
        )
        
        answer = response["messages"][-1].content
        
        return ChatResponse(answer=answer)
        
    except Exception as e:
        logger.error(f"{log_prefix} CRITICAL API ERROR: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail="An error occurred while the agent was generating a response."
        )