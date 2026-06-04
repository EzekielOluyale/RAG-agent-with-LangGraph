from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException, Request
from pydantic import BaseModel
from langchain_core.messages import HumanMessage

from src.utils import setup_environment
from src.database import get_vector_store
from src.agent import build_agent

class ChatRequest(BaseModel):
    message: str

class ChatResponse(BaseModel):
    answer: str

@asynccontextmanager
async def lifespan(app: FastAPI):
    setup_environment()
    
    print("📦 Connecting to Pinecone vector store...")
    vector_store = get_vector_store()
    
    print("🧠 Compiling RAG reasoning agent...")
    app.state.agent = build_agent(vector_store)
    
    print("🚀 Application started successfully and listening for traffic.")
    yield
    print("🛑 Application shutting down...")

app = FastAPI(
    title="RAG Agent API",
    description="LangChain + Gemini + Pinecone",
    version="1.0.0",
    lifespan=lifespan,
)

@app.get("/")
async def root():
    return {"message": "RAG Agent Running"}

@app.get("/health")
async def health_check():
    return {"status": "healthy"}

@app.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest, fastapi_req: Request):
    if not request.message.strip():
        raise HTTPException(status_code=400, detail="Message cannot be empty.")
        
    try:
        agent = fastapi_req.app.state.agent
        
        response = agent.invoke({
            "messages": [HumanMessage(content=request.message)]
        })
        
        answer = response["messages"][-1].content
        
        return ChatResponse(answer=answer)
        
    except Exception as e:
        print(f"CRITICAL API ERROR: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail="An error occurred while the agent was generating a response."
        )