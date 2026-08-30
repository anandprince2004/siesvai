from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from backend.rag_pipeline import (
    load_groq_client,
    load_vector_db,
    answer_question,
)

app_state = {}

@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Runs once when the server starts (before 'yield') and once when it
    shuts down (after 'yield'). This is the modern replacement for the
    deprecated @app.on_event("startup") decorator.
    """
    print("Loading vector database and embedding model...")
    embedding_model, collection = load_vector_db()
    print(f"Loaded collection with {collection.count()} chunks.")

    print("Connecting to Groq API...")
    groq_client = load_groq_client()

    app_state["embedding_model"] = embedding_model
    app_state["collection"] = collection
    app_state["groq_client"] = groq_client

    print("SIESVAI API is ready.")
    yield
    app_state.clear()

app = FastAPI(
    title="SIESVAI API",
    description="Multilingual RAG chatbot API for SIES (Nerul) College",
    version="0.1.0",
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
    message: str

class ChatResponse(BaseModel):
    answer: str
    translated_query: str

@app.get("/")
def root():
    """Simple health check endpoint."""
    return {"status": "ok", "service": "SIESVAI API"}

@app.get("/health")
def health():
    """
    Health check that also confirms the vector DB and Groq client loaded
    successfully. Useful for deployment platforms that ping this to check
    if the service is alive.
    """
    collection = app_state.get("collection")
    return {
        "status": "ok",
        "chunks_loaded": collection.count() if collection else 0,
    }

@app.post("/chat", response_model=ChatResponse)
def chat(request: ChatRequest):
    """
    Main chat endpoint. Takes a user message (in English, Hindi, Marathi,
    or Tamil, possibly Romanized) and returns a grounded answer in the
    same language.
    """
    message = request.message.strip()

    if not message:
        raise HTTPException(status_code=400, detail="Message cannot be empty.")

    groq_client = app_state.get("groq_client")
    embedding_model = app_state.get("embedding_model")
    collection = app_state.get("collection")

    if not groq_client or not collection:
        raise HTTPException(
            status_code=503,
            detail="Service is still starting up. Please try again in a moment.",
        )

    try:
        result = answer_question(groq_client, embedding_model, collection, message)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error generating answer: {e}")

    return ChatResponse(
        answer=result["answer"],
        translated_query=result["translated_query"],
    )
