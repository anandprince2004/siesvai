"""
SIESVAI - Step 4: RAG Chatbot (Retrieval + LLM)
Combines vector DB retrieval with a free Groq LLM call to produce natural,
grounded answers to user questions about SIES (Nerul) College.

Usage:
    python scripts/ask_chatbot.py

Requirements:
    - GROQ_API_KEY set in a .env file at the project root
    - vectorstore/ already built (run build_vector_db.py first)

How it works:
    1. User types a question.
    2. The question is embedded and the top-k most relevant chunks are
       retrieved from ChromaDB (same as test_retrieval.py).
    3. Those chunks are inserted into a prompt as "context".
    4. The Groq LLM is asked to answer ONLY using that context, so the
       chatbot doesn't hallucinate facts about the college.
"""

import os
import chromadb
from sentence_transformers import SentenceTransformer
from dotenv import load_dotenv
from groq import Groq

BASE_DIR = os.path.join(os.path.dirname(__file__), "..")
VECTOR_DB_DIR = os.path.join(BASE_DIR, "vectorstore")
COLLECTION_NAME = "siesvai_knowledge_base"
EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2"
TOP_K = 3
GROQ_MODEL = "qwen/qwen3.6-27b"

SYSTEM_PROMPT = """You are SIESVAI, a helpful assistant for SIES (Nerul) \
College of Arts, Science and Commerce (Autonomous), Navi Mumbai.

Answer the user's question using ONLY the information given in the \
"Context" section below. Do not use any outside knowledge about this or \
any other college.

Rules:
- If the context does not contain enough information to answer, say so \
honestly and suggest the user contact the college directly or check the \
official website (https://siesascn.edu.in). Do NOT guess or make up \
information.
- Keep answers clear, concise, and friendly.
- Respond in the same language the user asked the question in.
- Give your final answer directly. Do not show your internal reasoning \
or thinking process in the response.
"""

def load_env_and_client() -> Groq:
    """Load .env and create the Groq API client."""
    load_dotenv()
    api_key = os.getenv("GROQ_API_KEY")

    if not api_key:
        raise RuntimeError(
            "GROQ_API_KEY not found. Make sure you have a .env file at the "
            "project root with: GROQ_API_KEY=your_key_here"
        )

    return Groq(api_key=api_key)

def load_vector_db():
    """Load the embedding model and ChromaDB collection."""
    embedding_model = SentenceTransformer(EMBEDDING_MODEL_NAME)
    client = chromadb.PersistentClient(path=VECTOR_DB_DIR)
    collection = client.get_collection(name=COLLECTION_NAME)
    return embedding_model, collection

def retrieve_context(question: str, embedding_model, collection, top_k: int = TOP_K) -> list[str]:
    """Embed the question and retrieve the top-k most relevant chunks."""
    query_embedding = embedding_model.encode([question], convert_to_numpy=True)

    results = collection.query(
        query_embeddings=query_embedding.tolist(),
        n_results=top_k,
    )

    return results["documents"][0]

def strip_reasoning_tags(text: str) -> str:
    """
    Some models (e.g. Qwen reasoning models) emit an internal chain-of-
    thought wrapped in <think>...</think> before the actual answer. Users
    should only see the final answer, not the model's internal reasoning.
    This removes any such block if present.
    """
    if "<think>" in text and "</think>" in text:
        start = text.find("<think>")
        end = text.find("</think>") + len("</think>")
        text = text[:start] + text[end:]
    return text.strip()

def generate_answer(groq_client: Groq, question: str, context_chunks: list[str]) -> str:
    """Send the question + retrieved context to the Groq LLM and return its answer."""
    context_text = "\n\n---\n\n".join(context_chunks)

    user_message = f"""Context:
{context_text}

Question: {question}"""

    response = groq_client.chat.completions.create(
        model=GROQ_MODEL,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_message},
        ],
        temperature=0.3,
        max_tokens=1000,
    )

    raw_answer = response.choices[0].message.content
    return strip_reasoning_tags(raw_answer)

def main():
    print("Loading vector database and embedding model...")
    embedding_model, collection = load_vector_db()
    print(f"Loaded collection with {collection.count()} chunks.")

    print("Connecting to Groq API...")
    groq_client = load_env_and_client()

    print("\nSIESVAI is ready. Ask a question (or type 'quit' to exit):\n")

    while True:
        question = input("You: ").strip()
        if question.lower() in ("quit", "exit"):
            print("Goodbye!")
            break
        if not question:
            continue

        context_chunks = retrieve_context(question, embedding_model, collection)

        try:
            answer = generate_answer(groq_client, question, context_chunks)
        except Exception as e:
            print(f"\n[ERROR calling Groq API]: {e}\n")
            continue

        print(f"\nSIESVAI: {answer}\n")

if __name__ == "__main__":
    main()
