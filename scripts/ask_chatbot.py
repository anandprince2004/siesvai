import os
import chromadb
from sentence_transformers import SentenceTransformer
from dotenv import load_dotenv
from groq import Groq

BASE_DIR = os.path.join(os.path.dirname(__file__), "..")
VECTOR_DB_DIR = os.path.join(BASE_DIR, "vectorstore")
COLLECTION_NAME = "siesvai_knowledge_base"
EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2"
TOP_K = 5
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
- Respond in the same language AND the same script as the user's question. \
If the user writes in Romanized/transliterated form (e.g. Hindi written \
in English letters like "kaise ho", Marathi like "kasa ahes", or Tamil \
like "epdi irukinga"), you MUST reply in that same Romanized style using \
English letters. Do NOT switch to native script (Devanagari, Tamil script, \
etc.) even if you are capable of it — the user is typing in Roman letters \
and expects the same back.
- The context below is split into numbered excerpts. Carefully check EVERY \
excerpt before concluding the answer isn't available — the correct answer \
is often not in the first excerpt.
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

def translate_query_for_retrieval(groq_client: Groq, question: str) -> str:
    """
    Translate the user's question into plain English before embedding it
    for retrieval. The knowledge base (FAQ, contact info, course list) is
    entirely in English, and the embedding model (all-MiniLM-L6-v2) is
    English-centric, so Romanized Hindi/Marathi/Tamil queries often fail
    to retrieve the right chunks even when the LLM understands them fine.

    This does NOT change what language the final answer is given in —
    that's still handled separately by the main LLM call using the
    ORIGINAL question. This translation is only used internally to pick
    better search results.

    Falls back to the original question if translation fails for any
    reason, so a translation hiccup never blocks the whole pipeline.
    """
    translation_prompt = """Translate the following question into plain, \
    natural English. The question is being asked to a college information \
    chatbot for SIES (Nerul) College, a South Indian community college in \
    Navi Mumbai. The question may be in English, Hindi, Marathi, or Tamil, \
    possibly written in Roman/English letters (transliterated) rather than \
    native script.

    Since Hindi, Marathi, and Tamil share some similar-sounding Romanized \
    words with very different meanings (e.g. "pati" can mean "husband" in \
    Hindi/Marathi, but is also a common Tamil colloquial short form of \
    "patri"/"gurinji" meaning "about"), use the CONTEXT of a college \
    information chatbot to pick the sensible interpretation. A question about \
    a college is almost never literally about anyone's spouse.

    Reply with ONLY the English translation. No explanation, no quotes, no \
    extra text — just the translated question."""

    try:
        response = groq_client.chat.completions.create(
            model=GROQ_MODEL,
            messages=[
                {"role": "system", "content": translation_prompt},
                {"role": "user", "content": question},
            ],
            temperature=0.0,
            max_tokens=200,
            reasoning_effort="none",
        )
        translated = response.choices[0].message.content.strip()
        return translated if translated else question
    except Exception:
        return question

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

    Handles two cases:
    1. A complete <think>...</think> block -> remove it entirely.
    2. A truncated response where <think> appears but the response got
       cut off (by max_tokens) before </think> -> there is no real answer
       to show, so return a clear fallback message instead of dumping raw
       reasoning text on the user.
    """
    if "<think>" in text:
        if "</think>" in text:
            start = text.find("<think>")
            end = text.find("</think>") + len("</think>")
            text = text[:start] + text[end:]
            return text.strip()
        else:
            return ("Sorry, I need a moment to think that through properly. "
                    "Could you try asking again, or rephrase your question?")
    return text.strip()

def generate_answer(groq_client: Groq, question: str, context_chunks: list[str]) -> str:
    """Send the question + retrieved context to the Groq LLM and return its answer."""
    context_text = "\n\n".join(
        f"[Excerpt {i+1}]\n{chunk}" for i, chunk in enumerate(context_chunks)
    )

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
        max_tokens=1500,
        reasoning_effort="none",
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

        translated_question = translate_query_for_retrieval(groq_client, question)
        context_chunks = retrieve_context(translated_question, embedding_model, collection)

        try:
            answer = generate_answer(groq_client, question, context_chunks)
        except Exception as e:
            print(f"\n[ERROR calling Groq API]: {e}\n")
            continue

        print(f"\nSIESVAI: {answer}\n")

if __name__ == "__main__":
    main()
