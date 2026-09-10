import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from backend.rag_pipeline import load_groq_client, load_vector_db, answer_question


def main():
    print("Loading vector database and embedding model...")
    embedding_model, collection = load_vector_db()
    print(f"Loaded collection with {collection.count()} chunks.")

    print("Connecting to Groq API...")
    groq_client = load_groq_client()

    print("\nSIESVAI is ready. Ask a question (or type 'quit' to exit):\n")

    while True:
        question = input("You: ").strip()
        if question.lower() in ("quit", "exit"):
            print("Goodbye!")
            break
        if not question:
            continue

        result = answer_question(groq_client, embedding_model, collection, question)

        try:
            print(f"\nSIESVAI: {result['answer']}\n")
        except Exception as e:
            print(f"\n[ERROR]: {e}\n")


if __name__ == "__main__":
    main()
