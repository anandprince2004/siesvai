import os
import chromadb
from sentence_transformers import SentenceTransformer
from dotenv import load_dotenv

load_dotenv()

BASE_DIR = os.path.join(os.path.dirname(__file__), "..")
VECTOR_DB_DIR = os.path.join(BASE_DIR, "vectorstore")
COLLECTION_NAME = "siesvai_knowledge_base"
EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2"
TOP_K = 5

def main():
    print("Loading embedding model and vector DB...")
    embedding_model = SentenceTransformer(EMBEDDING_MODEL_NAME)
    client = chromadb.PersistentClient(path=VECTOR_DB_DIR)
    collection = client.get_collection(name=COLLECTION_NAME)

    print(f"Loaded collection with {collection.count()} chunks.\n")
    print("Type a question (or 'quit' to exit):\n")

    while True:
        query = input("Q: ").strip()
        if query.lower() in ("quit", "exit"):
            break
        if not query:
            continue

        query_embedding = embedding_model.encode([query], convert_to_numpy=True)

        results = collection.query(
            query_embeddings=query_embedding.tolist(),
            n_results=TOP_K,
        )

        print(f"\nTop {TOP_K} matching chunks:\n")
        for i, (doc, meta, dist) in enumerate(zip(
            results["documents"][0],
            results["metadatas"][0],
            results["distances"][0],
        )):
            print(f"--- Match {i+1} (source: {meta['source_file']}, "
                  f"distance: {dist:.4f}) ---")
            print(doc[:300])
            print()

if __name__ == "__main__":
    main()
