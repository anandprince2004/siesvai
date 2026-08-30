import os
import glob
import chromadb
from sentence_transformers import SentenceTransformer

BASE_DIR = os.path.join(os.path.dirname(__file__), "..")
SOURCE_DIRS = [
    os.path.join(BASE_DIR, "data", "processed"),
    os.path.join(BASE_DIR, "data", "manual"),
]
VECTOR_DB_DIR = os.path.join(BASE_DIR, "vectorstore")
COLLECTION_NAME = "siesvai_knowledge_base"

CHUNK_SIZE = 800
CHUNK_OVERLAP = 100
EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2"

def is_faq_style(text: str) -> bool:
    """
    Detect whether a file is written in 'Q: ... A: ...' FAQ format.
    If it contains multiple 'Q:' markers, treat it as FAQ-style.
    """
    return text.count("\nQ:") + (1 if text.startswith("Q:") else 0) >= 2

def chunk_faq_text(text: str) -> list[str]:
    """
    Split FAQ-style text into one chunk per Q&A pair, so each answer stays
    fully intact and isn't cut off mid-sentence or merged with the next
    question. This gives much cleaner, more precise retrieval than blind
    character-count chunking for structured Q&A content.
    """
    lines = text.split("\n")
    chunks = []
    current_chunk_lines = []

    for line in lines:
        if line.strip().startswith("Q:") and current_chunk_lines:
            chunks.append("\n".join(current_chunk_lines).strip())
            current_chunk_lines = [line]
        else:
            current_chunk_lines.append(line)

    if current_chunk_lines:
        chunks.append("\n".join(current_chunk_lines).strip())

    chunks = [c for c in chunks if "Q:" in c]

    return chunks

def chunk_text(text: str, chunk_size: int = CHUNK_SIZE, overlap: int = CHUNK_OVERLAP) -> list[str]:
    """
    Split text into overlapping chunks by character count.
    Tries to break on paragraph/line boundaries where possible, so chunks
    don't cut a sentence awkwardly in half.

    Used for non-FAQ files (e.g. contact.txt, courses_syllabus.txt).
    FAQ-style files are chunked separately via chunk_faq_text().
    """
    paragraphs = [p.strip() for p in text.split("\n") if p.strip()]

    chunks = []
    current_chunk = ""

    for paragraph in paragraphs:
        if len(current_chunk) + len(paragraph) + 1 <= chunk_size:
            current_chunk += (paragraph + "\n")
        else:
            if current_chunk:
                chunks.append(current_chunk.strip())
            overlap_text = current_chunk[-overlap:] if current_chunk else ""
            current_chunk = overlap_text + paragraph + "\n"

    if current_chunk.strip():
        chunks.append(current_chunk.strip())

    return chunks

def chunk_document(text: str) -> list[str]:
    """
    Route to the right chunking strategy based on the document's structure.
    FAQ-style docs (Q:/A: format) get split per Q&A pair; everything else
    gets standard character-count chunking.
    """
    if is_faq_style(text):
        return chunk_faq_text(text)
    return chunk_text(text)

def load_source_files() -> list[dict]:
    """
    Read every .txt file from SOURCE_DIRS.
    Returns a list of dicts: {source_file, text}
    """
    documents = []

    for directory in SOURCE_DIRS:
        if not os.path.isdir(directory):
            print(f"  Skipping missing directory: {directory}")
            continue

        txt_files = glob.glob(os.path.join(directory, "*.txt"))
        for filepath in txt_files:
            with open(filepath, "r", encoding="utf-8") as f:
                text = f.read()
            documents.append({
                "source_file": os.path.basename(filepath),
                "text": text,
            })
            print(f"  Loaded: {filepath} ({len(text)} chars)")

    return documents

def build_vector_db() -> None:
    print("Step 1: Loading source files...")
    documents = load_source_files()

    if not documents:
        print("No source files found. Make sure data/processed/ and "
              "data/manual/ contain .txt files.")
        return

    print(f"\nStep 2: Chunking {len(documents)} document(s) "
          f"(FAQ-style files split per Q&A pair, others by character count)...")
    all_chunks = []
    all_metadatas = []
    all_ids = []

    for doc in documents:
        chunks = chunk_document(doc["text"])
        for i, chunk in enumerate(chunks):
            all_chunks.append(chunk)
            all_metadatas.append({
                "source_file": doc["source_file"],
                "chunk_index": i,
            })
            all_ids.append(f"{doc['source_file']}_chunk_{i}")
        print(f"  {doc['source_file']} -> {len(chunks)} chunk(s)")

    print(f"\nTotal chunks to embed: {len(all_chunks)}")

    print(f"\nStep 3: Loading embedding model '{EMBEDDING_MODEL_NAME}' "
          f"(first run downloads ~80MB, runs on CPU)...")
    embedding_model = SentenceTransformer(EMBEDDING_MODEL_NAME)

    print("Step 4: Generating embeddings...")
    embeddings = embedding_model.encode(
        all_chunks,
        show_progress_bar=True,
        convert_to_numpy=True,
    )

    print(f"\nStep 5: Saving to ChromaDB at '{VECTOR_DB_DIR}'...")
    client = chromadb.PersistentClient(path=VECTOR_DB_DIR)

    existing_collections = [c.name for c in client.list_collections()]
    if COLLECTION_NAME in existing_collections:
        client.delete_collection(COLLECTION_NAME)
        print(f"Removed old '{COLLECTION_NAME}' collection before rebuilding.")

    collection = client.create_collection(name=COLLECTION_NAME)

    collection.add(
        ids=all_ids,
        embeddings=embeddings.tolist(),
        documents=all_chunks,
        metadatas=all_metadatas,
    )

    print(f"\nDone. {len(all_chunks)} chunks stored in ChromaDB collection "
          f"'{COLLECTION_NAME}' at {VECTOR_DB_DIR}")

if __name__ == "__main__":
    build_vector_db()
