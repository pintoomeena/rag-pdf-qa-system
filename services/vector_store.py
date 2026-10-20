
import hashlib
from pathlib import Path

import chromadb


# Store ChromaDB relative to the project root, not the terminal's
# current working directory.
PROJECT_DIR = Path(__file__).resolve().parents[1]
CHROMA_PATH = PROJECT_DIR / "chroma_db"

client = chromadb.PersistentClient(path=str(CHROMA_PATH))

COLLECTION_NAME = "documents"
EMBEDDING_DIMENSION = 3072


def _get_collection():
    """Get the collection without automatically deleting existing data."""
    try:
        collection = client.get_collection(name=COLLECTION_NAME)

        # Check stored dimensions when existing embeddings are available.
        existing = collection.peek(limit=1)
        stored_embeddings = existing.get("embeddings")

        if stored_embeddings is not None and len(stored_embeddings) > 0:
            dimension = len(stored_embeddings[0])

            if dimension != EMBEDDING_DIMENSION:
                raise ValueError(
                    f"Existing ChromaDB collection uses {dimension}-dimensional "
                    f"embeddings, but the configured model uses "
                    f"{EMBEDDING_DIMENSION}. Back up or recreate the collection "
                    "before uploading documents with the new model."
                )

        return collection

    except ValueError:
        raise
    except Exception as exc:
        # Create the collection only when it genuinely does not exist.
        try:
            return client.create_collection(name=COLLECTION_NAME)
        except Exception:
            raise RuntimeError(
                f"Could not access ChromaDB collection '{COLLECTION_NAME}'."
            ) from exc


collection = _get_collection()


def store_embeddings(chunks, embeddings, source_name="document"):
    """Store text chunks and their corresponding embeddings."""
    if len(chunks) != len(embeddings):
        raise ValueError("Chunks and embeddings must have equal lengths.")

    if not chunks:
        return

    ids = []
    documents = []
    vectors = []
    metadatas = []

    for index, (chunk, embedding) in enumerate(zip(chunks, embeddings)):
        if not isinstance(chunk, str) or not chunk.strip():
            continue

        embedding_values = (
            embedding.tolist()
            if hasattr(embedding, "tolist")
            else list(embedding)
        )

        if len(embedding_values) != EMBEDDING_DIMENSION:
            raise ValueError(
                f"Expected {EMBEDDING_DIMENSION}-dimensional embeddings, "
                f"received {len(embedding_values)}."
            )

        # Include source and chunk position to prevent identical text
        # from different documents from overwriting each other.
        identity = f"{source_name}:{index}:{chunk}"
        chunk_id = hashlib.sha256(identity.encode("utf-8")).hexdigest()

        ids.append(chunk_id)
        documents.append(chunk)
        vectors.append(embedding_values)
        metadatas.append({
            "source": source_name,
            "chunk_index": index,
        })

    if ids:
        collection.upsert(
            ids=ids,
            embeddings=vectors,
            documents=documents,
            metadatas=metadatas,
        )


def retrieve_relevant_chunks(query_embedding, top_k=3):
    """Retrieve the most relevant document chunks for a query."""
    if top_k < 1:
        raise ValueError("top_k must be at least 1.")

    embedding_values = (
        query_embedding.tolist()
        if hasattr(query_embedding, "tolist")
        else list(query_embedding)
    )

    if len(embedding_values) != EMBEDDING_DIMENSION:
        raise ValueError(
            f"Expected {EMBEDDING_DIMENSION}-dimensional query embedding, "
            f"received {len(embedding_values)}."
        )

    total = collection.count()

    if total == 0:
        return []

    results = collection.query(
        query_embeddings=[embedding_values],
        n_results=min(top_k, total),
        include=["documents", "distances"],
    )

    documents = results.get("documents") or []

    if not documents or not documents[0]:
        return []

    return documents[0]
