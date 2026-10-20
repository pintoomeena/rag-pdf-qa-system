
import hashlib
import chromadb

client = chromadb.PersistentClient(path="chroma_db")

COLLECTION_NAME = "documents"
EMBEDDING_DIMENSION = 3072

# Recreate the collection if its existing embedding dimension is incompatible.
try:
    collection = client.get_collection(COLLECTION_NAME)
    existing = collection.peek(limit=1)

    if existing["embeddings"] is not None and len(existing["embeddings"]) > 0:
        dimension = len(existing["embeddings"][0])
        if dimension != EMBEDDING_DIMENSION:
            client.delete_collection(COLLECTION_NAME)
            collection = client.create_collection(COLLECTION_NAME)
    else:
        collection = client.get_or_create_collection(COLLECTION_NAME)

except Exception:
    collection = client.get_or_create_collection(COLLECTION_NAME)


def store_embeddings(chunks, embeddings):
    if len(chunks) != len(embeddings):
        raise ValueError("Chunks and embeddings must have equal lengths.")

    for chunk, embedding in zip(chunks, embeddings):
        embedding_values = (
            embedding.tolist() if hasattr(embedding, "tolist") else list(embedding)
        )

        if len(embedding_values) != EMBEDDING_DIMENSION:
            raise ValueError(
                f"Expected {EMBEDDING_DIMENSION}-dimensional embeddings, "
                f"received {len(embedding_values)}."
            )

        chunk_id = hashlib.sha256(chunk.encode("utf-8")).hexdigest()

        collection.upsert(
            ids=[chunk_id],
            embeddings=[embedding_values],
            documents=[chunk],
        )


def retrieve_relevant_chunks(query_embedding, top_k=3):
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

    if collection.count() == 0:
        return []

    results = collection.query(
        query_embeddings=[embedding_values],
        n_results=min(top_k, collection.count()),
    )

    return results["documents"][0] if results["documents"] else []
