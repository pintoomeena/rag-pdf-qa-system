import chromadb

client = chromadb.PersistentClient(
    path="chroma_db"
)

collection = client.get_or_create_collection(
    name="documents"
)


def store_embeddings(
    chunks,
    embeddings
):

    for i, chunk in enumerate(chunks):

        collection.add(
            ids=[f"chunk_{i}_{hash(chunk)}"],
            embeddings=[embeddings[i].tolist()],
            documents=[chunk]
        )


def retrieve_relevant_chunks(
    query_embedding,
    top_k=1
):

    results = collection.query(
        query_embeddings=[
            query_embedding.tolist()
        ],
        n_results=top_k
    )

    return results['documents'][0]