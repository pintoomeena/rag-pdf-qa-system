from sentence_transformers import SentenceTransformer

model = SentenceTransformer(
    'all-MiniLM-L6-v2'
)


def generate_embeddings(chunks):

    embeddings = model.encode(chunks)

    return embeddings


def generate_query_embedding(query):

    embedding = model.encode(query)

    return embedding