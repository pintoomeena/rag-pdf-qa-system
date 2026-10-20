
import os
from pathlib import Path

from dotenv import load_dotenv
from google import genai
from google.genai import types

# Load .env from the project root for local development.
# On Render, GEMINI_API_KEY should be set in Environment Variables.
load_dotenv(Path(__file__).resolve().parents[1] / ".env")

EMBEDDING_MODEL = "gemini-embedding-001"


def get_client():
    """Create a Gemini API client."""
    api_key = os.getenv("GEMINI_API_KEY", "").strip()

    if not api_key:
        raise RuntimeError(
            "GEMINI_API_KEY is missing. Configure it in your .env "
            "file locally or in Render Environment Variables."
        )

    return genai.Client(api_key=api_key)


def generate_embeddings(chunks):
    """Generate document embeddings for a list of text chunks."""
    if not chunks:
        return []

    client = get_client()
    embeddings = []

    # Process batches to avoid sending an excessively large request.
    batch_size = 50

    for start in range(0, len(chunks), batch_size):
        batch = chunks[start : start + batch_size]

        response = client.models.embed_content(
            model=EMBEDDING_MODEL,
            contents=batch,
            config=types.EmbedContentConfig(
                task_type="RETRIEVAL_DOCUMENT",
            ),
        )

        if not response.embeddings:
            raise RuntimeError(
                "Gemini returned no embeddings for a document batch."
            )

        embeddings.extend(
            [list(embedding.values) for embedding in response.embeddings]
        )

    return embeddings


def generate_query_embedding(query):
    """Generate an embedding for a user's search query."""
    if not query or not query.strip():
        raise ValueError("Query cannot be empty.")

    client = get_client()

    response = client.models.embed_content(
        model=EMBEDDING_MODEL,
        contents=query.strip(),
        config=types.EmbedContentConfig(
            task_type="RETRIEVAL_QUERY",
        ),
    )

    if not response.embeddings or not response.embeddings[0].values:
        raise RuntimeError("Gemini returned no embedding for the query.")

    return list(response.embeddings[0].values)
