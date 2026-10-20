import os
from pathlib import Path

from dotenv import load_dotenv
from google import genai

# Load .env from the project root during local development.
# On Render, GEMINI_API_KEY must be configured in Environment Variables.
load_dotenv(Path(__file__).resolve().parents[1] / ".env")

GEMINI_MODEL = "gemini-3.8-flash"


def generate_answer(context: str, question: str) -> str:
    """Generate an answer grounded in the retrieved document context."""

    api_key = os.getenv("GEMINI_API_KEY", "").strip()

    if not api_key:
        raise RuntimeError("GEMINI_API_KEY is not configured.")

    if not question or not question.strip():
        raise ValueError("Question cannot be empty.")

    if not context or not context.strip():
        return "I could not find relevant information in the document."

    client = genai.Client(api_key=api_key)

    prompt = (
        "Answer the question using only the supplied document context. "
        "If the context does not contain the answer, say you could not "
        "find it in the document. Do not invent facts.\n\n"
        f"Document context:\n{context[:12000]}\n\n"
        f"Question: {question.strip()}"
    )

    response = client.models.generate_content(
        model=GEMINI_MODEL,
        contents=prompt,
        config={
            "temperature": 0.1,
            "max_output_tokens": 300,
        },
    )

    answer = response.text

    if not answer or not answer.strip():
        return "No answer was generated. Please try again."

    return answer.strip()
