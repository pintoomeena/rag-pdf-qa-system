import os
from pathlib import Path
from dotenv import load_dotenv
from google import genai

load_dotenv(Path(__file__).resolve().parents[1] / ".env")

def generate_answer(context: str, question: str) -> str:
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        raise RuntimeError("GEMINI_API_KEY is not configured.")

    client = genai.Client(api_key=api_key)

    response = client.models.generate_content(
        model="gemini-3.8-flash",
        contents=(
            "Answer using only the supplied document context. "
            "If the answer is missing, say you could not find it in the document. "
            "Do not invent facts.\n\n"
            f"Context:\n{context[:12000]}\n\n"
            f"Question: {question}"
        ),
        config={
            "temperature": 0.1,
            "max_output_tokens": 300,
        },
    )

    return response.text or "No answer was generated."
