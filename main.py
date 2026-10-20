
import os
from pathlib import Path
from typing import List

from dotenv import load_dotenv
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import HTMLResponse

from services.parser import extract_text_from_pdf
from services.chunking import chunk_text
from services.embeddings import (
    generate_embeddings,
    generate_query_embedding,
)
from services.vector_store import (
    store_embeddings,
    retrieve_relevant_chunks,
)
from services.llm_service import generate_answer


# Load the .env file from the project root.
PROJECT_DIR = Path(__file__).resolve().parent
load_dotenv(PROJECT_DIR / ".env")

app = FastAPI(title="RAG PDF QA API")

UPLOAD_DIR = PROJECT_DIR / "uploads"
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)

MAX_FILES = 20
MAX_FILE_SIZE = 20 * 1024 * 1024  # 20 MB per PDF


@app.get("/")
def home():
    return {"message": "RAG API Running"}


@app.post("/upload/")
async def upload_documents(files: List[UploadFile] = File(...)):
    if not files:
        raise HTTPException(
            status_code=400,
            detail="Upload at least one PDF.",
        )

    if len(files) > MAX_FILES:
        raise HTTPException(
            status_code=400,
            detail=f"Maximum {MAX_FILES} files allowed per request.",
        )

    uploaded_files = []

    for file in files:
        safe_name = Path(file.filename or "").name

        if not safe_name or Path(safe_name).suffix.lower() != ".pdf":
            raise HTTPException(
                status_code=400,
                detail="Only PDF files are allowed.",
            )

        file_path = None

        try:
            content = await file.read(MAX_FILE_SIZE + 1)

            if not content:
                raise HTTPException(
                    status_code=400,
                    detail=f"{safe_name} is empty.",
                )

            if len(content) > MAX_FILE_SIZE:
                raise HTTPException(
                    status_code=413,
                    detail=f"{safe_name} exceeds the 20 MB limit.",
                )

            # Use a unique name to avoid overwriting files with the same name.
            import uuid

            stored_name = f"{uuid.uuid4().hex}_{safe_name}"
            file_path = UPLOAD_DIR / stored_name
            file_path.write_bytes(content)

            text = extract_text_from_pdf(str(file_path))

            if not text or not text.strip():
                raise HTTPException(
                    status_code=422,
                    detail=(
                        f"No extractable text found in {safe_name}. "
                        "Scanned PDFs may require OCR."
                    ),
                )

            chunks = chunk_text(text)

            if not chunks:
                raise HTTPException(
                    status_code=422,
                    detail=f"No text chunks could be created from {safe_name}.",
                )

            embeddings = generate_embeddings(chunks)

            if len(embeddings) != len(chunks):
                raise RuntimeError(
                    "The number of generated embeddings does not match "
                    "the number of text chunks."
                )

            store_embeddings(chunks, embeddings)
            uploaded_files.append(safe_name)

        except HTTPException:
            raise
        except Exception:
            # Log the full traceback on the server, not in the API response.
            import logging
            logging.exception("Failed to process uploaded PDF: %s", safe_name)

            raise HTTPException(
                status_code=500,
                detail=f"Failed to process {safe_name}. Check backend logs.",
            )
        finally:
            await file.close()

    return {
        "message": "Files uploaded and processed successfully.",
        "files": uploaded_files,
    }


@app.get("/test-upload", response_class=HTMLResponse)
def test_upload_page():
    return """
    <!DOCTYPE html>
    <html>
      <body>
        <h2>Upload PDF</h2>
        <form action="/upload/" enctype="multipart/form-data" method="post">
          <input name="files" type="file" accept=".pdf" multiple required>
          <input type="submit" value="Upload">
        </form>
      </body>
    </html>
    """


@app.get("/query/")
def query_documents(question: str):
    question = question.strip()

    if not question:
        raise HTTPException(
            status_code=400,
            detail="Question cannot be empty.",
        )

    try:
        query_embedding = generate_query_embedding(question)
        results = retrieve_relevant_chunks(query_embedding)

        if not results:
            return {
                "question": question,
                "answer": "No relevant document content found. Upload a PDF first.",
                "relevant_chunks": [],
            }

        context = "\n\n".join(
            item if isinstance(item, str) else str(item)
            for item in results
        )

        answer = generate_answer(context, question)

        return {
            "question": question,
            "answer": answer,
            "relevant_chunks": results,
        }

    except HTTPException:
        raise
    except Exception:
        import logging
        logging.exception("Failed to answer question")

        raise HTTPException(
            status_code=500,
            detail="Failed to answer question. Check backend logs.",
        )
