
from dotenv import load_dotenv

load_dotenv()
from pathlib import Path
from typing import List

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

app = FastAPI(title="RAG PDF QA API")

UPLOAD_DIR = Path("uploads")
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)


@app.get("/")
def home():
    return {"message": "RAG API Running"}


@app.post("/upload/")
async def upload_documents(files: List[UploadFile] = File(...)):
    if not files:
        raise HTTPException(status_code=400, detail="Upload at least one PDF.")

    if len(files) > 20:
        raise HTTPException(
            status_code=400,
            detail="Maximum 20 files allowed.",
        )

    uploaded_files = []

    for file in files:
        if not file.filename or Path(file.filename).suffix.lower() != ".pdf":
            raise HTTPException(
                status_code=400,
                detail="Only PDF files are allowed.",
            )

        safe_name = Path(file.filename).name
        file_path = UPLOAD_DIR / safe_name

        try:
            content = await file.read()

            if not content:
                raise HTTPException(
                    status_code=400,
                    detail=f"{safe_name} is empty.",
                )

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
            store_embeddings(chunks, embeddings)

            uploaded_files.append(safe_name)

        except HTTPException:
            raise
        except Exception as exc:
            raise HTTPException(
                status_code=500,
                detail=f"Failed to process {safe_name}: {exc}",
            ) from exc
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

        context = "\n\n".join(results)

        # Gemini function signature: generate_answer(context, question)
        answer = generate_answer(context, question)

        return {
            "question": question,
            "answer": answer,
            "relevant_chunks": results,
        }

    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail=f"Failed to answer question: {exc}",
        ) from exc
