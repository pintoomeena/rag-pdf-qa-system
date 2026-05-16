from fastapi import FastAPI, UploadFile, File
from fastapi.responses import HTMLResponse
from typing import List
import os
from services.parser import extract_text_from_pdf
from services.chunking import chunk_text

from services.embeddings import (
    generate_embeddings,
    generate_query_embedding
)

from services.vector_store import (
    store_embeddings,
    retrieve_relevant_chunks
)
from services.llm_service import generate_answer

app = FastAPI()

UPLOAD_DIR = "uploads"

if not os.path.exists(UPLOAD_DIR):
    os.mkdir(UPLOAD_DIR)


@app.get("/")
def home():

    return {
        "message": "RAG API Running"
    }


@app.post("/upload/")
async def upload_documents(
    files: List[UploadFile] = File(...)
):

    if len(files) > 20:

        return {
            "error": "Maximum 20 files allowed"
        }

    uploaded_files = []

    for file in files:

        file_path = os.path.join(
            UPLOAD_DIR,
            file.filename
        )

        with open(file_path, "wb") as f:

            content = await file.read()

            f.write(content)

        uploaded_files.append(
            file.filename
        )

        text = extract_text_from_pdf(
            file_path
        )

        chunks = chunk_text(text)

        embeddings = generate_embeddings(
            chunks
        )

        store_embeddings(
            chunks,
            embeddings
        )

        print("Stored in ChromaDB")

        print(
            "Total Chunks:",
            len(chunks)
        )

    return {
        "message": "Files uploaded successfully",
        "files": uploaded_files
    }


@app.get("/query/")
def query_documents(
    question: str
):

    query_embedding = generate_query_embedding(
        question
    )

    results = retrieve_relevant_chunks(
        query_embedding
    )

    context = "\n".join(results)

    answer = generate_answer(
        question,
        context
    )

    return {
        "question": question,
        "answer": answer,
        "relevant_chunks": results
    }