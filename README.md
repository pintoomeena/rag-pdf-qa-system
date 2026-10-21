# RAG-Based PDF Question Answering System

An end-to-end **Retrieval-Augmented Generation (RAG)** application that allows users to upload PDF documents and ask natural-language questions about their content. The system uses **FastAPI** for its REST API, **ChromaDB** for vector storage and semantic retrieval, and **Google Gemini** for embeddings and answer generation.

## Overview

The application extracts text from PDFs, splits it into manageable chunks, generates embeddings, and stores them in ChromaDB. When a user asks a question, the system retrieves relevant chunks and passes them to Gemini to generate a context-aware answer.

## Features

- **Multi-PDF Upload:** Upload multiple PDF documents through the API.
- **PDF Text Extraction:** Extract text from PDFs using PyMuPDF.
- **Text Chunking:** Split extracted text into smaller chunks for efficient retrieval.
- **Semantic Search:** Retrieve relevant document content using vector embeddings.
- **Vector Database:** Store and retrieve embeddings using ChromaDB.
- **AI-Powered Answers:** Generate answers using the Google Gemini API.
- **REST API:** Expose document upload and question-answering functionality through FastAPI.
- **Interactive API Documentation:** Test endpoints using Swagger UI.

## Tech Stack

- **Backend:** FastAPI, Uvicorn
- **LLM & Embeddings:** Google Gemini API (`google-genai`)
- **Vector Database:** ChromaDB
- **PDF Processing:** PyMuPDF
- **PDF Validation:** pypdf
- **Configuration:** python-dotenv
- **Deployment:** Render

## Architecture

```text
PDF Documents
     |
     v
Text Extraction (PyMuPDF)
     |
     v
Text Chunking
     |
     v
Gemini Embeddings
     |
     v
ChromaDB Vector Storage
     |
     v
User Question
     |
     v
Query Embedding & Semantic Retrieval
     |
     v
Relevant Document Chunks
     |
     v
Gemini Answer Generation
     |
     v
Final Answer
```

## How It Works

1. **Upload PDFs:** The user uploads one or more PDF documents through the FastAPI endpoint.
2. **Extract and Chunk Text:** The application extracts text and splits it into smaller chunks.
3. **Generate Embeddings:** Gemini converts document chunks into vector representations.
4. **Store Vectors:** ChromaDB stores the embeddings along with document content and metadata.
5. **Retrieve Context:** The user's question is embedded, and semantically relevant chunks are retrieved.
6. **Generate Answer:** Gemini uses the retrieved context to generate an answer relevant to the uploaded documents.

## Getting Started

### Prerequisites

- Python 3.10 or later
- A Google Gemini API key
- Git

### 1. Clone the Repository

```bash
git clone https://github.com/pintoomeena/rag-pdf-qa-system.git
cd rag-pdf-qa-system
```

### 2. Create a Virtual Environment

```bash
python -m venv venv
```

Activate it on Windows:

```bash
venv\Scripts\activate
```

On Linux or macOS:

```bash
source venv/bin/activate
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

### 4. Configure Environment Variables

Create a `.env` file in the project root:

```env
GEMINI_API_KEY=your_gemini_api_key
```

Replace `your_gemini_api_key` with your actual API key. Never commit your `.env` file or expose your API key publicly.

### 5. Run the Application

```bash
uvicorn main:app --reload
```

The API will be available at:

```text
http://127.0.0.1:8000
```

### 6. Explore the API

Open Swagger UI in your browser:

```text
http://127.0.0.1:8000/docs
```

Use the interactive documentation to upload PDFs and test the available question-answering endpoints.

## Deployment

The FastAPI backend can be deployed on Render.

**Build command:**

```bash
pip install -r requirements.txt
```

**Start command:**

```bash
uvicorn main:app --host 0.0.0.0 --port $PORT
```

Configure `GEMINI_API_KEY` in the Render service's environment variables.

## Environment Variables

| Variable | Description |
|---|---|
| `GEMINI_API_KEY` | API key used to access Google Gemini |

## Future Improvements

- Add conversation history for follow-up questions.
- Improve retrieval with reranking and metadata filters.
- Add support for scanned PDFs using OCR.
- Implement authentication and document-level access controls.
- Add automated tests and CI workflows.
- Improve handling of temporary Gemini API errors using retry logic.

## Author

**Pintoo Meena**

- GitHub: [pintoomeena](https://github.com/pintoomeena)

---

If you find this project useful, consider giving the repository a star.
