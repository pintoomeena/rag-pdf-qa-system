# RAG-based PDF Question Answering System

## Overview

This project is a Retrieval-Augmented Generation (RAG) based PDF Question Answering system built using FastAPI.

Users can:
- Upload PDF documents
- Generate embeddings
- Store embeddings in ChromaDB
- Retrieve relevant chunks
- Ask questions from uploaded PDFs

---

## Tech Stack

- FastAPI
- Sentence Transformers
- ChromaDB
- HuggingFace Transformers
- PyMuPDF (fitz)
- Python

---

## Features

- Multi-PDF Upload
- Semantic Search
- Vector Database Storage
- Local LLM-based Answer Generation
- REST API Architecture

---

## Run Application

```bash
uvicorn main:app --reload
