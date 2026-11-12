# Clinical RAG Engine

A local clinical document RAG system that combines a FastAPI backend with a Streamlit dashboard for uploading PDFs, querying indexed protocols, and reviewing traceable citations.

## Overview

This project ingests clinical PDFs, splits them into chunks, stores them in a persistent ChromaDB vector database, and answers questions using a Groq-backed LLM grounded in retrieved evidence.

It is designed for workflows such as:

- Reviewing uploaded clinical study documents
- Searching protocol text by natural-language questions
- Restricting queries to one document via filename
- Checking the exact source document and page behind each answer

## Application architecture

The project has two main interfaces:

- Backend API: FastAPI app in `app/main.py`
- Frontend UI: Streamlit app in `app/user_interface/ui.py`

Together they provide:

- PDF upload and document indexing
- Vector retrieval with source filtering
- Query answering with inline citations
- Audit trail display in the UI
- Document listing and deletion tools

## Features

- Upload PDFs for indexing
- Chunk pages into searchable document segments
- Persist chunks to ChromaDB
- Use local embeddings (`sentence-transformers/all-MiniLM-L6-v2`)
- Query with natural language
- Filter retrieval by `source_file`
- Return grounded answers with citations like `[Doc: filename.pdf, Page: 8]`
- Provide a Streamlit dashboard for document management and clinical Q&A

## Project structure

```text
clinical-rag-engine/
├── app/
│   ├── __init__.py
│   ├── config.py
│   ├── main.py
│   ├── schemas.py
│   ├── services/
│   │   ├── __init__.py
│   │   ├── ingestion.py
│   │   ├── rag_engine.py
│   │   └── vector_store.py
│   └── user_interface/
│       └── ui.py
├── data/
│   └── chroma_db/
├── tests/
│   ├── __init__.py
│   ├── test_ingestion.py
│   └── test_rag.py
├── .gitignore
├── Readme.md
├── requirments.txt
└── .env
```

## Tech stack

- Python 3.11+
- FastAPI
- Streamlit
- Pydantic + Pydantic Settings
- LangChain
- ChromaDB
- `sentence-transformers`
- PyPDF
- Groq API via OpenAI-compatible endpoint

## Prerequisites

Before running the app, make sure you have:

- Python 3.11+
- A virtual environment for the project
- A Groq API key
- Internet access for embeddings and model inference

## Setup

1. Clone the repository

```bash
git clone <repo-url>
cd clinical-rag-engine
```

2. Create and activate a virtual environment

```bash
# Windows
python -m venv venv
.\venv\Scripts\activate

# macOS / Linux
python -m venv venv
source venv/bin/activate
```

3. Install dependencies

```bash
pip install -r requirments.txt
```

4. Create a `.env` file in the root directory:

```env
GROQ_API_KEY=your_groq_api_key_here
GROQ_MODEL_NAME=openai/gpt-oss-20b
GROQ_BASE_URL=https://api.groq.com/openai/v1
```

## Run the backend API

Start the FastAPI server:

```bash
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

Open the backend docs here:

- Swagger UI: `http://localhost:8000/docs`
- ReDoc: `http://localhost:8000/redoc`

## Run the Streamlit UI

Start the UI in a second terminal:

```bash
streamlit run app/user_interface/ui.py --server.port 8501
```

Then open:

- Streamlit app: `http://localhost:8501`

### UI features

The UI includes:

- System health panel for the backend and embedding model
- PDF upload form for ingesting clinical documents
- Document repository section with delete actions
- Query input box for natural-language clinical questions
- Optional document scope selector for searching one file or all files
- Audit trail showing retrieved chunk count, active document scope, and verified citations

## API overview

### Health check

```http
GET /api/v1/health
```

Returns backend and vector store status.

### Upload a PDF

```http
POST /api/v1/documents/upload
```

Form fields:

- `file`: PDF file

Example:

```bash
curl -X POST "http://localhost:8000/api/v1/documents/upload" \
  -F "file=@sample_protocol.pdf"
```

### Query the indexed documents

```http
POST /api/v1/query
```

Request body:

```json
{
  "question": "What adverse events were reported in the document?",
  "filter_metadata": {
    "source_file": "sample_protocol.pdf"
  }
}
```

Example response fields:

- `answer`
- `citations`
- `chunks_retrieved`
- `extracted_filter`
- `active_filter`
- `cleaned_query`

### List indexed documents

```http
GET /api/v1/documents
```

### Delete a document

```http
DELETE /api/v1/documents/{filename}
```

## Retrieval flow

1. A PDF is uploaded to the backend.
2. The PDF is parsed page-by-page using PyPDF.
3. Pages are chunked and enriched with metadata such as filename and page number.
4. Chunks are stored in ChromaDB with embeddings.
5. A question is sent to the backend from the UI or API.
6. Similar chunks are retrieved using semantic search.
7. The retrieved context is sent to the Groq model.
8. The model returns a source-grounded answer with inline page citations.

## Guardrails

The engine is intentionally conservative:

- It uses only the retrieved evidence
- It avoids guessing beyond the context
- It returns `Insufficient clinical documentation provided.` when evidence is weak or missing
- It only includes verified citations that are actually present in its answer

## Example Python usage

```python
from app.services.rag_engine import RagEngine

rag = RagEngine()

answer, citations, chunk_count, active_filters, cleaned_query = rag.answer_query(
    question="What are the key inclusion criteria described in the document?",
    explicit_filter=None
)

print(answer)
for citation in citations:
    print(citation.source_file, citation.page)
```

## Testing

Run the project tests with:

```bash
pytest
```

## Notes

- The vector store uses a local ChromaDB instance under `data/chroma_db`.
- This project is best treated as a local clinical research prototype or internal tool, not a production clinical decision system.
- Before exposing it beyond a trusted environment, review data handling, access control, and deployment security requirements.
