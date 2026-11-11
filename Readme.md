# Clinical RAG Engine

A FastAPI-based clinical document retrieval and question-answering service built for protocol PDFs and regulatory study material. The system ingests PDF pages, chunks them into searchable embeddings, and answers user queries with metadata-aware retrieval, citation support, and clinical-answer guardrails.

## What this project does

This application is designed to help teams work with clinical trial documents by combining:

- PDF ingestion and text chunking
- Vector search over clinical content using local embeddings
- Automatic metadata extraction for drug names and trial phases
- Retrieval with optional explicit filters
- Retrieval-augmented answer generation from a Groq-backed LLM
- Source-grounded citations with page references

It is useful for scenarios such as:

- Querying trial protocols and study notes
- Looking up adverse event reporting details
- Filtering by drug and phase when multiple documents are indexed
- Answering questions from indexed PDFs with traceable evidence

## Key features

- PDF upload endpoint for indexing clinical documents
- ChromaDB-backed persistent vector storage
- Sentence-transformers embeddings for local similarity search
- Automatic extraction of clinical intent from natural language questions
- Metadata filtering based on `drug_name` and `trial_phase`
- Grounded responses with document and page citations
- Fallback answer when there is insufficient evidence

## Project structure

```text
clinical-rag-engine/
├── app/
│   ├── __init__.py
│   ├── config.py
│   ├── main.py
│   ├── schemas.py
│   └── services/
│       ├── __init__.py
│       ├── ingestion.py
│       ├── rag_engine.py
│       └── vector_store.py
├── data/
│   └── chroma_db/
├── .env
├── .gitignore
├── Readme.md
└── requirments.txt
```

## Technology stack

- Python 3.11+
- FastAPI
- Pydantic + Pydantic Settings
- ChromaDB
- LangChain
- Hugging Face sentence-transformers embeddings
- Groq LLM API

## Prerequisites

Before running the project, make sure you have:

- Python installed
- A virtual environment created for the project
- A Groq API key
- Access to the internet for model requests

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

4. Create a `.env` file in the project root

```env
GROQ_API_KEY=your_groq_api_key_here
GROQ_MODEL_NAME=openai/gpt-oss-20b
GROQ_BASE_URL=https://api.groq.com/openai/v1
```

## Running the app

Start the API with Uvicorn:

```bash
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

The API docs will be available at:

- Swagger UI: `http://localhost:8000/docs`
- ReDoc: `http://localhost:8000/redoc`

## API overview

### Health check

```http
GET /api/v1/health
```

Returns status information for the vector store and configured LLM model.

### Upload a clinical PDF

```http
POST /api/v1/documents/upload
```

Form fields:

- `file`: PDF file
- `drug_name` (optional): drug or compound name
- `trial_phase` (optional): such as `Phase II` or `Phase III`

Example using curl:

```bash
curl -X POST "http://localhost:8000/api/v1/documents/upload" \
  -F "file=@sample_protocol.pdf" \
  -F "drug_name=Pembrolizumab" \
  -F "trial_phase=Phase III"
```

### Query the indexed documents

```http
POST /api/v1/query
```

Request body example:

```json
{
  "question": "What adverse events were reported in Phase III trials of Pembrolizumab?",
  "filter_metadata": {
    "drug_name": "Pembrolizumab",
    "trial_phase": "Phase III"
  }
}
```

Response includes:

- `answer`
- `citations`
- `chunks_retrieved`
- `extracted_filter`
- `cleaned_query`

## How retrieval works

1. A PDF is uploaded and split into chunks.
2. Each chunk is enriched with metadata such as the original file name, page number, drug, and trial phase.
3. The chunks are stored in ChromaDB with local embeddings.
4. A user question is translated into a search query and optional metadata filter.
5. Relevant chunks are retrieved by similarity search.
6. The LLM produces an answer using only the retrieved context, with citations back to the source pages.

## Notes and guardrails

The engine includes filtering and safety behavior intended for clinical use:

- It attempts to extract drug and phase hints from the user question.
- It can also accept explicit filters when users already know the target metadata.
- It rejects unsupported or low-confidence results using a threshold guard.
- When the evidence is weak or missing, it returns a conservative fallback message rather than guessing.

## Example usage in Python

```python
from app.services.rag_engine import RagEngine
from app.schemas import DocumentMetadatafilter

rag = RagEngine()

answer, citations, count, active_filter, cleaned_query = rag.answer_query(
    question="What are the key inclusion criteria for Phase II studies involving nivolumab?"
)

print(answer)
for citation in citations:
    print(citation.source_file, citation.page)
```

## Important project note

The project currently relies on a local ChromaDB instance under `data/chroma_db` and a Groq-backed LLM for synthesis. For production use, you should review model access, security controls, and data handling requirements before exposing the API beyond a trusted environment.
