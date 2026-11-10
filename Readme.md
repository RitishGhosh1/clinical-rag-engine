# Clinical RAG Engine

A Retrieval-Augmented Generation (RAG) system for querying clinical trial protocols and regulatory documents with auditable inline citations and metadata guardrails.

## Features

- **Clinical PDF Ingestion**: Streams pages from clinical trial PDFs, enriches them with trial metadata (`drug_name`, `trial_phase`, `source_file`, `page`), and stores them in ChromaDB using local Hugging Face embeddings (`sentence-transformers/all-MiniLM-L6-v2`).
- **Query Intent Extraction**: Automatically extracts mentioned drug names and trial phases from natural language questions, formatting a clean semantic search query.
- **Metadata Filtering**: Supports both automatic intent extraction and explicit metadata filtering (by drug and trial phase).
- **Auditability & Citations**: Responses include inline page-level citations (`[Doc: <filename>, Page: <page>]`) referencing the source documents.
- **Guardrails**:
  - Rejection of out-of-domain questions via cosine similarity threshold cutoff.
  - Phase and drug mismatch detection.
  - Strict fallback to `"Insufficient clinical documentation provided."` when evidence is missing or insufficient to prevent hallucination.

## Project Structure

```
clinical-rag-engine/
├── app/
│   ├── config.py             # Application settings & environment configurations
│   ├── schemas.py            # Pydantic schemas for queries, responses, and citations
│   ├── services/
│   │   ├── ingestion.py      # PDF parsing, page chunking, metadata enrichment, & vector storage
│   │   ├── vector_store.py   # ChromaDB initialization and connection management
│   │   └── rag_engine.py     # Intent extraction, filtered retrieval, and LLM answer synthesis
├── data/
│   └── chroma_db/            # Persistent ChromaDB vector database files
├── tests/
│   ├── test_ingestion.py     # Ingestion test script
│   └── test_rag.py           # Verification test suite with clinical query test cases
├── requirments.txt           # Project dependencies
└── .env                      # Environment configuration (API keys)
```

## Setup & Installation

### 1. Clone the repository and navigate to the project directory:
```bash
git clone <repo-url>
cd clinical-rag-engine
```

### 2. Create and activate a virtual environment:
```bash
# Windows
python -m venv venv
.\venv\Scripts\activate

# macOS / Linux
python -m venv venv
source venv/bin/activate
```

### 3. Install dependencies:
```bash
pip install -r requirments.txt
```

### 4. Configure Environment Variables:
Create a `.env` file in the root directory:
```env
GROQ_API_KEY=your_groq_api_key_here
GROQ_MODEL_NAME=openai/gpt-oss-20b
GROQ_BASE_URL=https://api.groq.com/openai/v1
```

## Usage

### Ingesting Clinical Trial Documents
To ingest a clinical document into the vector store:
```bash
python tests/test_ingestion.py
```
This chunks the PDF, enriches the chunks with trial metadata, and persists the vector embeddings in `data/chroma_db/`.

### Running the RAG Verification Suite
To execute the test suite covering auto-extraction, explicit filtering, citations, and guardrails:
```bash
python tests/test_rag.py
```

### Using the RAG Engine in Code
```python
from app.services.rag_engine import RagEngine
from app.schemas import DocumentMetadatafilter

rag = RagEngine()

# Example 1: Natural language query (auto-extracts drug & phase)
answer, citations, count, active_filter, cleaned_query = rag.answer_query(
    question="What are the primary considerations for sample size and statistical power in Phase III trials of Pembrolizumab?"
)

print("Answer:", answer)
for citation in citations:
    print(f"Source: {citation.source_file}, Page: {citation.page}")

# Example 2: Explicit metadata filter override
custom_filter = DocumentMetadatafilter(drug_name="Pembrolizumab", trial_phase="Phase III")
answer, citations, count, active_filter, cleaned_query = rag.answer_query(
    question="What are the reporting timelines for serious unexpected adverse events?",
    explicit_filter=custom_filter
)
```
