# app/schemas.py
from typing import Optional, List
from pydantic import BaseModel, Field


class DocumentMetadatafilter(BaseModel):
    drug_name: Optional[str] = Field(default=None, description="Exact drug name filter")
    trial_phase: Optional[str] = Field(default=None, description="e.g., Phase I, Phase II, Phase III, Phase IV")


class ExtractedQueryIntent(BaseModel):
    """Structured output schema for LLM query intent extraction."""
    cleaned_query: str = Field(
        ...,
        description="The semantic core clinical question with specific entity constraints (e.g. drug name, phase) removed."
    )
    drug_name: Optional[str] = Field(
        default=None,
        description="The identified pharmaceutical/drug name (e.g., 'Pembrolizumab') normalized to Title Case, or null if none."
    )
    trial_phase: Optional[str] = Field(
        default=None,
        description="Standardized trial phase: strictly 'Phase I', 'Phase II', 'Phase III', 'Phase IV', or null if none."
    )


class QueryRequest(BaseModel):
    question: str = Field(..., min_length=3, description="Natural language clinical question")
    filter_metadata: Optional[DocumentMetadatafilter] = Field(
        default=None,
        description="Optional explicit metadata override. If omitted, extracted automatically from question."
    )


class Citation(BaseModel):
    source_file: str
    page: int
    drug_name: Optional[str] = None
    trial_phase: Optional[str] = None


class QueryResponse(BaseModel):
    answer: str
    citations: List[Citation]
    chunks_retrieved: int
    extracted_filter: Optional[DocumentMetadatafilter] = None
    cleaned_query: str


class UploadResponse(BaseModel):
    filename: str
    total_chunks: int
    drug_name: Optional[str]
    trial_phase: Optional[str]
    status: str = "Ingested successfully"


class HealthCheckResponse(BaseModel):
    status: str
    vector_store_active: bool
    persisted_documents_count: int
    embedding_model: str
    llm_model: str