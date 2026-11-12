# app/schemas.py
from typing import Optional, List, Dict
from pydantic import BaseModel, Field, model_validator


class DocumentMetadatafilter(BaseModel):
    source_file: Optional[str] = Field(
        default=None,
        description="Optional filename to scope retrieval to a single indexed PDF."
    )


class QueryRequest(BaseModel):
    question: str = Field(..., min_length=3, description="Natural language clinical question")
    filter_metadata: Optional[DocumentMetadatafilter] = Field(
        default=None,
        description="Optional filename to scope retrieval to a single indexed PDF."
    )


class Citation(BaseModel):
    source_file: str
    page: int


class QueryResponse(BaseModel):
    answer: str
    citations: List[Citation]
    chunks_retrieved: int
    extracted_filter: List[Dict[str, str]] = Field(default_factory=list)
    active_filter: Optional[List[Dict[str, str]]] = None
    cleaned_query: str

    @model_validator(mode="after")
    def sync_filters(self):
        if self.active_filter is None:
            self.active_filter = self.extracted_filter
        elif not self.extracted_filter and self.active_filter:
            self.extracted_filter = self.active_filter
        return self


class UploadResponse(BaseModel):
    filename: str
    total_chunks: int
    status: str = "Ingested successfully"


class HealthCheckResponse(BaseModel):
    status: str
    vector_store_active: bool
    persisted_documents_count: int
    embedding_model: str
    llm_model: str
