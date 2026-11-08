from typing import Optional,List
from pydantic import BaseModel,Field

class DocumentMetadatafilter(BaseModel):
    drug_name: Optional[str]=Field(default=None,description="Exact drug name filter! ")
    trial_phase:Optional[str]=Field(default=None, description="e.g., Phase I, Phase II, Phase III")

class QueryRequest(BaseModel):
    query: str=Field(..., min_length=5, description="Clinical Query or adverse event lookup!! ")
    filter_metadata: Optional[DocumentMetadatafilter]=Field(
        default=None,
        description="Optional Metadata filtering constraints! If None, search across all platforms!! "
    )

class Citation(BaseModel):
    source_file: str
    page: int 
    drug_name: Optional[str]=None
    trial_phase: Optional[str]=None

class QueryResponse(BaseModel):
    answer:str
    citations:List[Citation]
    chunks_retrieved:int

class UploadDocuments(BaseModel):
    filename:str
    total_pages:int
    total_chunks:int
    drug_name:Optional[str]
    trial_phase:Optional[str]
    status: str = "Ingested Successfully!!"

