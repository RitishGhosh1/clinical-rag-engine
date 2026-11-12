from fastapi import FastAPI, UploadFile, HTTPException, File, status
from fastapi.middleware.cors import CORSMiddleware
from typing import List, Dict, Any
from app.config import get_settings
from app.schemas import HealthCheckResponse, UploadResponse, QueryRequest, QueryResponse
from app.services.vector_store import get_vector_store_manager
from app.services.ingestion import IngestionService
from app.services.rag_engine import RagEngine

settings = get_settings()
vs_manager = get_vector_store_manager()
ingestion = IngestionService()
retrieval = RagEngine()

app = FastAPI(
    title=settings.PROJECT_NAME,
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

api_prefix = settings.API_V1_PREFIX


@app.get(f"{api_prefix}/health", response_model=HealthCheckResponse, tags=["Diagnostics"])
def get_health():
    try:
        try:
            doc_counts = vs_manager.count()
        except Exception:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="Cannot count the number of document collections in the Vector store database"
            )

        _ = vs_manager.collection.peek(limit=1)
        return HealthCheckResponse(
            status="Healthy",
            vector_store_active=True,
            persisted_documents_count=doc_counts,
            embedding_model=vs_manager.embeddings.model_name,
            llm_model=settings.GROQ_MODEL_NAME
        )
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=f"Vector store database probe failed!! {str(e)}"
        )


@app.post(
    f"{settings.API_V1_PREFIX}/documents/upload",
    response_model=UploadResponse,
    status_code=status.HTTP_201_CREATED,
    tags=["Ingestion"]
)
def upload_protocol_document(
    file: UploadFile = File(...)
):
    filename = file.filename or "uploaded_document.pdf"
    if not filename.lower().endswith(".pdf"):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Invalid file extension. Only .pdf files are accepted."
        )

    try:
        result = ingestion.ingest_clinical_pdf(file=file)

        return UploadResponse(
            filename=result["filename"],
            total_chunks=result["total_chunks"],
            status=result["status"]
        )
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Document ingestion failed: {str(exc)}"
        )


@app.post(f"{api_prefix}/query", response_model=QueryResponse, status_code=status.HTTP_200_OK, tags=["Query"])
def post_query(
    query: QueryRequest
):
    if not query or not query.question or not query.question.strip():
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Query String cannot be empty! "
        )
    clean_query = query.question.strip()
    try:
        answer_text, citations, chunk_count, active_filter, search_query = retrieval.answer_query(
            question=clean_query,
            explicit_filter=query.filter_metadata
        )
        return QueryResponse(
            answer=answer_text,
            citations=citations,
            chunks_retrieved=chunk_count,
            extracted_filter=active_filter,
            active_filter=active_filter,
            cleaned_query=search_query
        )
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Inference execution failed: {str(e)}"
        )


@app.get(f"{api_prefix}/documents", response_model=List[Dict[str, Any]], tags=["list ingested documents"])
def get_uploaded_documents():
    return vs_manager.list_unique_documents()


@app.delete(f"{api_prefix}/documents/{{filename}}", tags=["Deletion"])
def delete_protocol_documents(filename: str):
    deleted_chunks = vs_manager.delete_document_by_filename(filename)
    if deleted_chunks == 0:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"{filename} not found in vector index! "
        )
    return {
        "filename": filename,
        "deleted_chunks": deleted_chunks,
        "status": "Document successfully deleted from index! "
    }
