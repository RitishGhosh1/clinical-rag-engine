from fastapi import FastAPI,UploadFile,HTTPException,File,Form,status
from fastapi.middleware.cors import CORSMiddleware
from typing import Optional
from app.config import get_settings
from app.schemas import HealthCheckResponse,UploadResponse,QueryRequest,QueryResponse
from app.services.vector_store import get_vector_store_manager
from app.services.ingestion import IngestionService
from app.services.rag_engine import RagEngine
settings=get_settings()
vs_manager=get_vector_store_manager()
ingestion=IngestionService()
retrieval=RagEngine()

app=FastAPI(
    title=settings.PROJECT_NAME,
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

api_prefix=settings.API_V1_PREFIX

@app.get(f"{api_prefix}/health", response_model=HealthCheckResponse, tags=["Diagonstics"])
def get_health():
    try:
        try:
            doc_counts=vs_manager.count()
        except:
            return HTTPException(status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail="Cannot count the number of document collections in the Vector store database")
        
        _=vs_manager.collection.peek(limit=1)
        return HealthCheckResponse(
            status="Healthy",
            vector_store_active=True,
            persisted_documents_count=doc_counts,
            embedding_model=vs_manager.embeddings.model_name,
            llm_model=settings.GROQ_MODEL_NAME
        )
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=f"Vector store database probe failed!! {str(e)}"
        )


@app.post(f"{api_prefix}/documents/upload", response_model=UploadResponse,status_code=status.HTTP_201_CREATED,tags=["Ingestion"])
def upload_protocol_documents(
    file:UploadFile = File(..., description="Clinical Protocol File !!"),
    drug_name:Optional[str]=Form(None, description="Optional Target Compound/drug name"),
    trial_phase:Optional[str]=Form(None, description="Optional Trial Phase")
):
    if not file.filename.lower().endswith(".pdf"):
        raise HTTPException(
            status_code=status.HTTP_415_UNSUPPORTED_MEDIA_TYPE,
            detail="Unsupported file. Only .pdf files are accepted "
        )

    if file.content_type not in ["application/pdf", "application/x-pdf", "application/octet-stream"]:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Invalid MME type '{file.content_type}'. Must be application/pdf "
        )

    try:
        result=ingestion._ingest_clinical_pdf(
            file=file,
            drug_name=drug_name.strip(),
            trial_phase=trial_phase.strip()
        )

        return UploadResponse(
            filename=result["filename"],
            total_chunks=result["total_chunks"],
            drug_name=result["drug_name"],
            trial_phase=result["trial_phase"],
            status="Ingested and Indexed Successfully "
        )
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Document ingestion failed: {str(e)}"
        )


@app.post(f"{api_prefix}/query",response_model=QueryResponse,status_code=status.HTTP_200_OK,tags=["Query"])
def post_query(
    query:QueryRequest
):
    if not query:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Query String cannot be empty! "
        )
    clean_query=query.question.strip()
    try:
        answer_text,citations,chunk_count,active_filter,search_query=retrieval.answer_query(
            question=clean_query,
            explicit_filter=query.filter_metadata
        )
        return QueryResponse(
            answer=answer_text,
            citations=citations,
            chunks_retrieved=chunk_count,
            extracted_filter=active_filter,
            cleaned_query=search_query
        )
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Inference execution failed: {str(e)}"
        )

@app.get(f"{api_prefix}/documents",tags=["list ingested documents"])
def get_uploaded_documents():
    return {"documents":vs_manager.list_unique_documents()}

@app.delete(f"{api_prefix}/documents/{{filename}}",tags=["Deletion"])
def delete_protocol_documents(filename:str):
    deleted_chunks=vs_manager.delete_document_by_filename(filename)
    if deleted_chunks == 0:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"{filename} not found in vector index! "
        )
    return {
        "filename":filename,
        "deleted_chunks":deleted_chunks,
        "status":"Document successfully deleted from index! "
    }


