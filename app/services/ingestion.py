from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader
from langchain_core.documents import Document
import shutil
import os
import tempfile
from typing import List, Iterator, Optional
from fastapi import UploadFile, HTTPException

from app.config import get_settings
from app.services.vector_store import get_vector_store_manager

settings = get_settings()


class IngestionService:
    def __init__(self):
        self.vs_manager = get_vector_store_manager()

        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=settings.CHUNK_SIZE,
            chunk_overlap=settings.CHUNK_OVERLAP,
            separators=["\n\n", "\n", ". ", " ", ""]
        )

    # FILE I.O. & PAGE EXTRACTION
    def _stream_page(self, file: UploadFile) -> Iterator[Document]:
        temp_file_path = None
        try:
            with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tempfile_buffer:
                temp_file_path = tempfile_buffer.name
                shutil.copyfileobj(file.file, tempfile_buffer)

            loader = PyPDFLoader(temp_file_path)
            has_pages = False
            for page in loader.lazy_load():
                has_pages = True
                yield page

            if not has_pages:
                raise HTTPException(
                    status_code=400,
                    detail="The uploaded pdf contains no extractable page!! "
                )

        except HTTPException:
            raise
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Failed to parse PDF: {str(e)}")
        finally:
            if temp_file_path and os.path.exists(temp_file_path):
                os.remove(temp_file_path)
            file.file.close()

    # METADATA ENRICHMENT
    def _enrich_page_metadata(
        self,
        page: Document,
        filename: str
    ) -> Document:
        page_num = int(page.metadata.get("page", 0)) + 1
        page.metadata = {
            "source_file": filename,
            "page": page_num
        }
        return page

    # CHUNK THE PAGES
    def _chunk_page(self, file: UploadFile) -> List[Document]:
        final_chunks: List[Document] = []
        filename = (
            file.filename.strip()
            if isinstance(file.filename, str) and file.filename.strip()
            else "uploaded_document.pdf"
        )

        for raw_page in self._stream_page(file):
            enriched_page = self._enrich_page_metadata(
                page=raw_page,
                filename=filename
            )
            page_chunks = self.text_splitter.split_documents([enriched_page])
            final_chunks.extend(page_chunks)
        return final_chunks

    def _persist_chunks(self, chunks: List[Document]) -> int:
        """Generates deterministic audit IDs and pushes chunks into ChromaDB."""
        if not chunks:
            raise HTTPException(
                status_code=400,
                detail="No chunks generated from the document."
            )

        # Deterministic IDs: <filename>_p<page>_c<idx>
        chunk_ids = [
            f"{chunk.metadata['source_file']}_p{chunk.metadata['page']}_c{idx}"
            for idx, chunk in enumerate(chunks)
        ]

        store = self.vs_manager.get_store()
        store.add_documents(documents=chunks, ids=chunk_ids)

        return len(chunks)

    def ingest_clinical_pdf(self, file: UploadFile) -> dict:
        filename = (
            file.filename.strip()
            if isinstance(file.filename, str) and file.filename.strip()
            else "uploaded_document.pdf"
        )
        chunks = self._chunk_page(file=file)
        total_chunks = self._persist_chunks(chunks)
        return {
            "filename": filename,
            "total_chunks": total_chunks,
            "status": "Indexed Successfully"
        }

    _ingest_clinical_pdf = ingest_clinical_pdf
