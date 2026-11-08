import os
import shutil
from pathlib import Path
from fastapi import UploadFile

from app.config import get_settings
from app.services.vector_store import get_vector_store_manager
from app.services.ingestion import IngestionService

settings = get_settings()

def create_sample_pdf(filepath: str):
    """Generates a minimal 2-page test PDF if you don't have one handy."""
    from pypdf import PdfWriter
    
    # We create a simple empty/text PDF using pypdf directly
    writer = PdfWriter()
    writer.add_blank_page(width=200, height=200)
    writer.add_blank_page(width=200, height=200)
    
    with open(filepath, "wb") as f:
        writer.write(f)
    print(f"Created dummy blank PDF at: {filepath}")

def run_test():
    print("\n--- 1. Initializing Vector Store ---")
    vs_manager = get_vector_store_manager()
    initial_count = vs_manager.count()
    print(f"Current documents in ChromaDB collection: {initial_count}")

    # Use a real PDF if you have one, or generate a dummy one
    test_pdf_path = "test_protocol.pdf"
    
    if not os.path.exists(test_pdf_path):
        print("\nNo 'test_protocol.pdf' found in root. Generating a mock PDF...")
        # If you have an actual clinical trial PDF, drop it in root and name it test_protocol.pdf!
        # Note: blank PDFs have no text, so test with a PDF that contains readable text.
        print(">> TIP: Place a real PDF with text as 'test_protocol.pdf' to test extraction.")
        return

    print(f"\n--- 2. Simulating UploadFile for: {test_pdf_path} ---")
    with open(test_pdf_path, "rb") as f:
        # Simulate FastAPI UploadFile
        upload_file = UploadFile(
            file=f,
            filename="KEYNOTE_001_Phase3_Brief.pdf"
        )

        service = IngestionService()
        
        print("\n--- 3. Running Ingestion Pipeline ---")
        result = service._ingest_clinical_pdf(
            file=upload_file,
            drug_name="Pembrolizumab",
            trial_phase="Phase III"
        )
        print(f"Ingestion result: {result}")

    print("\n--- 4. Verifying ChromaDB Persistence ---")
    new_count = vs_manager.count()
    print(f"Updated collection count: {new_count} (Added {new_count - initial_count} chunks)")

    # Inspect the newly added chunks in ChromaDB
    collection = vs_manager.collection
    sample_data = collection.peek(limit=2)
    
    print("\n--- 5. Inspecting Sample Ingested Chunk ---")
    if sample_data and sample_data["ids"]:
        print(f"Chunk ID: {sample_data['ids'][0]}")
        print(f"Metadata: {sample_data['metadatas'][0]}")
        print(f"Content Preview:\n{sample_data['documents'][0][:150]}...")
    else:
        print("Warning: Collection is empty!")

if __name__ == "__main__":
    run_test()