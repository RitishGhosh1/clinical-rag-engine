#Sets up
#Local CPU Embeddings: all-MiniLM-L6-v2 with normalize_embeddings=True (critical for accurate cosine space).
#ChromaDB Client: Disk persistence in ./data/chroma_db configured with {"hnsw:space": "cosine"}.

import chromadb
from chromadb.config import Settings as ChromaClientSettings
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from app.config import get_settings

settings=get_settings()

class VectorStoreManager:
    _instance=None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(VectorStoreManager,cls).__new__(cls)
            cls._instance._init_store()
        return cls._instance
    
    def _init_store(self):
        #1. 384-dimension local CPU embeddings
        self.embeddings=HuggingFaceEmbeddings(
            model_name=settings.EMBEDDING_MODEL_NAME,
            model_kwargs={"device":"cpu"},
            encode_kwargs={"normalize_embeddings":True}
        )
        #2. Local Persistent Chroma Client
        self.chroma_client=chromadb.PersistentClient(
            path=settings.CHROMA_PERSIST_DIR,
            settings=ChromaClientSettings(anonymized_telemetry=False)
        )
        #3.Collection with explicit cosine space
        self.collection = self.chroma_client.get_or_create_collection(
            name=settings.CHROMA_COLLECTION_NAME,
            metadata={"hnsw:space":"cosine"}
        )
        #4. Langchain Chroma Wrapper for easy similarity search
        self.vector_store=Chroma(
            client=self.chroma_client,
            collection_name=settings.CHROMA_COLLECTION_NAME,
            embedding_function=self.embeddings
        )

    def get_store(self) -> Chroma:
        return self.vector_store

    def count(self) -> int:
        return self.collection.count()

def get_vector_store_manager() -> VectorStoreManager:
    return VectorStoreManager()