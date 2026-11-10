from functools import lru_cache
from pathlib import Path
from pydantic_settings import BaseSettings,SettingsConfigDict
from pydantic import SecretStr

class Settings(BaseSettings):
    PROJECT_NAME: str = "ClinicalTrialDoc-Insight"
    API_V1_PREFIX: str = "/api/v1"

    #Base & Local directories
    BASE_DIR: Path = Path(__file__).resolve().parent.parent
    CHROMA_PERSIST_DIR:str = str(BASE_DIR / "data" / "chroma_db")
    UPLOAD_DIR:str = str(BASE_DIR / "data" / "uploads")

    #Vector store setup
    CHROMA_COLLECTION_NAME: str = "clinical_trial_protocols"
    EMBEDDING_MODEL_NAME: str = "sentence-transformers/all-MiniLM-L6-v2"

    #Groq LLM Setup
    GROQ_API_KEY: SecretStr
    GROQ_MODEL_NAME: str = "openai/gpt-oss-20b"
    GROQ_BASE_URL: str = "https://api.groq.com/openai/v1"

    #Ingestion & Retrieval bounds
    CHUNK_SIZE:int = 700
    CHUNK_OVERLAP: int = 100
    SIMILARITY_SCORE_THRESOLD: float=0.25
    TOP_K:int=5

    model_config=SettingsConfigDict(env_file=".env", extra="ignore")

@lru_cache
def get_settings()->Settings:
    return Settings()