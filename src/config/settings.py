"""Centralized configuration management for AI Sarthi."""

import os
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables from .env file if present
load_dotenv()

# Base Paths
BASE_DIR = Path(__file__).resolve().parent.parent.parent
DATA_DIR = BASE_DIR / "data"
MODELS_DIR = BASE_DIR / "models"
DOCS_DIR = BASE_DIR / "docs"

# Model Configurations
DEFAULT_MODEL = os.getenv("DEFAULT_MODEL", "fine-tuned")
OLLAMA_HOST = os.getenv("OLLAMA_HOST", "http://localhost:11434")
HF_BASE_MODEL = os.getenv("HF_BASE_MODEL", "microsoft/Phi-3-mini-4k-instruct")
HF_ADAPTER_PATH = os.getenv("HF_ADAPTER_PATH", str(MODELS_DIR / "outputs"))

# Vector Store Configurations
CHROMA_PERSIST_DIR = os.getenv("CHROMA_PERSIST_DIR", str(BASE_DIR / "vector_store"))
CHROMA_COLLECTION_NAME = os.getenv("CHROMA_COLLECTION_NAME", "healthcare_chunks")

# Retrieval Configurations
DEFAULT_TOP_K = int(os.getenv("DEFAULT_TOP_K", "15"))

# Embedding Configurations
DEFAULT_EMBEDDING_MODEL = os.getenv("DEFAULT_EMBEDDING_MODEL", "BAAI/bge-small-en-v1.5")
