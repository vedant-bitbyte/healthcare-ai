"""Build and manage a ChromaDB vector store from ingested JSON chunks."""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

import chromadb
from chromadb.api.models.Collection import Collection

from src.embeddings.embedding_model import EmbeddingModelError, embed_texts

logger = logging.getLogger(__name__)

DEFAULT_CHUNKS_DIR = Path("data/processed")
DEFAULT_PERSIST_DIR = Path("vector_store")
DEFAULT_COLLECTION_NAME = "healthcare_chunks"


class VectorStoreError(Exception):
    """Raised when vector store operations fail."""


def _make_document_id(source: str, chunk_id: int) -> str:
    """Create a stable, unique ChromaDB document ID."""
    safe_source = source.replace(" ", "_")
    return f"{safe_source}::{chunk_id}"


def load_chunk_records(json_path: str | Path) -> list[dict[str, Any]]:
    """Load chunk records from a single JSON file.

    Args:
        json_path: Path to a chunk JSON file produced by the ingestion pipeline.

    Returns:
        List of chunk dictionaries with ``chunk_id``, ``source``, and ``text``.

    Raises:
        FileNotFoundError: If the JSON file does not exist.
        VectorStoreError: If the file format is invalid.
    """
    path = Path(json_path).resolve()

    if not path.exists():
        raise FileNotFoundError(f"Chunk JSON file not found: {path}")

    try:
        with path.open("r", encoding="utf-8") as file:
            records = json.load(file)
    except json.JSONDecodeError as exc:
        raise VectorStoreError(f"Invalid JSON in '{path.name}': {exc}") from exc
    except OSError as exc:
        raise VectorStoreError(f"Could not read '{path.name}': {exc}") from exc

    if not isinstance(records, list):
        raise VectorStoreError(f"Expected a JSON list in '{path.name}'")

    validated: list[dict[str, Any]] = []
    for index, record in enumerate(records):
        if not isinstance(record, dict):
            raise VectorStoreError(
                f"Invalid record at index {index} in '{path.name}': expected object"
            )

        for field in ("chunk_id", "source", "text"):
            if field not in record:
                raise VectorStoreError(
                    f"Missing '{field}' in record {index} of '{path.name}'"
                )

        if not str(record["text"]).strip():
            logger.debug("Skipping empty chunk %s in '%s'", record["chunk_id"], path.name)
            continue

        validated.append(record)

    logger.info("Loaded %d chunk(s) from %s", len(validated), path.name)
    return validated


def load_chunks_from_directory(chunks_dir: str | Path) -> list[dict[str, Any]]:
    """Load chunk records from all JSON files in a directory.

    Args:
        chunks_dir: Directory containing ``*_chunks.json`` files.

    Returns:
        Combined list of chunk records from all JSON files.

    Raises:
        FileNotFoundError: If the directory does not exist.
        VectorStoreError: If the path is not a directory.
    """
    directory = Path(chunks_dir).resolve()

    if not directory.exists():
        raise FileNotFoundError(f"Chunks directory not found: {directory}")

    if not directory.is_dir():
        raise VectorStoreError(f"Path is not a directory: {directory}")

    json_files = sorted(directory.glob("*.json"))
    if not json_files:
        logger.warning("No JSON chunk files found in %s", directory)
        return []

    all_chunks: list[dict[str, Any]] = []
    for json_file in json_files:
        all_chunks.extend(load_chunk_records(json_file))

    logger.info(
        "Loaded %d total chunk(s) from %d JSON file(s) in %s",
        len(all_chunks),
        len(json_files),
        directory,
    )
    return all_chunks


class VectorStore:
    """ChromaDB-backed vector store for healthcare document chunks."""

    def __init__(
        self,
        persist_directory: str | Path = DEFAULT_PERSIST_DIR,
        collection_name: str = DEFAULT_COLLECTION_NAME,
    ) -> None:
        """Initialize the vector store configuration.

        Args:
            persist_directory: Directory where ChromaDB persists its data.
            collection_name: Name of the ChromaDB collection.
        """
        self.persist_directory = Path(persist_directory)
        self.collection_name = collection_name
        self._client: chromadb.ClientAPI | None = None
        self._collection: Collection | None = None

    def _get_client(self) -> chromadb.ClientAPI:
        """Return a persistent ChromaDB client."""
        if self._client is None:
            self.persist_directory.mkdir(parents=True, exist_ok=True)
            logger.info("Initializing ChromaDB at %s", self.persist_directory)
            try:
                self._client = chromadb.PersistentClient(
                    path=str(self.persist_directory),
                )
            except Exception as exc:
                logger.exception("Failed to initialize ChromaDB client")
                raise VectorStoreError(f"ChromaDB initialization failed: {exc}") from exc
        return self._client

    def get_collection(self) -> Collection:
        """Return the ChromaDB collection, creating it if needed."""
        if self._collection is None:
            client = self._get_client()
            try:
                self._collection = client.get_or_create_collection(
                    name=self.collection_name,
                    metadata={"hnsw:space": "cosine"},
                )
            except Exception as exc:
                logger.exception("Failed to get or create collection '%s'", self.collection_name)
                raise VectorStoreError(
                    f"Could not access collection '{self.collection_name}': {exc}"
                ) from exc
            logger.info("Using ChromaDB collection '%s'", self.collection_name)
        return self._collection

    def index_chunks(self, chunks: list[dict[str, Any]]) -> int:
        """Generate embeddings and upsert chunks into ChromaDB.

        Args:
            chunks: Chunk records with ``chunk_id``, ``source``, and ``text``.

        Returns:
            Number of chunks indexed.

        Raises:
            VectorStoreError: If indexing fails.
            EmbeddingModelError: If embedding generation fails.
        """
        if not chunks:
            logger.warning("No chunks provided for indexing")
            return 0

        ids = [_make_document_id(str(chunk["source"]), int(chunk["chunk_id"])) for chunk in chunks]
        documents = [str(chunk["text"]) for chunk in chunks]
        metadatas = [
            {
                "chunk_id": int(chunk["chunk_id"]),
                "source": str(chunk["source"]),
            }
            for chunk in chunks
        ]

        logger.info("Generating embeddings for %d chunk(s)", len(chunks))

        try:
            embeddings = embed_texts(documents)
        except EmbeddingModelError:
            raise
        except Exception as exc:
            logger.exception("Unexpected error during embedding generation")
            raise VectorStoreError(f"Embedding generation failed: {exc}") from exc

        collection = self.get_collection()

        batch_size = 100

        try:
            for start in range(0, len(chunks), batch_size):
                end = min(start + batch_size, len(chunks))
                collection.upsert(
                    ids=ids[start:end],
                    documents=documents[start:end],
                    embeddings=embeddings[start:end],
                    metadatas=metadatas[start:end],
                )
                logger.debug("Upserted batch %d-%d into ChromaDB", start + 1, end)
        except Exception as exc:
            logger.exception("Failed to upsert chunks into ChromaDB")
            raise VectorStoreError(f"ChromaDB upsert failed: {exc}") from exc

        logger.info("Indexed %d chunk(s) into collection '%s'", len(chunks), self.collection_name)
        return len(chunks)

    def index_chunks_directory(self, chunks_dir: str | Path = DEFAULT_CHUNKS_DIR) -> int:
        """Load all JSON chunk files from a directory and index them.

        Args:
            chunks_dir: Directory containing chunk JSON files.

        Returns:
            Number of chunks indexed.
        """
        chunks = load_chunks_from_directory(chunks_dir)
        return self.index_chunks(chunks)

    def count(self) -> int:
        """Return the number of documents in the collection."""
        return self.get_collection().count()
