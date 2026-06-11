import logging

from src.embeddings.vector_store import VectorStore

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)

store = VectorStore()

count = store.index_chunks_directory()

print(f"\nIndexed Chunks: {count}")
print(f"Collection Count: {store.count()}")