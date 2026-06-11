import logging

from src.retrieval.retriever import Retriever

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)

retriever = Retriever()

queries = [
    "doctor shortage in India",
    "maternal health",
    "health infrastructure",
    "disease burden"
]

for query in queries:
    print("\n" + "=" * 80)
    print(f"QUERY: {query}")
    print("=" * 80)

    results = retriever.retrieve(query, top_k=3)

    for i, result in enumerate(results, start=1):
        print(f"\nResult {i}")
        print(f"Source: {result.source}")
        print(f"Distance: {result.distance:.4f}")
        print(result.text[:500])