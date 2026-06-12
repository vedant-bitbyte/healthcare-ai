import logging

from src.rag.rag_pipeline import RAGPipeline

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)

rag = RAGPipeline()

questions = [
    "Which states have the fewest specialists?",
    "Which states have doctor shortages?",
    "Which states require healthcare workforce investment?",
    "What are the healthcare infrastructure challenges in India?",
]

for question in questions:
    print("\n" + "=" * 100)
    print(question)
    print("=" * 100)

    result = rag.run(question)

    print("\nANSWER:\n")
    print(result["answer"])

    print("\nSOURCES:")
    print(result["sources"])

    print("\nCHUNKS:")
    for chunk in result["retrieved_chunks"]:
        print(chunk)