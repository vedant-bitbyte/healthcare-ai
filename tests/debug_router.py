# tests/debug_router.py

from src.retrieval.query_router import QueryRouter

router = QueryRouter()

queries = [
    "Which states have the fewest specialists?",
    "doctor shortage in Bihar",
    "maternal mortality rate",
    "Ayushman Bharat financing policy",
]

for query in queries:
    print("\nQUERY:", query)
    print("SOURCES:", router.route(query))