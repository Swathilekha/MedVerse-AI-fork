import sys
from pathlib import Path
from functools import lru_cache

# Locate the existing RAG implementation
PROJECT_ROOT = Path(__file__).resolve().parents[3]
RAG_DIR = PROJECT_ROOT / "ai-services" / "clinical"

if str(RAG_DIR) not in sys.path:
    sys.path.insert(0, str(RAG_DIR))

import ollama_rag


@lru_cache(maxsize=1)
def get_rag_index():
    return ollama_rag.build_index()


def ask_rag(question: str):
    question = question.strip()

    if not question:
        raise ValueError("Question cannot be empty.")

    index = get_rag_index()

    evidence = ollama_rag.retrieve(question, index)

    if not evidence:
        return {
            "status": "abstained",
            "answer": "No relevant evidence found in the available knowledge base.",
            "sources": []
        }

    answer = ollama_rag.generate_answer(
        question,
        evidence
    )

    return {
        "status": "answered",
        "answer": answer,
        "sources": [
            {
                "id": item["id"],
                "section": item["section"],
                "similarity": round(item["score"], 4),
                "url": item["url"]
            }
            for item in evidence
        ]
    }