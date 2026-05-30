import sys
import os

# Add backend to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from app.engine.user_level_detector import UserLevelDetector

def run_tests():
    detector = UserLevelDetector()
    
    tests = [
        {
            "id": "Test 1 (should be civilian)",
            "text": "I want AI for my small bakery to answer customer questions",
            "expected": "civilian"
        },
        {
            "id": "Test 2 (should be business)",
            "text": "We need to automate customer support for our 50-person company, reduce ticket volume by 30%",
            "expected": "business"
        },
        {
            "id": "Test 3 (should be junior_dev)",
            "text": "I want to build a RAG chatbot using LangChain and ChromaDB, I have some Python experience",
            "expected": "junior_dev"
        },
        {
            "id": "Test 4 (should be senior_engineer)",
            "text": "Need hybrid RAG with BM25 + dense retrieval, Redis semantic cache, Presidio PII masking, Celery workers for async indexing, Prometheus observability",
            "expected": "senior_engineer"
        },
        {
            "id": "Test 5 (should be architect)",
            "text": "Evaluating LangGraph vs LlamaIndex for a multi-tenant RAG system, 500M vectors, multi-hop reasoning, need tradeoff analysis for checkpointing strategies at enterprise scale",
            "expected": "architect"
        }
    ]

    print("=== User Level Detector Test ===")
    for t in tests:
        level = detector.detect_persona(t["text"])
        s1 = detector._score_vocab(t["text"])
        s2 = detector._score_specificity(t["text"])
        s3 = detector._score_density(t["text"])
        s4 = detector._score_structure(t["text"])
        s5 = detector._score_memory([])
        total = s1 + s2 + s3 + s4 + s5
        
        print(f"\n{t['id']}")
        print(f"Message: '{t['text']}'")
        print(f"Detected: {level} (Expected: {t['expected']}) | Score: {total}")
        print(f"Breakdown: Vocab={s1}, Specificity={s2}, Density={s3}, Structure={s4}, Memory={s5}")
        
        if level != t['expected']:
            print(">>> MISMATCH DETECTED <<<")

if __name__ == "__main__":
    run_tests()
