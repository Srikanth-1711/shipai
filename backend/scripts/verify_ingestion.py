"""
ShipAI -- Quick verification of book ingestion quality.
Run after ingesting a book to verify all 4 levels are populated.
"""
import chromadb
import sys

CHROMA_DIR = "knowledge_base/chroma_db"

def verify():
    client = chromadb.PersistentClient(path=CHROMA_DIR)
    
    try:
        col = client.get_collection("shipai_books")
    except Exception:
        print("ERROR: Collection 'shipai_books' not found. Run ingestion first.")
        sys.exit(1)
    
    total = col.count()
    print(f"Total documents in index: {total}")
    
    if total == 0:
        print("ERROR: Index is empty. Ingestion did not produce any documents.")
        sys.exit(1)
    
    # Check all 4 levels
    all_good = True
    for level in ["book", "chapter", "chunk", "principle"]:
        try:
            results = col.get(where={"level": level})
            count = len(results["ids"])
            print(f"  Level '{level}': {count} items")
            if count == 0:
                print(f"    WARNING: Level '{level}' is empty!")
                all_good = False
        except Exception as e:
            print(f"  Level '{level}': ERROR - {e}")
            all_good = False
    
    # Check for zero-vector embeddings (the bug we just fixed)
    print("\n--- Embedding Quality Check ---")
    try:
        sample = col.query(
            query_texts=["data replication and partitioning"],
            where={"level": "chunk"},
            n_results=3,
        )
        if sample["documents"] and sample["documents"][0]:
            print(f"  Query 'data replication and partitioning' returned {len(sample['documents'][0])} results")
            for i, doc in enumerate(sample["documents"][0]):
                dist = sample["distances"][0][i] if sample["distances"] else "?"
                print(f"    [{i+1}] distance={dist:.4f} | {doc[:80]}...")
            
            # If all distances are 0 or very close, embeddings are likely zero vectors
            if sample["distances"] and all(d < 0.001 for d in sample["distances"][0]):
                print("  WARNING: All distances near 0 -- embeddings may be zero vectors!")
                all_good = False
        else:
            print("  WARNING: No chunk results returned")
            all_good = False
    except Exception as e:
        print(f"  Query failed: {e}")
        all_good = False
    
    # Sample a principle
    print("\n--- Principle Quality Check ---")
    try:
        principles = col.query(
            query_texts=["when to use async processing"],
            where={"level": "principle"},
            n_results=2,
        )
        if principles["documents"] and principles["documents"][0]:
            print(f"  Found {len(principles['documents'][0])} principles")
            for i, p in enumerate(principles["documents"][0]):
                print(f"  Principle {i+1}:")
                print(f"    {p[:300]}")
                print()
        else:
            print("  WARNING: No principles found")
            all_good = False
    except Exception as e:
        print(f"  Principle query failed: {e}")
        all_good = False
    
    if all_good:
        print("\n=== VERIFICATION PASSED ===")
    else:
        print("\n=== VERIFICATION FAILED -- Fix issues above ===")
        sys.exit(1)


if __name__ == "__main__":
    verify()
