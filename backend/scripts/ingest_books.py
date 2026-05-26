"""
ShipAI -- Multi-Level Book Ingestion Pipeline
==============================================
NotebookLM-grade knowledge indexing with 4 levels:
  Level 1: Full book summary (LLM-generated)
  Level 2: Chapter/section summaries (LLM-generated)
  Level 3: Semantic chunks (standard vector embeddings)
  Level 4: Principle extraction (actionable engineering rules)

Usage:
  python backend/scripts/ingest_books.py                              # All books, all levels
  python backend/scripts/ingest_books.py --layer layer1_data_systems  # One layer only
  python backend/scripts/ingest_books.py --book "Designing Data"      # One book only
  python backend/scripts/ingest_books.py --fast                       # L3 only (no LLM, fast)
  python backend/scripts/ingest_books.py --llm-only                   # L1/L2/L4 only (LLM heavy)
  python backend/scripts/ingest_books.py --resume                     # Resume from checkpoint
"""
import os
import sys
import glob
import json
import argparse
import hashlib
import time
from pathlib import Path

# Add backend to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import chromadb
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
import httpx

# --- Configuration -------------------------------------------------------
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
BOOKS_DIR = os.path.join(PROJECT_ROOT, "knowledge_base", "books")
CHROMA_DIR = os.path.join(PROJECT_ROOT, "knowledge_base", "chroma_db")
PROGRESS_FILE = os.path.join(PROJECT_ROOT, "knowledge_base", ".ingestion_progress.json")
OLLAMA_URL = "http://localhost:11434"
EMBED_MODEL = "nomic-embed-text"
LLM_MODEL = "qwen2.5:3b"

LAYER_MAP = {
    "layer1_data_systems":    {"layer": "data",   "focus": "data engineering, databases, streaming, pipelines"},
    "layer2_ml_systems":      {"layer": "ml",     "focus": "ML system design, MLOps, model deployment"},
    "layer3_llm_genai":       {"layer": "llm",    "focus": "LLM applications, transformers, GenAI services"},
    "layer4_production_ops":  {"layer": "ops",    "focus": "production systems, SRE, microservices, architecture"},
    "layer5_math_theory":     {"layer": "math",   "focus": "mathematics, deep learning theory, probability"},
    "layer6_domain_business": {"layer": "domain", "focus": "business strategy, NLP, alignment, domain knowledge"},
}


# --- Progress / Checkpoint -----------------------------------------------
def load_progress() -> dict:
    if Path(PROGRESS_FILE).exists():
        try:
            return json.load(open(PROGRESS_FILE))
        except Exception:
            pass
    return {"completed_l3": [], "completed_llm": [], "failed": []}


def save_progress(progress: dict):
    os.makedirs(os.path.dirname(PROGRESS_FILE), exist_ok=True)
    with open(PROGRESS_FILE, "w") as f:
        json.dump(progress, f, indent=2)


# --- Embedding via Ollama -------------------------------------------------
def embed_texts(texts: list[str], batch_size: int = 20) -> list[list[float]]:
    """Embed texts using local Ollama nomic-embed-text model."""
    all_embeddings = []
    failures = 0
    for i in range(0, len(texts), batch_size):
        batch = texts[i:i + batch_size]
        for text in batch:
            try:
                resp = httpx.post(
                    f"{OLLAMA_URL}/api/embed",
                    json={"model": EMBED_MODEL, "input": text},
                    timeout=60,
                )
                if resp.status_code == 200:
                    data = resp.json()
                    # Handle both /api/embed (new) and /api/embeddings (old) response formats
                    if "embeddings" in data:
                        all_embeddings.append(data["embeddings"][0])
                    elif "embedding" in data:
                        all_embeddings.append(data["embedding"])
                    else:
                        failures += 1
                        all_embeddings.append([0.0] * 768)
                else:
                    failures += 1
                    all_embeddings.append([0.0] * 768)
            except Exception as e:
                failures += 1
                all_embeddings.append([0.0] * 768)
        done = min(i + batch_size, len(texts))
        if len(texts) > batch_size:
            print(f"    Embedded {done}/{len(texts)} chunks", end="")
            if failures > 0:
                print(f" ({failures} failures)", end="")
            print()
    if failures > 0:
        print(f"    [WARN] {failures}/{len(texts)} embeddings failed")
    return all_embeddings


def llm_generate(prompt: str, max_tokens: int = 2048) -> str:
    """Generate text using local Ollama LLM."""
    try:
        resp = httpx.post(
            f"{OLLAMA_URL}/api/generate",
            json={
                "model": LLM_MODEL,
                "prompt": prompt,
                "stream": False,
                "options": {"temperature": 0.3, "num_predict": max_tokens},
            },
            timeout=180,
        )
        if resp.status_code == 200:
            return resp.json().get("response", "")
    except Exception as e:
        print(f"    [WARN] LLM generation failed: {e}")
    return ""


def make_id(text: str) -> str:
    return hashlib.md5(text.encode()).hexdigest()[:16]


# --- Level 3: Semantic Chunks (FAST, no LLM) -----------------------------
def ingest_level3_chunks(collection, book_title: str, full_text: str, meta: dict):
    """Standard semantic chunking for fine-grained retrieval. No LLM needed."""
    print(f"  [L3] Chunking and embedding...")

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len,
    )
    chunks = splitter.split_text(full_text)

    # Cap at 200 chunks per book
    if len(chunks) > 200:
        step = len(chunks) // 200
        chunks = chunks[::step][:200]

    ids = [make_id(f"L3_{book_title}_{i}") for i in range(len(chunks))]
    metadatas = [
        {
            "title": book_title,
            "level": "chunk",
            "layer": meta["layer"],
            "chunk_index": i,
            "type": "semantic_chunk",
        }
        for i in range(len(chunks))
    ]

    embeddings = embed_texts(chunks)
    collection.upsert(ids=ids, documents=chunks, embeddings=embeddings, metadatas=metadatas)
    print(f"  [L3] {len(chunks)} semantic chunks stored")


# --- Level 1: Book Summary (LLM) -----------------------------------------
def ingest_level1_summary(collection, book_title: str, full_text: str, meta: dict):
    print(f"  [L1] Generating book summary...")

    excerpt = full_text[:6000]
    prompt = f"""You are indexing a technical book for an AI engineering platform called ShipAI.
Book title: {book_title}
Book focus area: {meta.get('focus', 'engineering')}

Based on the following excerpt from the book, write a comprehensive summary covering:
1. The 5 most important engineering principles from this book
2. What types of systems or architectures this book helps you build
3. Key patterns, anti-patterns, and trade-offs discussed
4. When an AI architect should reference this book

Book excerpt:
{excerpt}

Write a detailed, actionable summary (300-500 words):"""

    summary = llm_generate(prompt)
    if not summary:
        summary = f"Book: {book_title}. Content excerpt: {excerpt[:500]}"

    doc_id = make_id(f"L1_{book_title}")
    emb = embed_texts([summary])
    collection.upsert(
        ids=[doc_id],
        documents=[summary],
        embeddings=emb,
        metadatas=[{
            "title": book_title,
            "level": "book",
            "layer": meta["layer"],
            "type": "summary",
        }],
    )
    print(f"  [L1] Book summary stored ({len(summary)} chars)")


# --- Level 2: Chapter Summaries (LLM) ------------------------------------
def ingest_level2_chapters(collection, book_title: str, full_text: str, meta: dict):
    print(f"  [L2] Generating chapter summaries...")

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=4000,
        chunk_overlap=200,
        separators=["\nChapter ", "\nPART ", "\nSection ", "\n\n\n", "\n\n"],
    )
    sections = splitter.split_text(full_text)
    sections = sections[:15]

    ids = []
    documents = []
    metadatas = []

    for i, section in enumerate(sections):
        prompt = f"""Extract the key engineering principles from this section of "{book_title}".
What decisions does this inform for building AI systems?
Be specific and actionable. 100-200 words max.

Section text:
{section[:3000]}

Key principles:"""

        chapter_summary = llm_generate(prompt, max_tokens=512)
        if not chapter_summary:
            chapter_summary = section[:500]

        doc_id = make_id(f"L2_{book_title}_{i}")
        ids.append(doc_id)
        documents.append(chapter_summary)
        metadatas.append({
            "title": book_title,
            "level": "chapter",
            "layer": meta["layer"],
            "section_index": i,
            "type": "chapter_summary",
        })

    embeddings = embed_texts(documents)
    collection.upsert(ids=ids, documents=documents, embeddings=embeddings, metadatas=metadatas)
    print(f"  [L2] {len(documents)} chapter summaries stored")


# --- Level 4: Principle Extraction (LLM) ---------------------------------
def ingest_level4_principles(collection, book_title: str, full_text: str, meta: dict):
    print(f"  [L4] Extracting engineering principles...")

    # Sample from multiple parts of the book for better coverage
    total_len = len(full_text)
    excerpts = []
    if total_len > 0:
        excerpts.append(full_text[:4000])  # beginning
    if total_len > 8000:
        mid = total_len // 2
        excerpts.append(full_text[mid:mid + 4000])  # middle
    if total_len > 16000:
        excerpts.append(full_text[-4000:])  # end

    combined_excerpt = "\n\n---\n\n".join(excerpts)

    prompt = f"""You are ShipAI's Lead Architect extracting hardcore engineering principles from "{book_title}".

CRITICAL INSTRUCTIONS:
Extract exactly 8 concrete, deeply technical, and actionable engineering principles.
DO NOT write generic truisms like "handle failures gracefully" or "ensure data quality".
Every principle MUST contain specific architectural patterns, algorithms, or technical tradeoffs.

STRICT RULES — violation means the principle is rejected:
Rule 1: Every principle must name a specific algorithm, data structure, or named pattern.
        REJECT: "Handle failures gracefully"
        ACCEPT: "Use exponential backoff with jitter (not fixed retry intervals) when retrying failed network calls — fixed intervals cause thundering herd under load"

Rule 2: Every principle must state a specific condition (when to apply) AND a specific consequence (what happens if you don't).
        REJECT: "Use caching to improve performance"
        ACCEPT: "Cache embedding lookups in Redis with TTL=3600s when query volume exceeds 100 req/min — without this, identical queries re-embed at full LLM cost, burning 40-60% of inference budget on redundant computation"

Rule 3: Every principle must be falsifiable — a senior engineer must be able to disagree with it.
        REJECT: "Design for scalability"  
        ACCEPT: "Prefer vertical scaling over horizontal sharding until you exceed 10TB or 100k writes/sec — premature sharding adds operational complexity that kills small team velocity"

For EACH principle, provide exactly one paragraph containing:
1. The Principle (highly technical, naming specific patterns/tools/algorithms)
2. When to apply it (specific scale, query type, or system state)
3. The anti-pattern it prevents

Number each principle 1 through 8.

Book excerpts (use these to find the most technical patterns discussed):
{combined_excerpt}

8 Deeply Technical Engineering Principles:"""

    principles_text = llm_generate(prompt, max_tokens=3000)
    if not principles_text:
        print(f"  [L4] Skipped (LLM failed)")
        return

    # Parse numbered items
    principles = []
    current = ""
    for line in principles_text.split("\n"):
        stripped = line.strip()
        if stripped and len(stripped) > 2:
            # Detect numbered items: "1.", "1)", "1:", etc.
            first_chars = stripped[:4]
            if any(c.isdigit() for c in first_chars) and any(c in first_chars for c in ".):"):
                if current.strip():
                    principles.append(current.strip())
                current = stripped
            else:
                current += " " + stripped
    if current.strip():
        principles.append(current.strip())

    # Filter out empty or too-short principles
    principles = [p for p in principles if len(p) > 30]

    if not principles:
        principles = [principles_text[:500]]

    ids = [make_id(f"L4_{book_title}_{i}") for i in range(len(principles))]
    metadatas = [
        {
            "title": book_title,
            "level": "principle",
            "layer": meta["layer"],
            "principle_index": i,
            "type": "principle",
        }
        for i in range(len(principles))
    ]

    embeddings = embed_texts(principles)
    collection.upsert(ids=ids, documents=principles, embeddings=embeddings, metadatas=metadatas)
    print(f"  [L4] {len(principles)} principles extracted and stored")


# --- Main Pipeline --------------------------------------------------------
def extract_text(pdf_path: str) -> str:
    """Load a PDF and return full text."""
    loader = PyPDFLoader(pdf_path)
    docs = loader.load()
    return "\n".join([d.page_content for d in docs])


def ingest_single_book(collection, pdf_path: str, layer_name: str, mode: str = "all"):
    """Run ingestion on a single book.
    mode: 'all' | 'fast' (L3 only) | 'llm' (L1/L2/L4 only)
    """
    filename = os.path.basename(pdf_path)
    book_title = filename.replace(".pdf", "")
    meta = LAYER_MAP.get(layer_name, {"layer": "domain", "focus": "general"})

    print(f"\n{'='*60}")
    print(f"INGESTING: {book_title[:55]}")
    print(f"  Layer: {meta['layer']} | Mode: {mode}")
    print(f"{'='*60}")

    print(f"  Loading PDF...")
    try:
        full_text = extract_text(pdf_path)
    except Exception as e:
        print(f"  [ERROR] Failed to load PDF: {e}")
        return False

    print(f"  Extracted {len(full_text)} characters")

    if len(full_text) < 100:
        print(f"  [SKIP] Too little text extracted. Possibly a scanned/image PDF.")
        return False

    start = time.time()

    if mode in ("all", "fast"):
        ingest_level3_chunks(collection, book_title, full_text, meta)

    if mode in ("all", "llm"):
        ingest_level1_summary(collection, book_title, full_text, meta)
        ingest_level2_chapters(collection, book_title, full_text, meta)
        
    if mode in ("all", "llm", "l4"):
        ingest_level4_principles(collection, book_title, full_text, meta)

    elapsed = time.time() - start
    print(f"  DONE in {elapsed:.1f}s")
    return True


def get_all_pdfs(layer_filter: str = None, book_filter: str = None) -> list[tuple[str, str]]:
    """Returns list of (pdf_path, layer_name) tuples."""
    results = []
    for layer_name in sorted(LAYER_MAP.keys()):
        if layer_filter and layer_filter != layer_name:
            continue
        layer_dir = os.path.join(BOOKS_DIR, layer_name)
        if not os.path.exists(layer_dir):
            continue
        for pdf_path in sorted(glob.glob(os.path.join(layer_dir, "*.pdf"))):
            if book_filter and book_filter.lower() not in os.path.basename(pdf_path).lower():
                continue
            results.append((pdf_path, layer_name))
    return results


def main():
    parser = argparse.ArgumentParser(description="ShipAI Multi-Level Book Ingestion")
    parser.add_argument("--layer", type=str, default=None, help="Only ingest books from this layer folder")
    parser.add_argument("--book", type=str, default=None, help="Only ingest books matching this name substring")
    parser.add_argument("--fast", action="store_true", help="L3 only (embeddings, no LLM). Fast pass.")
    parser.add_argument("--llm-only", action="store_true", help="L1/L2/L4 only (LLM-heavy). Pair with --fast.")
    parser.add_argument("--l4-only", action="store_true", help="L4 principles only.")
    parser.add_argument("--resume", action="store_true", help="Skip already-completed books (from checkpoint).")
    parser.add_argument("--reset", action="store_true", help="Clear progress checkpoint and start fresh.")
    args = parser.parse_args()

    mode = "all"
    if args.fast:
        mode = "fast"
    elif args.l4_only:
        mode = "l4"
    elif args.llm_only:
        mode = "llm"

    os.makedirs(CHROMA_DIR, exist_ok=True)

    # Verify Ollama is running
    try:
        resp = httpx.get(f"{OLLAMA_URL}/api/version", timeout=3)
        print(f"Ollama is running (v{resp.json().get('version', '?')})")
    except Exception:
        print("ERROR: Ollama is not running. Start with: ollama serve")
        sys.exit(1)

    # Verify embedding model is available
    if mode in ("all", "fast"):
        try:
            test_resp = httpx.post(
                f"{OLLAMA_URL}/api/embed",
                json={"model": EMBED_MODEL, "input": "test"},
                timeout=30,
            )
            if test_resp.status_code != 200:
                print(f"ERROR: Embedding model '{EMBED_MODEL}' not available.")
                print(f"  Pull it with: ollama pull {EMBED_MODEL}")
                sys.exit(1)
            print(f"Embedding model '{EMBED_MODEL}' ready")
        except Exception as e:
            print(f"ERROR: Cannot reach embedding model: {e}")
            print(f"  Pull it with: ollama pull {EMBED_MODEL}")
            sys.exit(1)

    # Initialize ChromaDB
    client = chromadb.PersistentClient(path=CHROMA_DIR)
    collection = client.get_or_create_collection("shipai_books")
    print(f"ChromaDB ready at {CHROMA_DIR}")
    print(f"Current documents in index: {collection.count()}")

    # Load checkpoint
    progress = load_progress()
    if args.reset:
        progress = {"completed_l3": [], "completed_llm": [], "failed": []}
        save_progress(progress)
        print("Progress reset.")

    # Discover books
    all_books = get_all_pdfs(layer_filter=args.layer, book_filter=args.book)
    print(f"\nFound {len(all_books)} books to process (mode={mode})")

    total_done = 0
    total_skipped = 0
    total_failed = 0
    total_start = time.time()

    for pdf_path, layer_name in all_books:
        book_id = os.path.basename(pdf_path)
        progress_key = "completed_l3" if mode == "fast" else "completed_l4" if mode == "l4" else "completed_llm"

        if args.resume and book_id in progress.get(progress_key, []):
            print(f"  [SKIP] {book_id[:50]} -- already done")
            total_skipped += 1
            continue

        try:
            success = ingest_single_book(collection, pdf_path, layer_name, mode=mode)
            if success:
                progress.setdefault(progress_key, []).append(book_id)
                save_progress(progress)
                total_done += 1
                print(f"  [{total_done}/{len(all_books) - total_skipped}] Checkpoint saved")
            else:
                progress.setdefault("failed", []).append({"book": book_id, "error": "extraction_failed"})
                save_progress(progress)
                total_failed += 1
        except Exception as e:
            print(f"  [ERROR] {book_id}: {e}")
            progress.setdefault("failed", []).append({"book": book_id, "error": str(e)[:200]})
            save_progress(progress)
            total_failed += 1
            # Continue to next book -- don't crash the whole batch

    total_elapsed = time.time() - total_start
    print(f"\n{'='*60}")
    print(f"INGESTION COMPLETE")
    print(f"  Mode: {mode}")
    print(f"  Books processed: {total_done}")
    print(f"  Books skipped: {total_skipped}")
    print(f"  Books failed: {total_failed}")
    print(f"  Total documents in index: {collection.count()}")
    print(f"  Total time: {total_elapsed:.1f}s")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
