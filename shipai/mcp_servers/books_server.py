"""
ShipAI — 30-Book Knowledge Base MCP Server
===========================================
Multi-level retrieval from the ingested book index.
Supports: book summaries, chapter summaries, semantic chunks, principles.
Runs as a local MCP server via stdio transport.
"""
from fastmcp import FastMCP
import chromadb
import os

mcp = FastMCP("shipai-books")

# Path to the multi-level ChromaDB index
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
CHROMA_DIR = os.path.join(PROJECT_ROOT, "knowledge_base", "chroma_db")

if os.path.exists(CHROMA_DIR):
    client = chromadb.PersistentClient(path=CHROMA_DIR)
    collection = client.get_or_create_collection("shipai_books")
else:
    client = None
    collection = None


@mcp.tool()
async def query_knowledge(
    query: str, 
    layer: str = "all",      # data / ml / llm / ops / math / domain
    top_k: int = 5
) -> list[dict]:
    """Query the 30-book knowledge base with multi-level retrieval.
    Returns results from book summaries, principles, and chunks simultaneously."""
    if not collection:
        return [{"error": "Knowledge base not found. Run: python backend/scripts/ingest_books.py"}]
    
    results_out = []
    
    # Build where filter
    where_base = {"layer": layer} if layer != "all" else None
    
    # Query principles first (most actionable)
    where_principles = {"$and": [{"level": "principle"}, {"layer": layer}]} if layer != "all" else {"level": "principle"}
    try:
        principles = collection.query(
            query_texts=[query],
            n_results=min(top_k, 3),
            where=where_principles,
        )
        if principles["documents"] and principles["documents"][0]:
            for i in range(len(principles["documents"][0])):
                results_out.append({
                    "type": "principle",
                    "book": principles["metadatas"][0][i].get("title", "Unknown"),
                    "layer": principles["metadatas"][0][i].get("layer", "?"),
                    "content": principles["documents"][0][i],
                    "relevance": round(1 - principles["distances"][0][i], 3) if principles["distances"] else 0,
                })
    except Exception:
        pass
    
    # Then book-level summaries (broad context)
    where_books = {"$and": [{"level": "book"}, {"layer": layer}]} if layer != "all" else {"level": "book"}
    try:
        books = collection.query(
            query_texts=[query],
            n_results=min(top_k, 2),
            where=where_books,
        )
        if books["documents"] and books["documents"][0]:
            for i in range(len(books["documents"][0])):
                results_out.append({
                    "type": "book_summary",
                    "book": books["metadatas"][0][i].get("title", "Unknown"),
                    "layer": books["metadatas"][0][i].get("layer", "?"),
                    "content": books["documents"][0][i],
                    "relevance": round(1 - books["distances"][0][i], 3) if books["distances"] else 0,
                })
    except Exception:
        pass
    
    # Then detailed chunks (specifics)
    where_chunks = {"$and": [{"level": "chunk"}, {"layer": layer}]} if layer != "all" else {"level": "chunk"}
    try:
        chunks = collection.query(
            query_texts=[query],
            n_results=min(top_k, 3),
            where=where_chunks,
        )
        if chunks["documents"] and chunks["documents"][0]:
            for i in range(len(chunks["documents"][0])):
                results_out.append({
                    "type": "chunk",
                    "book": chunks["metadatas"][0][i].get("title", "Unknown"),
                    "layer": chunks["metadatas"][0][i].get("layer", "?"),
                    "content": chunks["documents"][0][i][:500],  # Truncate to save tokens
                    "relevance": round(1 - chunks["distances"][0][i], 3) if chunks["distances"] else 0,
                })
    except Exception:
        pass
    
    if not results_out:
        return [{"error": "No relevant insights found for this query."}]
    
    # Sort by relevance
    results_out.sort(key=lambda x: x.get("relevance", 0), reverse=True)
    return results_out[:top_k]


@mcp.tool()
async def get_book_principles(book_title: str) -> dict:
    """Get all indexed principles from a specific book"""
    if not collection:
        return {"error": "Knowledge base not found."}
    
    try:
        results = collection.get(
            where={"$and": [{"title": book_title}, {"level": "principle"}]},
            limit=20,
        )
    except Exception:
        # Fallback: try partial match via query
        results = collection.query(
            query_texts=[book_title],
            where={"level": "principle"},
            n_results=10,
        )
        if results["documents"] and results["documents"][0]:
            return {
                "book": book_title,
                "principles": results["documents"][0],
            }
        return {"error": f"Book '{book_title}' not found in index."}
    
    if not results["documents"]:
        return {"error": f"No principles found for '{book_title}'."}
    
    return {
        "book": book_title,
        "principles": results["documents"],
    }


@mcp.tool()
async def get_cross_book_synthesis(query: str, requirements: str = "") -> str:
    """Synthesize insights from MULTIPLE books simultaneously.
    This is the NotebookLM-grade capability."""
    if not collection:
        return "Knowledge base not found."
    
    # Fetch from all levels
    all_context = []
    
    for level in ["book", "principle", "chapter"]:
        try:
            results = collection.query(
                query_texts=[query],
                where={"level": level},
                n_results=3,
            )
            if results["documents"] and results["documents"][0]:
                for i, doc in enumerate(results["documents"][0]):
                    book = results["metadatas"][0][i].get("title", "Unknown")
                    all_context.append(f"[{level.upper()} from '{book}']: {doc[:400]}")
        except Exception:
            pass
    
    if not all_context:
        return "No knowledge found for this query."
    
    # Return the raw multi-source context for the LLM to synthesize
    return "\n\n---\n\n".join(all_context)


if __name__ == "__main__":
    mcp.run(transport="stdio")
