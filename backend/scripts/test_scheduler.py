import sys
import os
import asyncio

# Add backend to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from app.engine.intelligence_scheduler import GitHubSourceHandler, ArxivSourceHandler, DocsSourceHandler, BlogSourceHandler, get_live_collection

async def run_scheduler():
    print("=== Live Intelligence Scheduler Manual Run ===\n")
    
    print("[1/4] Running GitHubSourceHandler...")
    github_handler = GitHubSourceHandler()
    github_items = await github_handler.fetch()
    github_handler.process_and_store(github_items)
    
    print("\n[2/4] Running ArxivSourceHandler...")
    arxiv_handler = ArxivSourceHandler()
    arxiv_items = arxiv_handler.fetch()
    arxiv_handler.process_and_store(arxiv_items)
    
    print("\n[3/4] Running DocsSourceHandler...")
    docs_handler = DocsSourceHandler()
    docs_items = await docs_handler.fetch()
    docs_handler.process_and_store(docs_items)
    
    print("\n[4/4] Running BlogSourceHandler...")
    blog_handler = BlogSourceHandler()
    blog_items = blog_handler.fetch()
    blog_handler.process_and_store(blog_items)
    
    print("\n=== Scheduler Run Complete ===")
    
    col = get_live_collection()
    data = col.get()
    
    total = len(data["ids"]) if data else 0
    print(f"\nTotal documents ingested: {total}")
    
    if total > 0:
        sources = {}
        highest_score = 0
        highest_item = None
        for meta in data["metadatas"]:
            src = meta.get("source", "unknown")
            sources[src] = sources.get(src, 0) + 1
            
            score = meta.get("score", 0)
            if score > highest_score:
                highest_score = score
                highest_item = meta
                
        print("Count per source type:")
        for src, count in sources.items():
            print(f"  - {src}: {count}")
            
        print("\nHighest quality scored item:")
        print(f"  Source: {highest_item.get('source')}")
        if 'repo' in highest_item: print(f"  Repo: {highest_item.get('repo')}")
        if 'paper_id' in highest_item: print(f"  Paper ID: {highest_item.get('paper_id')}")
        print(f"  Score: {highest_item.get('score'):.2f}")

if __name__ == "__main__":
    asyncio.run(run_scheduler())
