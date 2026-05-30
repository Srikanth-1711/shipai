import os
import datetime
import httpx
import feedparser
import arxiv
import chromadb
from typing import List, Dict, Any
from dateutil import parser as date_parser

# Base configurations
CHROMA_DIR = os.path.join(os.path.dirname(__file__), "..", "..", "..", "knowledge_base", "chroma_db")

def get_live_collection():
    client = chromadb.PersistentClient(path=CHROMA_DIR)
    return client.get_or_create_collection("shipai_live")

def score_quality(content: str, source_type: str, metadata: dict = None) -> float:
    """Scores content quality to ensure only high-signal data is indexed."""
    metadata = metadata or {}
    
    # Base scores
    base_scores = {
        "official_docs": 0.95,
        "arxiv_paper": 0.85,
        "github_readme_1k_stars": 0.80,
        "known_author_blog": 0.75,
        "github_readme_500stars": 0.65,
        "unknown": 0.30
    }
    
    # Determine base score
    score = base_scores.get(source_type, base_scores["unknown"])
    if source_type == "github":
        stars = metadata.get("stars", 0)
        if stars >= 1000:
            score = base_scores["github_readme_1k_stars"]
        elif stars >= 500:
            score = base_scores["github_readme_500stars"]
            
    # Bonus signals
    content_lower = content.lower()
    if "```" in content_lower or "def " in content_lower or "class " in content_lower:
        score += 0.10  # has_code_examples
        
    if "benchmark" in content_lower or "evaluat" in content_lower or "accuracy" in content_lower or "latency" in content_lower:
        score += 0.10  # has_benchmarks
        
    # Recency bonus
    published_date = metadata.get("published_date")
    if published_date:
        if isinstance(published_date, str):
            try:
                published_date = date_parser.parse(published_date)
            except Exception:
                published_date = None
        
        if published_date:
            # Check if aware timezone
            if published_date.tzinfo is None:
                now = datetime.datetime.now()
            else:
                now = datetime.datetime.now(datetime.timezone.utc)
                
            delta = now - published_date
            if delta.days < 7 and source_type == "arxiv_paper":
                score += 0.15
            elif delta.days < 180: # published < 6 months
                score += 0.10

    # Contradiction flag (simplified for now, would involve checking ChromaDB)
    # If it contradicts, we'd flag it, but for scoring purposes we cap it.
    
    return min(score, 1.0)


class GitHubSourceHandler:
    """Monitors key AI repos for architectural patterns and breaking changes."""
    def __init__(self):
        self.repos = [
            "langchain-ai/langgraph", "run-llama/llama_index", "deepset-ai/haystack", 
            "chroma-core/chroma", "microsoft/presidio", "explodinggradients/ragas", 
            "confident-ai/deepeval", "vllm-project/vllm", "unslothai/unsloth", 
            "julep-ai/fastmcp"
        ]
        
    async def fetch(self) -> List[Dict[str, Any]]:
        # This would ideally use the existing github_server via MCP, 
        # but for scheduling we can hit the GitHub API directly or import the logic.
        results = []
        async with httpx.AsyncClient() as client:
            for repo in self.repos:
                try:
                    resp = await client.get(f"https://api.github.com/repos/{repo}")
                    if resp.status_code == 200:
                        data = resp.json()
                        if data.get("stargazers_count", 0) > 500:
                            # Fetch README
                            readme_resp = await client.get(f"https://api.github.com/repos/{repo}/readme")
                            if readme_resp.status_code == 200:
                                readme_data = readme_resp.json()
                                import base64
                                content = base64.b64decode(readme_data.get("content", "")).decode("utf-8")
                                results.append({
                                    "source": "github",
                                    "repo": repo,
                                    "stars": data.get("stargazers_count"),
                                    "last_commit": data.get("updated_at"),
                                    "layer": "llm",
                                    "content": content
                                })
                except Exception as e:
                    print(f"Failed to fetch repo {repo}: {e}")
        return results

    def process_and_store(self, items: List[Dict[str, Any]]):
        col = get_live_collection()
        for item in items:
            score = score_quality(item["content"], "github", {"stars": item["stars"], "published_date": item["last_commit"]})
            if score >= 0.65:
                doc_id = f"github_{item['repo'].replace('/', '_')}"
                metadata = {
                    "source": "github", 
                    "repo": item["repo"], 
                    "stars": item["stars"], 
                    "last_commit": item["last_commit"], 
                    "layer": item["layer"],
                    "score": score
                }
                # Chunking and embedding would ideally happen here. For now, store raw.
                col.upsert(ids=[doc_id], documents=[item["content"][:2000]], metadatas=[metadata])
                print(f"Indexed GitHub: {item['repo']} (Score: {score:.2f})")

class ArxivSourceHandler:
    """Monitors arXiv for new AI engineering and research papers."""
    def __init__(self):
        self.categories = ["cs.AI", "cs.LG", "cs.CL", "cs.IR"]
        self.keywords = ["RAG", "retrieval augmented generation", "LLM agents", "fine-tuning", "vector search", "multi-agent", "MCP", "LLM evaluation", "hallucination", "chain of thought"]
        
    def fetch(self) -> List[Dict[str, Any]]:
        results = []
        query = " OR ".join([f'cat:{cat}' for cat in self.categories])
        # Also filter by keywords in title/abstract if possible with arxiv API
        search = arxiv.Search(
            query=query,
            max_results=50,
            sort_by=arxiv.SortCriterion.SubmittedDate,
            sort_order=arxiv.SortOrder.Descending
        )
        
        client = arxiv.Client()
        for paper in client.results(search):
            title_abstract = f"{paper.title} {paper.summary}".lower()
            if any(kw.lower() in title_abstract for kw in self.keywords):
                results.append({
                    "source": "arxiv",
                    "paper_id": paper.get_short_id(),
                    "title": paper.title,
                    "authors": [a.name for a in paper.authors],
                    "published": str(paper.published),
                    "content": f"Title: {paper.title}\nAbstract: {paper.summary}",
                    "layer": "llm"
                })
        return results

    def process_and_store(self, items: List[Dict[str, Any]]):
        col = get_live_collection()
        for item in items:
            score = score_quality(item["content"], "arxiv_paper", {"published_date": item["published"]})
            if score >= 0.65:
                doc_id = f"arxiv_{item['paper_id']}"
                metadata = {
                    "source": "arxiv",
                    "paper_id": item["paper_id"],
                    "authors": ", ".join(item["authors"]),
                    "published": item["published"],
                    "layer": item["layer"],
                    "score": score
                }
                col.upsert(ids=[doc_id], documents=[item["content"]], metadatas=[metadata])
                print(f"Indexed Arxiv: {item['title']} (Score: {score:.2f})")

class DocsSourceHandler:
    """Crawls official documentation for high-fidelity technical facts."""
    def __init__(self):
        self.sources = [
            "https://docs.langchain.com",
            "https://langchain-ai.github.io/langgraph",
            "https://docs.anthropic.com",
            "https://docs.chroma.tech",
            "https://platform.openai.com/docs",
            "https://docs.llamaindex.ai"
        ]
        
    async def fetch(self) -> List[Dict[str, Any]]:
        # Simplified placeholder for a real documentation crawler
        # In a real app, you'd use something like Scrapy or Playwright to crawl docs.
        return []
        
    def process_and_store(self, items: List[Dict[str, Any]]):
        col = get_live_collection()
        for item in items:
            score = score_quality(item["content"], "official_docs")
            if score >= 0.65:
                doc_id = f"docs_{hash(item['url'])}"
                # Change detection: store page hash, only re-index if changed (omitted for brevity)
                col.upsert(ids=[doc_id], documents=[item["content"]], metadatas=[{"source": "official_docs", "url": item["url"], "score": score}])
                print(f"Indexed Docs: {item['url']} (Score: {score:.2f})")

class BlogSourceHandler:
    """Pulls curated high-signal RSS feeds."""
    def __init__(self):
        self.feeds = [
            "https://huyenchip.com/feed.xml",
            "https://eugeneyan.com/rss/",
            "https://blog.langchain.dev/rss/"
        ]
        
    def fetch(self) -> List[Dict[str, Any]]:
        results = []
        for url in self.feeds:
            feed = feedparser.parse(url)
            for entry in feed.entries[:5]: # Top 5 recent
                results.append({
                    "source": "blog",
                    "url": entry.link,
                    "title": entry.title,
                    "published": entry.get("published", ""),
                    "content": entry.get("description", entry.get("summary", ""))
                })
        return results

    def process_and_store(self, items: List[Dict[str, Any]]):
        col = get_live_collection()
        for item in items:
            score = score_quality(item["content"], "known_author_blog", {"published_date": item["published"]})
            if score >= 0.65:
                doc_id = f"blog_{hash(item['url'])}"
                col.upsert(ids=[doc_id], documents=[item["content"]], metadatas=[{"source": "blog", "url": item["url"], "title": item["title"], "score": score}])
                print(f"Indexed Blog: {item['title']} (Score: {score:.2f})")

# Celery Beat tasks would look like this:
"""
from celery import Celery
from celery.schedules import crontab

app = Celery('intelligence_scheduler', broker='redis://localhost:6379/0')

@app.on_after_configure.connect
def setup_periodic_tasks(sender, **kwargs):
    # GitHub + Arxiv: every 24 hours at 2am
    sender.add_periodic_task(crontab(hour=2, minute=0), run_daily_sync.s(), name='Daily Sync')
    # Official docs: every 7 days, Sunday 3am
    sender.add_periodic_task(crontab(hour=3, minute=0, day_of_week=0), run_docs_sync.s(), name='Weekly Docs Sync')
    # Blogs: every 7 days, Sunday 4am
    sender.add_periodic_task(crontab(hour=4, minute=0, day_of_week=0), run_blogs_sync.s(), name='Weekly Blogs Sync')
"""
