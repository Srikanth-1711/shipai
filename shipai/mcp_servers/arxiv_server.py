from mcp.server.fastmcp import FastMCP
import arxiv
from typing import List, Dict, Any

mcp = FastMCP("shipai_arxiv")

@mcp.tool()
def search_papers(query: str, days_back: int = 30, max_results: int = 10) -> List[Dict[str, Any]]:
    """Search ArXiv for papers based on a query."""
    try:
        search = arxiv.Search(
            query=query,
            max_results=max_results,
            sort_by=arxiv.SortCriterion.SubmittedDate,
            sort_order=arxiv.SortOrder.Descending
        )
        
        client = arxiv.Client()
        results = []
        for paper in client.results(search):
            results.append({
                "id": paper.get_short_id(),
                "title": paper.title,
                "authors": [a.name for a in paper.authors],
                "published": str(paper.published),
                "summary": paper.summary,
                "url": paper.pdf_url
            })
        return results
    except Exception as e:
        return [{"error": str(e)}]

@mcp.tool()
def get_paper_details(arxiv_id: str) -> Dict[str, Any]:
    """Get detailed information about a specific ArXiv paper."""
    try:
        search = arxiv.Search(id_list=[arxiv_id])
        paper = next(search.results())
        return {
            "id": paper.get_short_id(),
            "title": paper.title,
            "authors": [a.name for a in paper.authors],
            "published": str(paper.published),
            "summary": paper.summary,
            "url": paper.pdf_url,
            "categories": paper.categories
        }
    except Exception as e:
        return {"error": str(e)}

@mcp.tool()
def get_trending_papers(category: str = "cs.AI", days_back: int = 7) -> List[Dict[str, Any]]:
    """Get trending papers in a specific category."""
    try:
        search = arxiv.Search(
            query=f"cat:{category}",
            max_results=10,
            sort_by=arxiv.SortCriterion.SubmittedDate,
            sort_order=arxiv.SortOrder.Descending
        )
        
        results = []
        for paper in search.results():
            results.append({
                "id": paper.get_short_id(),
                "title": paper.title,
                "authors": [a.name for a in paper.authors],
                "published": str(paper.published),
                "summary": paper.summary,
                "url": paper.pdf_url
            })
        return results
    except Exception as e:
        return [{"error": str(e)}]

if __name__ == "__main__":
    mcp.run()
