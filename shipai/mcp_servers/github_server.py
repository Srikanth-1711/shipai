from fastmcp import FastMCP
import httpx
import os
import json
import base64

mcp = FastMCP("shipai-github")

def get_github_token() -> str:
    """Read token from ~/.shipai/config.json"""
    config_file = os.path.expanduser("~/.shipai/config.json")
    if os.path.exists(config_file):
        try:
            with open(config_file, "r") as f:
                return json.load(f).get("github_token", "")
        except:
            pass
    return ""

def get_headers() -> dict:
    token = get_github_token()
    headers = {"Accept": "application/vnd.github.v3+json"}
    if token:
        headers["Authorization"] = f"token {token}"
    return headers

@mcp.tool()
async def search_repos(query: str, min_stars: int = 1000) -> list[dict]:
    """Search GitHub for high-starred repos matching a pattern"""
    url = f"https://api.github.com/search/repositories?q={query}+stars:>={min_stars}&sort=stars&order=desc"
    async with httpx.AsyncClient() as client:
        resp = await client.get(url, headers=get_headers(), timeout=10)
        if resp.status_code != 200:
            return [{"error": f"GitHub API error: {resp.status_code} - {resp.text}"}]
        
        items = resp.json().get("items", [])[:5]
        return [
            {
                "name": item.get("full_name"),
                "description": item.get("description"),
                "url": item.get("html_url"),
                "stars": item.get("stargazers_count")
            }
            for item in items
        ]

@mcp.tool()
async def get_repo_readme(owner: str, repo: str) -> str:
    """Fetch and return the README of a specific repo"""
    url = f"https://api.github.com/repos/{owner}/{repo}/readme"
    async with httpx.AsyncClient() as client:
        resp = await client.get(url, headers=get_headers(), timeout=10)
        if resp.status_code != 200:
            return f"Error: {resp.status_code} - {resp.text}"
            
        data = resp.json()
        content = data.get("content", "")
        if content:
            return base64.b64decode(content).decode("utf-8", errors="ignore")
        return "README is empty or unreadable."

@mcp.tool()
async def get_recent_commits(owner: str, repo: str, days: int = 30) -> list[dict]:
    """Get recent commits — surface new patterns and breaking changes"""
    from datetime import datetime, timedelta
    since = (datetime.utcnow() - timedelta(days=days)).isoformat() + "Z"
    url = f"https://api.github.com/repos/{owner}/{repo}/commits?since={since}"
    
    async with httpx.AsyncClient() as client:
        resp = await client.get(url, headers=get_headers(), timeout=10)
        if resp.status_code != 200:
            return [{"error": f"Error: {resp.status_code}"}]
            
        commits = resp.json()[:10]  # Return top 10 recent
        return [
            {
                "sha": c.get("sha")[:7],
                "message": c.get("commit", {}).get("message", "").split("\n")[0],
                "author": c.get("commit", {}).get("author", {}).get("name"),
                "date": c.get("commit", {}).get("author", {}).get("date")
            }
            for c in commits
        ]

@mcp.tool()
async def get_repo_structure(owner: str, repo: str) -> dict:
    """Get file tree — understand architecture from folder layout"""
    # Fetch default branch
    url_repo = f"https://api.github.com/repos/{owner}/{repo}"
    async with httpx.AsyncClient() as client:
        resp_repo = await client.get(url_repo, headers=get_headers(), timeout=5)
        if resp_repo.status_code != 200:
            return {"error": f"Cannot find repo {owner}/{repo}"}
            
        default_branch = resp_repo.json().get("default_branch", "main")
        
        # Fetch tree
        url_tree = f"https://api.github.com/repos/{owner}/{repo}/git/trees/{default_branch}?recursive=1"
        resp_tree = await client.get(url_tree, headers=get_headers(), timeout=10)
        if resp_tree.status_code != 200:
            return {"error": f"Cannot fetch tree: {resp_tree.status_code}"}
            
        tree = resp_tree.json().get("tree", [])
        
        # We'll just return a simplified directory structure, ignoring deeply nested files to save tokens
        directories = [t["path"] for t in tree if t["type"] == "tree"]
        top_level_files = [t["path"] for t in tree if t["type"] == "blob" and "/" not in t["path"]]
        
        return {
            "directories": directories[:50],  # Limit to top 50
            "top_level_files": top_level_files
        }

if __name__ == "__main__":
    mcp.run(transport="stdio")
