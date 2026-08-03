"""Web search tool (optional — off by default, requires SERPAPI_API_KEY)."""

from __future__ import annotations

import requests

import config

SPEC = {
    "name": "web_search",
    "description": (
        "Search the public web for current information. Only use this when the "
        "question needs information beyond your training data or personal context."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "query": {"type": "string", "description": "The search query."}
        },
        "required": ["query"],
    },
}


def run(args: dict) -> str:
    query = str(args.get("query", "")).strip()
    if not query:
        return "Error: no query provided."

    if not config.WEB_SEARCH_ENABLED or not config.WEB_SEARCH_API_KEY:
        return (
            "Web search is disabled. Set PRATIK_WEB_SEARCH_ENABLED=true and "
            "SERPAPI_API_KEY in your .env to enable it (this keeps the assistant "
            "offline-first by default)."
        )

    try:
        response = requests.get(
            "https://serpapi.com/search",
            params={"q": query, "api_key": config.WEB_SEARCH_API_KEY},
            timeout=15,
        )
        response.raise_for_status()
        data = response.json()
        results = data.get("organic_results", [])[:3]
        if not results:
            return "No results found."
        lines = [f"- {r.get('title')}: {r.get('snippet', '')}" for r in results]
        return "\n".join(lines)
    except Exception as exc:  # noqa: BLE001
        return f"Error: web search failed ({exc})."
