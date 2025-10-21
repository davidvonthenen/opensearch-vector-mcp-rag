"""Mock MCP server exposing a news.search tool for testing."""
from __future__ import annotations

import json
from datetime import datetime, timedelta
from typing import Any, Dict, List

from flask import Flask, jsonify, request

APP = Flask(__name__)


def _jsonrpc_response(result: Any, request_id: Any) -> Dict[str, Any]:
    return {"jsonrpc": "2.0", "id": request_id, "result": result}


def _jsonrpc_error(code: int, message: str, request_id: Any) -> Dict[str, Any]:
    return {"jsonrpc": "2.0", "id": request_id, "error": {"code": code, "message": message}}


def _build_articles(query: str, limit: int) -> List[Dict[str, Any]]:
    base_time = datetime.utcnow()
    articles = []
    for idx in range(limit):
        published = (base_time - timedelta(hours=idx)).isoformat() + "Z"
        articles.append(
            {
                "id": f"mock-{idx+1}",
                "title": f"Latest insight about {query} #{idx+1}",
                "url": f"https://example.com/{query.replace(' ', '-').lower()}/{idx+1}",
                "snippet": f"A short summary for {query} article {idx+1}.",
                "published_at": published,
            }
        )
    return articles


def _handle_initialize(payload: Dict[str, Any]) -> Dict[str, Any]:
    request_id = payload.get("id")
    result = {
        "capabilities": {"tools": {"listChanged": False}},
        "protocolVersion": "2025-06-18",
        "serverInfo": {"name": "mock-mcp", "version": "0.1.0"},
    }
    return _jsonrpc_response(result, request_id)


def _handle_tools_list(payload: Dict[str, Any]) -> Dict[str, Any]:
    request_id = payload.get("id")
    result = {
        "tools": [
            {
                "name": "news.search",
                "description": "Return mock latest news for a query.",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "query": {"type": "string", "description": "search string"},
                        "limit": {
                            "type": "integer",
                            "minimum": 1,
                            "maximum": 10,
                            "default": 5,
                        },
                    },
                    "required": ["query"],
                },
                "outputSchema": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "id": {"type": "string"},
                            "title": {"type": "string"},
                            "url": {"type": "string"},
                            "snippet": {"type": "string"},
                            "published_at": {"type": "string"},
                        },
                        "required": ["id", "title", "url", "snippet", "published_at"],
                    },
                },
            }
        ]
    }
    return _jsonrpc_response(result, request_id)


def _handle_tools_call(payload: Dict[str, Any]) -> Dict[str, Any]:
    request_id = payload.get("id")
    params = payload.get("params") or {}
    name = params.get("name")
    if name != "news.search":
        return _jsonrpc_error(-32601, f"Unknown tool: {name}", request_id)

    arguments = params.get("arguments") or {}
    query = str(arguments.get("query", "")).strip()
    if not query:
        return _jsonrpc_error(-32602, "Missing required argument: query", request_id)
    limit = arguments.get("limit", 5)
    try:
        limit = int(limit)
    except (TypeError, ValueError):
        return _jsonrpc_error(-32602, "limit must be an integer", request_id)
    limit = max(1, min(limit, 10))

    articles = _build_articles(query, limit)
    serialized = json.dumps(articles)
    result = {
        "content": [{"type": "text", "text": serialized}],
        "structuredContent": articles,
        "isError": False,
    }
    return _jsonrpc_response(result, request_id)


@APP.post("/mcp")
def handle_mcp() -> Any:
    payload = request.get_json(force=True, silent=False)
    if not isinstance(payload, dict):
        return jsonify(_jsonrpc_error(-32600, "Invalid JSON-RPC envelope", None)), 400

    method = payload.get("method")
    if method == "initialize":
        response = _handle_initialize(payload)
    elif method == "tools/list":
        response = _handle_tools_list(payload)
    elif method == "tools/call":
        response = _handle_tools_call(payload)
    else:
        response = _jsonrpc_error(-32601, f"Unknown method: {method}", payload.get("id"))
    return jsonify(response)


if __name__ == "__main__":
    APP.run(host="0.0.0.0", port=8787)
