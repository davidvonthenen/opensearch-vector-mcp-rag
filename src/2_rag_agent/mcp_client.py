"""HTTP client for interacting with a Model Context Protocol server."""
from __future__ import annotations

import itertools
import json
from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple

import requests


class MCPClientError(RuntimeError):
    """Raised when the MCP server returns an error or malformed payload."""


@dataclass
class MCPCallResult:
    """Container for MCP tool call results."""

    content: Any
    structured_content: Any
    is_error: bool = False


class MCPHttpClient:
    """Minimal JSON-RPC client for MCP servers over HTTP."""

    def __init__(self, base_url: str) -> None:
        self.base_url = base_url.rstrip("/")
        self._counter = itertools.count(1)
        self._session_id: Optional[str] = None

    def _headers(self) -> Dict[str, str]:
        headers = {"Content-Type": "application/json"}
        if self._session_id:
            headers["Mcp-Session-Id"] = self._session_id
        return headers

    def _post(self, method: str, params: Dict[str, Any]) -> Tuple[Dict[str, Any], requests.Response]:
        payload = {
            "jsonrpc": "2.0",
            "id": next(self._counter),
            "method": method,
            "params": params,
        }
        response = requests.post(self.base_url, headers=self._headers(), json=payload, timeout=10)
        if response.status_code != 200:
            raise MCPClientError(f"MCP request failed with HTTP {response.status_code}: {response.text}")
        try:
            data = response.json()
        except json.JSONDecodeError as exc:
            raise MCPClientError("MCP server returned invalid JSON") from exc
        if "error" in data:
            raise MCPClientError(f"MCP error: {data['error']}")
        return data, response

    def initialize(self) -> Dict[str, Any]:
        """Perform the MCP initialize handshake and return server capabilities."""

        data, response = self._post("initialize", {"protocolVersion": "2025-06-18"})
        result = data.get("result")
        if not isinstance(result, dict):
            raise MCPClientError("Invalid MCP initialize response")
        session_id = response.headers.get("Mcp-Session-Id")
        if session_id:
            self._session_id = session_id
        return result

    def tools_list(self) -> Dict[str, Any]:
        """Return the available tool definitions from the MCP server."""

        data, _ = self._post("tools/list", {})
        result = data.get("result")
        if not isinstance(result, dict):
            raise MCPClientError("Invalid MCP tools/list response")
        return result

    def tools_call(self, name: str, arguments: Dict[str, Any]) -> MCPCallResult:
        """Invoke a tool by name with the provided arguments."""

        data, _ = self._post("tools/call", {"name": name, "arguments": arguments})
        result = data.get("result")
        if not isinstance(result, dict):
            raise MCPClientError("Invalid MCP tools/call response")

        content = result.get("content")
        structured = result.get("structuredContent")
        is_error = bool(result.get("isError", False))
        return MCPCallResult(content=content, structured_content=structured, is_error=is_error)

    def search_news(self, query: str, limit: int = 5) -> Any:
        """Convenience wrapper for the mock news.search tool."""

        arguments = {"query": query, "limit": limit}
        call_result = self.tools_call("news.search", arguments)
        if call_result.is_error:
            raise MCPClientError("news.search returned an error")
        return call_result.structured_content


__all__ = ["MCPHttpClient", "MCPClientError", "MCPCallResult"]
