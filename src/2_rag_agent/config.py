"""Configuration utilities for the RAG agent with MCP integration."""
from __future__ import annotations

from dataclasses import dataclass
import os
from typing import Final

_DEFAULT_OPENSEARCH_URL: Final[str] = "http://127.0.0.1:9201"
_DEFAULT_LLM_BASE_URL: Final[str] = "http://127.0.0.1:8080/v1"
_DEFAULT_LLM_MODEL: Final[str] = "neural-chat-7b-v3-3"
_DEFAULT_MCP_URL: Final[str] = "http://127.0.0.1:8787/mcp"
_DEFAULT_INDEX: Final[str] = "bbc"
_DEFAULT_EMBEDDING_MODEL: Final[str] = "thenlper/gte-small"


@dataclass(frozen=True)
class AgentConfig:
    """Runtime configuration for the MCP-enabled RAG agent."""

    opensearch_url: str = _DEFAULT_OPENSEARCH_URL
    opensearch_index: str = _DEFAULT_INDEX
    llm_base_url: str = _DEFAULT_LLM_BASE_URL
    llm_model: str = _DEFAULT_LLM_MODEL
    mcp_http_url: str = _DEFAULT_MCP_URL
    top_k: int = 5
    embedding_model: str = _DEFAULT_EMBEDDING_MODEL
    llm_api_key: str = "sk-no-key"
    server_host: str = "0.0.0.0"
    server_port: int = 7000


def _int_from_env(name: str, default: int) -> int:
    value = os.getenv(name)
    if value is None or value == "":
        return default
    try:
        return int(value)
    except ValueError:
        return default


def load_config() -> AgentConfig:
    """Load configuration values from the environment with defaults."""

    return AgentConfig(
        opensearch_url=os.getenv("OPENSEARCH_URL", _DEFAULT_OPENSEARCH_URL),
        opensearch_index=os.getenv("OPENSEARCH_INDEX", _DEFAULT_INDEX),
        llm_base_url=os.getenv("LLM_BASE_URL", _DEFAULT_LLM_BASE_URL),
        llm_model=os.getenv("LLM_MODEL", _DEFAULT_LLM_MODEL),
        mcp_http_url=os.getenv("MCP_HTTP_URL", _DEFAULT_MCP_URL),
        top_k=_int_from_env("TOP_K", 5),
        embedding_model=os.getenv("EMBEDDING_MODEL", _DEFAULT_EMBEDDING_MODEL),
        llm_api_key=os.getenv("LLM_API_KEY", "sk-no-key"),
        server_host=os.getenv("SERVER_HOST", "0.0.0.0"),
        server_port=_int_from_env("SERVER_PORT", 7000),
    )


__all__ = ["AgentConfig", "load_config"]
