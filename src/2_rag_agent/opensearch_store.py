"""Utilities for retrieving contextual snippets from OpenSearch."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Sequence

from opensearchpy import OpenSearch

from ..common.embeddings import EmbeddingModel, to_list
from .config import AgentConfig


@dataclass
class RetrievedDocument:
    """Metadata captured for each retrieved OpenSearch document."""

    path: str
    title: str
    category: str
    score: float
    text: str

    def to_summary(self) -> Dict[str, Any]:
        """Return a serialisable summary for the API response."""

        return {
            "path": self.path,
            "title": self.title,
            "category": self.category,
            "score": self.score,
        }


class OpenSearchStore:
    """Helper responsible for embedding queries and fetching OpenSearch context."""

    def __init__(self, config: AgentConfig) -> None:
        self.config = config
        self._client = OpenSearch(
            hosts=[config.opensearch_url],
            http_compress=True,
            use_ssl=False,
            verify_certs=False,
        )
        # Reuse the shared embedding stack by emulating the required settings structure.
        embedding_settings = type("_EmbeddingSettings", (), {"embedding_model": config.embedding_model})
        self._embedder = EmbeddingModel(embedding_settings())

    def retrieve(self, query: str, top_k: int | None = None) -> tuple[str, List[RetrievedDocument]]:
        """Return a formatted context block and the retrieved documents."""

        k = top_k if top_k is not None else self.config.top_k
        embedding = self._embedder.encode([query])[0]
        body = {
            "size": k,
            "query": {"knn": {"embedding": {"vector": to_list(embedding), "k": k}}},
            "_source": ["path", "title", "category", "text"],
        }
        response = self._client.search(index=self.config.opensearch_index, body=body)
        hits = response.get("hits", {}).get("hits", [])
        documents: List[RetrievedDocument] = []
        for hit in hits:
            source = hit.get("_source", {})
            documents.append(
                RetrievedDocument(
                    path=str(source.get("path", "")),
                    title=str(source.get("title", "")),
                    category=str(source.get("category", "")),
                    score=float(hit.get("_score", 0.0)),
                    text=str(source.get("text", "")),
                )
            )
        context = build_context_block(documents)
        return context, documents


def build_context_block(documents: Sequence[RetrievedDocument], *, max_snippet_chars: int = 900) -> str:
    """Produce a newline separated context block for the LLM."""

    if not documents:
        return ""
    parts: List[str] = []
    for idx, doc in enumerate(documents, start=1):
        snippet = trim_snippet(doc.text, max_snippet_chars)
        parts.append(
            "[DOC {idx} | source: {category}/{title} | path: {path}]\n{snippet}".format(
                idx=idx,
                category=doc.category or "unknown",
                title=doc.title or "unknown",
                path=doc.path,
                snippet=snippet,
            )
        )
    return "\n\n".join(parts)


def trim_snippet(text: str, max_length: int) -> str:
    """Trim snippets to avoid exceeding the LLM context window."""

    if len(text) <= max_length:
        return text
    trimmed = text[:max_length]
    last_space = trimmed.rfind(" ")
    if last_space == -1:
        return trimmed + "..."
    return trimmed[:last_space] + "..."


__all__ = ["OpenSearchStore", "RetrievedDocument", "build_context_block", "trim_snippet"]
