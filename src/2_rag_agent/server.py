"""Flask server bridging OpenSearch RAG with an MCP-driven llama.cpp backend."""
from __future__ import annotations

import json
import time
import uuid
from threading import Lock
from typing import Any, Dict, List, Sequence

from flask import Flask, jsonify, request
from openai import OpenAI
from opensearchpy.exceptions import OpenSearchException

from ..common.logging import get_logger
from .config import AgentConfig, load_config
from .mcp_client import MCPClientError, MCPHttpClient
from .opensearch_store import OpenSearchStore, RetrievedDocument

LOGGER = get_logger(__name__)
APP = Flask(__name__)
CONFIG: AgentConfig = load_config()
STORE = OpenSearchStore(CONFIG)
LLM_CLIENT = OpenAI(base_url=CONFIG.llm_base_url, api_key=CONFIG.llm_api_key)
MCP_CLIENT = MCPHttpClient(CONFIG.mcp_http_url)
_MCP_INITIALIZED = False
_MCP_LOCK = Lock()

_NEWS_TOOL_DEFINITION = {
    "type": "function",
    "function": {
        "name": "news.search",
        "description": "Return mock latest news for a query.",
        "parameters": {
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
    },
}

_INITIAL_SYSTEM_PROMPT = (
    "You are a news-focused RAG agent. Always start by calling the news.search tool "
    "with the user query verbatim before forming an answer."
)
_FINAL_SYSTEM_PROMPT = (
    "You are a BBC news analyst. Synthesize an answer using the MCP news results and "
    "the OpenSearch context. Cite news items as [news:<id>] and OpenSearch documents as "
    "[doc:<number>]. If information is missing, respond that you do not know."
)


def _ensure_mcp_initialized() -> None:
    global _MCP_INITIALIZED
    if _MCP_INITIALIZED:
        return
    with _MCP_LOCK:
        if _MCP_INITIALIZED:
            return
        try:
            MCP_CLIENT.initialize()
            MCP_CLIENT.tools_list()
            _MCP_INITIALIZED = True
        except MCPClientError as exc:
            LOGGER.warning("Unable to initialize MCP server: %s", exc)


def _extract_user_question(messages: Sequence[Dict[str, Any]]) -> str:
    for message in reversed(messages):
        if message.get("role") == "user":
            content = message.get("content", "")
            if isinstance(content, str) and content.strip():
                return content.strip()
    raise ValueError("No user message found in request")


def _serialize_documents(documents: Sequence[RetrievedDocument]) -> List[Dict[str, Any]]:
    return [doc.to_summary() for doc in documents]


def _assistant_message_dict(choice_message: Any) -> Dict[str, Any]:
    tool_calls = []
    for call in getattr(choice_message, "tool_calls", []) or []:
        tool_calls.append(
            {
                "id": call.id,
                "type": call.type,
                "function": {
                    "name": call.function.name,
                    "arguments": call.function.arguments,
                },
            }
        )
    return {
        "role": choice_message.role,
        "content": choice_message.content or "",
        "tool_calls": tool_calls,
    }


def _sum_usage(*usages: Any) -> Dict[str, int]:
    totals = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
    for usage in usages:
        if usage is None:
            continue
        totals["prompt_tokens"] += getattr(usage, "prompt_tokens", 0) or 0
        totals["completion_tokens"] += getattr(usage, "completion_tokens", 0) or 0
        totals["total_tokens"] += getattr(usage, "total_tokens", 0) or 0
    return totals


@APP.post("/v1/chat/completions")
def chat_completions() -> Any:
    payload = request.get_json(force=True, silent=False)
    if not isinstance(payload, dict):
        return jsonify({"error": "Invalid JSON payload"}), 400

    messages = payload.get("messages")
    if not isinstance(messages, list):
        return jsonify({"error": "messages must be a list"}), 400

    try:
        question = _extract_user_question(messages)
    except ValueError as exc:
        return jsonify({"error": str(exc)}), 400

    try:
        context_block, documents = STORE.retrieve(question, CONFIG.top_k)
    except OpenSearchException as exc:
        LOGGER.exception("OpenSearch query failed: %s", exc)
        return jsonify({"error": "OpenSearch query failed", "details": str(exc)}), 502
    except Exception as exc:  # noqa: BLE001
        LOGGER.exception("Unexpected error retrieving OpenSearch context")
        return jsonify({"error": "OpenSearch retrieval failed", "details": str(exc)}), 502

    _ensure_mcp_initialized()

    initial_messages = [
        {"role": "system", "content": _INITIAL_SYSTEM_PROMPT},
        {"role": "user", "content": question},
    ]

    try:
        initial_completion = LLM_CLIENT.chat.completions.create(
            model=CONFIG.llm_model,
            messages=initial_messages,
            tools=[_NEWS_TOOL_DEFINITION],
            tool_choice={"type": "function", "function": {"name": "news.search"}},
            temperature=float(payload.get("temperature", 0.2)),
        )
    except Exception as exc:  # noqa: BLE001
        LOGGER.exception("Initial LLM call failed")
        return jsonify({"error": "LLM tool planning failed", "details": str(exc)}), 502

    initial_choice = initial_completion.choices[0]
    assistant_message = _assistant_message_dict(initial_choice.message)
    tool_calls = assistant_message.get("tool_calls", [])
    if not tool_calls:
        assistant_message.pop("tool_calls", None)

    news_results: List[Dict[str, Any]] | None = None
    tool_error: str | None = None
    tool_messages: List[Dict[str, Any]] = []

    if not tool_calls:
        tool_error = "LLM did not return the required news.search tool call."
    else:
        for call in tool_calls:
            if call["function"]["name"] != "news.search":
                continue
            try:
                arguments = json.loads(call["function"]["arguments"] or "{}")
            except json.JSONDecodeError as exc:
                tool_error = f"Invalid tool arguments from LLM: {exc}"  # noqa: TRY400
                break
            try:
                call_result = MCP_CLIENT.tools_call("news.search", arguments)
                if call_result.is_error:
                    tool_error = "MCP reported an error during news.search."
                else:
                    structured = call_result.structured_content
                    if structured is None:
                        news_results = None
                    else:
                        news_results = structured
                        pretty_json = json.dumps(news_results, indent=2)
                        tool_messages.append(
                            {
                                "role": "tool",
                                "tool_call_id": call["id"],
                                "name": "news.search",
                                "content": pretty_json,
                            }
                        )
                break
            except MCPClientError as exc:
                tool_error = f"Failed to execute news.search: {exc}"
                break
        else:
            tool_error = "news.search tool call missing in LLM response."

    if news_results is None and tool_error is None:
        tool_error = "news.search did not return structured results."

    final_messages: List[Dict[str, Any]] = [
        {"role": "system", "content": _FINAL_SYSTEM_PROMPT},
        assistant_message,
        *tool_messages,
    ]

    if tool_error:
        final_messages.append(
            {
                "role": "system",
                "content": f"MCP tool execution issue: {tool_error}",
            }
        )
    final_messages.append(
        {
            "role": "system",
            "content": "OpenSearch Context:\n" + (context_block or "No relevant context retrieved."),
        }
    )
    final_messages.append({"role": "user", "content": question})

    try:
        final_completion = LLM_CLIENT.chat.completions.create(
            model=CONFIG.llm_model,
            messages=final_messages,
            temperature=float(payload.get("temperature", 0.2)),
            top_p=float(payload.get("top_p", 0.9)),
            max_tokens=int(payload.get("max_tokens", 512)),
        )
    except Exception as exc:  # noqa: BLE001
        LOGGER.exception("Final LLM call failed")
        return jsonify({"error": "LLM response generation failed", "details": str(exc)}), 502

    final_choice = final_completion.choices[0]
    final_message = {
        "role": final_choice.message.role,
        "content": final_choice.message.content or "",
    }

    usage = _sum_usage(initial_completion.usage, final_completion.usage)

    response_body = {
        "id": final_completion.id or str(uuid.uuid4()),
        "object": "chat.completion",
        "created": int(time.time()),
        "model": payload.get("model", CONFIG.llm_model),
        "choices": [
            {
                "index": 0,
                "message": final_message,
                "finish_reason": final_choice.finish_reason,
            }
        ],
        "usage": usage,
        "rag_context": {
            "index": CONFIG.opensearch_index,
            "k": CONFIG.top_k,
            "documents": _serialize_documents(documents),
            "hits": _serialize_documents(documents),
        },
        "news_search": {
            "results": news_results or [],
            "error": tool_error,
        },
    }
    return jsonify(response_body)


@APP.get("/__healthz")
def healthz() -> Any:
    return jsonify({"status": "ok"})


def main() -> None:
    LOGGER.info("Starting MCP-enabled RAG agent on %s:%s", CONFIG.server_host, CONFIG.server_port)
    APP.run(host=CONFIG.server_host, port=CONFIG.server_port)


if __name__ == "__main__":
    main()
