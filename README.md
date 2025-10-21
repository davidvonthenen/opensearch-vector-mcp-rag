# OpenSearch Vector RAG Agent with MCP-integrated llama.cpp

This project provides an end-to-end Retrieval Augmented Generation (RAG) workflow
for the BBC news dataset. OpenSearch stores sentence-transformer embeddings, a
Flask service exposes an OpenAI-compatible `/v1/chat/completions` endpoint, and
all tool calls are brokered through the Model Context Protocol (MCP). A mock MCP
server returns synthetic breaking news so that every query exercises tool calling
against a llama.cpp OpenAI-compatible server running in
`chatml-function-calling` mode.

## Prerequisites

- Python 3.11
- OpenSearch 3.2 running locally with security disabled (see Docker snippet
  below)
- BBC dataset available in `./bbc`
- llama.cpp installed with the `llama_cpp.server` entrypoint available on your
  PATH
- Quantized GGUF model (default `TheBloke/neural-chat-7B-v3-3.Q4_K_M.gguf`)

### Local OpenSearch via Docker

```bash
docker network create opensearch-net

docker run -d \
  --name opensearch-longterm \
  --network opensearch-net \
  -p 9201:9200 -p 9601:9600 \
  -e "discovery.type=single-node" \
  -e "DISABLE_SECURITY_PLUGIN=true" \
  -v "$HOME/opensearch-longterm/data:/usr/share/opensearch/data" \
  -v "$HOME/opensearch-longterm/snapshots:/mnt/snapshots" \
  opensearchproject/opensearch:3.2.0
```

(Optional) Dashboards:

```bash
docker run -d \
  --name opensearch-longterm-dashboards \
  --network opensearch-net \
  -p 5601:5601 \
  -e 'OPENSEARCH_HOSTS=["http://opensearch-longterm:9200"]' \
  -e 'DISABLE_SECURITY_DASHBOARDS_PLUGIN=true' \
  opensearchproject/opensearch-dashboards:3.2.0
```

## Installation

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Run the workflow

1. **Ingest** the BBC dataset into OpenSearch (vector index `bbc` by default):
   ```bash
   make ingest
   ```
2. **Start the llama.cpp OpenAI-compatible server** in tool-calling chat mode:
   ```bash
   make llm-server MODEL_PATH=/absolute/path/to/neural-chat-7b-v3-3.Q4_K_M.gguf
   ```
3. **Launch the mock MCP server** (JSON-RPC over HTTP):
   ```bash
   make mcp-server
   ```
4. **Run the MCP-enabled RAG agent** (Flask on port 7000 by default):
   ```bash
   make rag-agent
   ```
5. **Query the agent** using the bundled client:
   ```bash
   make query QUESTION="What happened with the Bank of England and inflation?"
   ```

The client still accepts the original `--question` flag; the Makefile forwards a
`--q` alias to keep the legacy interface working.

## Environment configuration

The agent reads configuration from environment variables (see
`src/2_rag_agent/config.py` for defaults):

- `OPENSEARCH_URL` (default `http://127.0.0.1:9201`)
- `OPENSEARCH_INDEX` (default `bbc`)
- `LLM_BASE_URL` (default `http://127.0.0.1:8080/v1`)
- `LLM_MODEL` (default `neural-chat-7b-v3-3`)
- `LLM_API_KEY` (default `sk-no-key` for llama.cpp)
- `MCP_HTTP_URL` (default `http://127.0.0.1:8787/mcp`)
- `TOP_K` (default `5`)
- `EMBEDDING_MODEL` (default `thenlper/gte-small`)
- `SERVER_HOST`/`SERVER_PORT` (defaults `0.0.0.0:7000`)

## MCP JSON-RPC smoke tests

```bash
curl -s http://127.0.0.1:8787/mcp \
  -H 'Content-Type: application/json' \
  -d '{"jsonrpc":"2.0","id":1,"method":"initialize","params":{"protocolVersion":"2025-06-18"}}' | jq .

curl -s http://127.0.0.1:8787/mcp \
  -H 'Content-Type: application/json' \
  -d '{"jsonrpc":"2.0","id":2,"method":"tools/list","params":{}}' | jq .

curl -s http://127.0.0.1:8787/mcp \
  -H 'Content-Type: application/json' \
  -d '{"jsonrpc":"2.0","id":3,"method":"tools/call","params":{"name":"news.search","arguments":{"query":"UK election","limit":3}}}' | jq .
```

## Hitting the agent via `curl`

```bash
curl -s http://127.0.0.1:7000/v1/chat/completions \
  -H 'Content-Type: application/json' -H 'Authorization: Bearer sk-no-key' \
  -d '{
    "model": "neural-chat-7b-v3-3",
    "messages": [{"role":"user","content":"Summarize today\'s headlines about the Bank of England and relate it to inflation."}]
  }' | jq .
```

## Project structure

```
src/
├── 1_ingest        # BBC ingestion + embedding pipeline
├── 2_rag_agent     # Flask agent, MCP client, mock server, OpenSearch helpers
└── 3_rag_client    # Minimal CLI client that queries /v1/chat/completions
```

Key components inside `src/2_rag_agent`:

- `config.py` – environment-driven configuration for the agent
- `mock_mcp_server.py` – Flask JSON-RPC server returning mock `news.search` data
- `mcp_client.py` – thin MCP HTTP client
- `opensearch_store.py` – embeddings + OpenSearch retrieval helper
- `server.py` – Flask API orchestrating MCP + OpenSearch + llama.cpp

Each RAG request performs the following steps:

1. Embed the user query and retrieve contextual snippets from OpenSearch.
2. Call the LLM with a forced `news.search` tool invocation.
3. Execute the MCP `news.search` tool via JSON-RPC, appending the results to the
   conversation.
4. Call the LLM again to synthesise an answer using both the MCP output and the
   OpenSearch context, returning an OpenAI-compatible response payload.
