PYTHON ?= python
MODEL_PATH ?= /path/to/neural-chat-7b-v3-3.Q4_K_M.gguf
QUESTION ?= What happened with the Bank of England and inflation?
INDEX ?= bbc
DATA_DIR ?= ./bbc

.PHONY: ingest mcp-server llm-server rag-agent query

ingest:
	$(PYTHON) -m src.1_ingest.ingest --data-dir $(DATA_DIR) --index-name $(INDEX)

mcp-server:
	FLASK_APP=src/2_rag_agent/mock_mcp_server.py flask run -p 8787

llm-server:
	$(PYTHON) -m llama_cpp.server --model $(MODEL_PATH) --chat_format chatml-function-calling --host 127.0.0.1 --port 8080

rag-agent:
	$(PYTHON) -m src.2_rag_agent.server

query:
	$(PYTHON) -m src.3_rag_client --q "$(QUESTION)" --port 7000
