PY := .venv/bin
DATA ?= data/sample
Q ?= What is dropout?

.PHONY: install test lint ingest ask eval eval-answers serve docker

install:
	uv venv && uv pip install -e ".[dev]"

test:
	$(PY)/pytest -q

lint:
	$(PY)/ruff check .

ingest:
	$(PY)/study-rag ingest $(DATA)

ask:
	$(PY)/study-rag ask "$(Q)"

eval:
	$(PY)/study-rag eval eval/golden.jsonl

eval-answers:
	$(PY)/study-rag eval-answers eval/golden.jsonl --limit 5

serve:
	$(PY)/uvicorn study_rag.api:app --reload --port 8080

docker:
	docker build -t study-rag . && docker run --rm -p 8080:8080 \
	  -v $(PWD)/.index:/tmp/idx -e STUDY_RAG_INDEX=/tmp/idx/memory.pkl \
	  -e STUDY_RAG_LLM -e ANTHROPIC_API_KEY study-rag
