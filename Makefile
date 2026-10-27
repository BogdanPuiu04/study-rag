RUN := uv run --frozen
DATA ?= data/sample
Q ?= What is dropout?
BUCKET ?= $(shell cd infra/terraform 2>/dev/null && terraform output -raw data_bucket 2>/dev/null)

.PHONY: install test test-plumbing lint typecheck check ingest ask quiz eval baseline eval-answers candidates serve docker upload-docs

install:
	uv sync --frozen --extra dev

test:
	$(RUN) pytest -q

test-plumbing:
	$(RUN) pytest -q tests/test_plumbing.py tests/test_production.py tests/test_quiz.py tests/test_baseline.py

lint:
	$(RUN) ruff check .

typecheck:
	$(RUN) mypy

# Everything CI runs on every PR, minus the Terraform jobs.
check: lint typecheck test

ingest:
	$(RUN) study-rag ingest $(DATA)

ask:
	$(RUN) study-rag ask "$(Q)"

quiz:
	$(RUN) study-rag quiz --n 5

eval:
	$(RUN) study-rag eval eval/golden.jsonl $(if $(wildcard eval/baseline.json),--baseline eval/baseline.json)

# Accept the current retrieval quality as the new reference (commit eval/baseline.json
# in the same PR as the change that moved the numbers, and say why in the description).
baseline:
	$(RUN) study-rag eval eval/golden.jsonl --save-baseline eval/baseline.json

eval-answers:
	$(RUN) study-rag eval-answers eval/golden.jsonl --limit 5

candidates:
	$(RUN) study-rag feedback-candidates

serve:
	$(RUN) --extra api uvicorn study_rag.api:app --reload --port 8080

docker:
	docker build -t study-rag . && docker run --rm -p 8080:8080 \
	  -v $(PWD)/.index:/tmp/idx -e STUDY_RAG_INDEX=/tmp/idx/index.npz \
	  -e STUDY_RAG_LLM -e ANTHROPIC_API_KEY study-rag

# Your real course files -> S3 -> Lambda reindexes automatically (data/sample excluded).
upload-docs:
	@test -n "$(BUCKET)" || (echo "BUCKET not set and no terraform output found" && exit 1)
	aws s3 sync data/ s3://$(BUCKET)/docs/ --exclude "sample/*" --delete
