# One image for local runs and AWS Lambda.
# Lambda Web Adapter turns Lambda invocations into plain HTTP requests to uvicorn,
# so the app has no Lambda-specific code and `docker run -p 8080:8080` works locally.
# Non-HTTP events (S3 uploads) arrive as POST /events.

# --- 1. Resolve exact versions from uv.lock -------------------------------------------
FROM public.ecr.aws/docker/library/python:3.11-slim AS lock
RUN pip install --no-cache-dir uv
WORKDIR /src
COPY pyproject.toml uv.lock ./
# The lock resolves torch from PyPI (CUDA build + nvidia-* libs, ~5 GB). Lambda has no
# GPU, so drop those lines (torch, triton, nvidia-*, cuda-*) and install the CPU wheel
# of the *same* torch version below.
RUN uv export --frozen --no-hashes --no-emit-project \
      --extra local --extra claude --extra api --extra aws -o requirements.txt \
 && grep -E '^torch==' requirements.txt | cut -d' ' -f1 > torch.txt \
 && grep -vE '^(torch|triton|nvidia-|cuda-)' requirements.txt > requirements.notorch.txt

# --- 2. Runtime image -------------------------------------------------------------------
FROM public.ecr.aws/docker/library/python:3.11-slim

COPY --from=public.ecr.aws/awsguru/aws-lambda-adapter:0.8.4 /lambda-adapter /opt/extensions/lambda-adapter

WORKDIR /app
ENV PIP_NO_CACHE_DIR=1 \
    HF_HOME=/opt/hf \
    PORT=8080 \
    AWS_LWA_READINESS_CHECK_PATH=/health

COPY --from=lock /src/torch.txt /src/requirements.notorch.txt ./
# torch's own deps (sympy, networkx, ...) are already pinned in the lock, hence --no-deps.
RUN pip install -r requirements.notorch.txt \
 && pip install --no-deps -r torch.txt --index-url https://download.pytorch.org/whl/cpu

COPY pyproject.toml README.md ./
COPY src ./src
COPY prompts ./prompts
RUN pip install --no-deps .

# Bake models into the image: no download on cold start, no outbound HF traffic.
ARG EMBED_MODEL=BAAI/bge-small-en-v1.5
ARG RERANK_MODEL=cross-encoder/ms-marco-MiniLM-L-6-v2
RUN python -c "from sentence_transformers import SentenceTransformer, CrossEncoder; \
SentenceTransformer('${EMBED_MODEL}'); CrossEncoder('${RERANK_MODEL}')"

ENV HF_HUB_OFFLINE=1 \
    STUDY_RAG_EMBEDDER=hf \
    STUDY_RAG_EMBED_MODEL=${EMBED_MODEL} \
    STUDY_RAG_RERANKER=hf \
    STUDY_RAG_INDEX=/tmp/index.npz \
    STUDY_RAG_EMBED_CACHE=/tmp/embed_cache.npz \
    STUDY_RAG_TRACES=-

EXPOSE 8080
CMD ["sh", "-c", "uvicorn study_rag.api:app --host 0.0.0.0 --port ${PORT}"]
