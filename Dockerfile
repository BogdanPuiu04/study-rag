# One image for local runs and AWS Lambda.
# Lambda Web Adapter turns Lambda invocations into plain HTTP requests to uvicorn,
# so the app has no Lambda-specific code and `docker run -p 8080:8080` works locally.
FROM public.ecr.aws/docker/library/python:3.11-slim

COPY --from=public.ecr.aws/awsguru/aws-lambda-adapter:0.8.4 /lambda-adapter /opt/extensions/lambda-adapter

WORKDIR /app
ENV PIP_NO_CACHE_DIR=1 \
    HF_HOME=/opt/hf \
    PORT=8080 \
    AWS_LWA_READINESS_CHECK_PATH=/health

# CPU-only torch keeps the image ~1 GB instead of ~5 GB with CUDA.
RUN pip install torch --index-url https://download.pytorch.org/whl/cpu

COPY pyproject.toml README.md ./
COPY src ./src
COPY prompts ./prompts
RUN pip install ".[local,claude,api,aws]"

# Bake models into the image: no download on cold start, no outbound HF traffic.
ARG EMBED_MODEL=BAAI/bge-small-en-v1.5
ARG RERANK_MODEL=cross-encoder/ms-marco-MiniLM-L-6-v2
RUN python -c "from sentence_transformers import SentenceTransformer, CrossEncoder; \
SentenceTransformer('${EMBED_MODEL}'); CrossEncoder('${RERANK_MODEL}')"

ENV HF_HUB_OFFLINE=1 \
    STUDY_RAG_EMBEDDER=hf \
    STUDY_RAG_RERANKER=hf \
    STUDY_RAG_INDEX=/tmp/index.pkl \
    STUDY_RAG_EMBED_CACHE=/tmp/embed_cache.pkl \
    STUDY_RAG_TRACES=-

EXPOSE 8080
CMD ["sh", "-c", "uvicorn study_rag.api:app --host 0.0.0.0 --port ${PORT}"]
