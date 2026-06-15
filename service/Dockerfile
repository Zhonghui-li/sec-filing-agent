# SEC Filing Agent — live demo for Cloud Run. Build context = repo root.
# Reranker pre-baked so cold starts don't download it. The filing text lives in
# pgvector (DATABASE_URL); only the exact-numbers JSON is baked into the image.
FROM python:3.11-slim

WORKDIR /app
ENV PYTHONUNBUFFERED=1 \
    RERANK=1 \
    HF_HUB_DISABLE_PROGRESS_BARS=1 \
    TRANSFORMERS_VERBOSITY=error

RUN pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu
COPY service/requirements.txt ./service/requirements.txt
RUN pip install --no-cache-dir -r service/requirements.txt

# Pre-bake the reranker model (no runtime download / cold-start stall).
RUN python -c "from sentence_transformers import CrossEncoder; CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2')"

COPY agents/ ./agents/
COPY data/financials.json ./data/financials.json
COPY service/ ./service/

ENV PORT=8080
CMD uvicorn service.app:app --host 0.0.0.0 --port ${PORT}
