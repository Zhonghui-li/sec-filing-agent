# The document parser, as its own Cloud Run service. Build context = repo root.
#
# This image is heavy (docling pulls torch, opencv, rapidocr, scipy, transformers) and that is
# exactly why it is not the agent's image: the agent's cold start already costs 15-60s before the
# first line of Python runs, and every visitor pays it, while only an uploader needs a parser.
FROM python:3.11-slim

WORKDIR /app
ENV PYTHONUNBUFFERED=1

# CPU-only torch, as in the agent image: the GPU wheels are several GB and nothing here uses one.
RUN pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu
COPY ingest/requirements.txt ./ingest/requirements.txt
RUN pip install --no-cache-dir -r ingest/requirements.txt \
    && pip install --no-cache-dir fastapi==0.115.14 uvicorn==0.35.0 google-cloud-storage==3.4.1

# Pre-bake docling's layout and table models, the same reason the agent pre-bakes its reranker:
# otherwise the FIRST upload pays a model download on top of a cold start, over a connection we
# do not control, and a demo's first upload is the one someone is watching.
# layout + tableformer are what a filing needs; rapidocr covers a scanned upload. code_formula
# and picture_classifier are defaults this corpus has no use for, and they are not free in size.
RUN python -c "from docling.utils.model_downloader import download_models; \
    download_models(with_code_formula=False, with_picture_classifier=False)"

COPY ingest/ ./ingest/

ENV PORT=8080
CMD uvicorn ingest.server:app --host 0.0.0.0 --port ${PORT}
