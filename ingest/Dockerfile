# The document parser, as its own Cloud Run service. Build context = repo root.
#
# This image is heavy (docling pulls torch, opencv, rapidocr, scipy, transformers) and that is
# exactly why it is not the agent's image: the agent's cold start already costs 15-60s before the
# first line of Python runs, and every visitor pays it, while only an uploader needs a parser.
FROM python:3.11-slim

WORKDIR /app
ENV PYTHONUNBUFFERED=1

# CPU-only torch, as in the agent image: the GPU wheels are several GB and nothing here uses one.
# torchvision comes from the SAME index and in the same command, which is the whole point. Asking
# for torch alone here left requirements.txt to pull torchvision from PyPI, and a PyPI torchvision
# against a +cpu torch is a mismatched pair: it imports fine and then dies at first use with
# "operator torchvision::nms does not exist". That surfaced two layers away as a
# ModuleNotFoundError for AutoProcessor inside a transformers import, which looks like anything
# except a wheel-pairing problem.
RUN pip install --no-cache-dir torch torchvision --index-url https://download.pytorch.org/whl/cpu
COPY ingest/requirements.txt ./ingest/requirements.txt
RUN pip install --no-cache-dir -r ingest/requirements.txt \
    && pip install --no-cache-dir fastapi==0.115.14 uvicorn==0.35.0 google-cloud-storage==3.4.1

# Image contract: the pair must still be the matched cpu build after requirements had their say.
# This CALLS the operator that was missing rather than just importing the module, because the
# import is what succeeded last time. A broken pair now fails the build instead of failing on
# someone's first upload.
RUN python -c "import torch, torchvision; from torchvision.ops import nms; \
    nms(torch.tensor([[0.,0.,1.,1.]]), torch.tensor([0.5]), 0.5); \
    print('torch', torch.__version__, '+ torchvision', torchvision.__version__, 'ops OK')"

# Pre-bake docling's layout and table models, the same reason the agent pre-bakes its reranker:
# otherwise the FIRST upload pays a model download on top of a cold start, over a connection we
# do not control, and a demo's first upload is the one someone is watching.
# layout + tableformer are what a filing needs; rapidocr covers a scanned upload. code_formula
# and picture_classifier are defaults this corpus has no use for, and they are not free in size.
RUN python -c "from docling.utils.model_downloader import download_models; \
    download_models(with_code_formula=False, with_picture_classifier=False)"

COPY ingest/ ./ingest/

# The service must import before it is deployed, not on its first request. The agent image learned
# this the expensive way (service/smoke.py).
RUN python -c "import ingest.server; print('ingest.server imports OK')"

ENV PORT=8080
CMD uvicorn ingest.server:app --host 0.0.0.0 --port ${PORT}
