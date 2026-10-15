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

# Pre-bake the models by PARSING SOMETHING, not by naming what to download. The reason the
# agent pre-bakes its reranker applies here -- otherwise the first upload pays a model fetch on
# top of a cold start, over a connection we do not control -- but naming the models got it wrong:
# download_models() fetches the set IT knows about, while the default pipeline resolves its own
# (do_ocr is on and the OCR engine is "auto", chosen at runtime). The deployed image passed that
# step and then failed on a real document with "cannot find the appropriate snapshot folder ...
# pass a HF_TOKEN", because the two sets were not the same set.
#
# Converting a document runs the exact code an upload runs, so whatever it reaches for is what
# gets cached, and anything it cannot reach fails the BUILD rather than someone's first upload.
RUN python -c "\
from reportlab.pdfgen import canvas; \
c = canvas.Canvas('/tmp/smoke.pdf'); c.drawString(72, 720, 'Total revenues 101,808'); c.save(); \
from docling.document_converter import DocumentConverter; \
md = DocumentConverter().convert('/tmp/smoke.pdf').document.export_to_markdown(); \
assert '101,808' in md, 'docling parsed the page but lost its text: ' + repr(md[:200]); \
print('docling parse OK:', md.strip()[:60])"

COPY ingest/ ./ingest/

# The service must import before it is deployed, not on its first request. The agent image learned
# this the expensive way (service/smoke.py).
RUN python -c "import ingest.server; print('ingest.server imports OK')"

ENV PORT=8080
CMD uvicorn ingest.server:app --host 0.0.0.0 --port ${PORT}
