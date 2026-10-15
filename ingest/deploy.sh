#!/usr/bin/env bash
# Build + deploy the DOCUMENT PARSER as its own Cloud Run service. RUN FROM repo root.
#
# Separate from the agent on purpose: docling's dependencies are ~1.2 GB, and the agent's cold
# start already costs 15-60s before the first line of Python. Putting the parser in that image
# would charge every visitor for a capability only an uploader uses. Here it scales to zero and
# its cold start is paid by an upload, which is async and whose UI already says it takes minutes.
#
# This service is PRIVATE (--no-allow-unauthenticated). The agent calls it with an identity
# token; nothing on the internet can reach it, which is what stops a stranger from running
# docling on our CPU by finding the URL.
set -euo pipefail

PROJECT=proslm
REGION=us-central1
REPO=us-central1-docker.pkg.dev/proslm/proslm-repo/sec-filing-parser

GITSHA=$(git rev-parse --short HEAD)
if ! git diff-index --quiet HEAD -- 2>/dev/null; then GITSHA="${GITSHA}-dirty"; fi
IMAGE="${REPO}:${GITSHA}"
echo "Building ${IMAGE}"

gcloud builds submit --project "$PROJECT" \
  --config ingest/cloudbuild.yaml \
  --substitutions=_IMAGE="$IMAGE" .

# memory: docling holds the layout + table models while parsing a full filing, which does not fit
# in the agent's 2Gi. timeout 1800: a full 10-Q measured at ~17 minutes on 2 vCPU.
#
# concurrency=1 is the setting that matters here, and leaving it at the default 160 is what made
# a working service look broken. Cloud Run routes by concurrency: at 160 it believes ONE instance
# can take 160 simultaneous parses, so it keeps queueing onto a box already saturated by a single
# 17-minute CPU-bound job instead of starting another, and once the queue fills it answers 429.
# A parse uses a whole instance, so an instance should advertise exactly one.
gcloud run deploy sec-filing-parser \
  --image "$IMAGE" \
  --project "$PROJECT" --region "$REGION" \
  --no-allow-unauthenticated \
  --concurrency=1 \
  --max-instances=4 \
  --memory 4Gi --cpu 2 \
  --timeout 1800 --port 8080 \
  --set-env-vars EMB_MODEL=text-embedding-3-small \
  --set-secrets OPENAI_API_KEY=OPENAI_API_KEY:latest,DATABASE_URL=DATABASE_URL:latest

echo "Parser URL (set this as INGEST_URL on the agent service):"
gcloud run services describe sec-filing-parser --project "$PROJECT" --region "$REGION" \
  --format='value(status.url)'
