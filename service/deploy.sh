#!/usr/bin/env bash
# Build + deploy the SEC Filing Agent demo to Cloud Run. RUN FROM repo root.
# Uses Cloud Build (no local Docker; builds amd64 natively on GCP). Review first —
# uses the lab GCP project + keys in Secret Manager.
set -euo pipefail

PROJECT=proslm
REGION=us-central1
REPO=us-central1-docker.pkg.dev/proslm/proslm-repo/sec-filing-agent

# Tag by commit, not :v1. Every build used to overwrite the same tag, so the tag could not say
# which build was running: Cloud Run pins the digest internally and rollback stayed exact, but
# "what is live" could only be answered by reading a revision's digest. A dirty tree gets a
# -dirty suffix so an uncommitted build can never be mistaken for the commit it was built near.
GITSHA=$(git rev-parse --short HEAD)
if ! git diff-index --quiet HEAD -- 2>/dev/null; then GITSHA="${GITSHA}-dirty"; fi
IMAGE="${REPO}:${GITSHA}"
echo "Building ${IMAGE}"

# Langfuse (audit-trail observability) — dedicated sec-filing-agent project. Public key
# + host are non-secret; the secret key lives in Secret Manager (create it once):
#   printf '%s' 'sk-lf-...' | gcloud secrets create SEC_LANGFUSE_SECRET_KEY \
#       --project proslm --data-file=-
LANGFUSE_PUBLIC_KEY=pk-lf-960e32be-ffe9-4fdd-9526-0a8b12e2f8cd
LANGFUSE_HOST=https://us.cloud.langfuse.com

# Google sign-in — turns on per-account isolation for uploaded documents (the login page appears).
# The web client id is public (not a secret); its Authorized JavaScript origins must include this
# service's Cloud Run URL. Unset it to run the public demo without login (anonymous per-session).
GOOGLE_CLIENT_ID=759005971862-u65f4afgm23gg9ukclmlpdaqg0796lgr.apps.googleusercontent.com

# Uploaded originals live here rather than on the container filesystem, which dies with the
# instance (this service scales to zero after ~13 idle minutes). Unset it and uploads fall back
# to local disk, which is right for a dev run and wrong for Cloud Run.
GCS_BUCKET=proslm-sec-filing-agent-uploads

# The docling parser, deployed separately by ingest/deploy.sh. Unset it and the service looks for
# ingest/.venv instead, which exists on a dev machine and has never existed in the image.
INGEST_URL=$(gcloud run services describe sec-filing-parser --project "$PROJECT" \
               --region "$REGION" --format='value(status.url)' 2>/dev/null || true)
if [ -z "$INGEST_URL" ]; then
  echo "WARNING: sec-filing-parser is not deployed — uploads will fail. Run ingest/deploy.sh first." >&2
fi

# 1. Build on GCP (native amd64) and push to Artifact Registry.
gcloud builds submit --project "$PROJECT" \
  --config service/cloudbuild.yaml \
  --substitutions=_IMAGE="$IMAGE" .

# 2. Deploy. max-instances=1 makes the in-memory daily quota an exact global cap.
gcloud run deploy sec-filing-agent \
  --image "$IMAGE" \
  --project "$PROJECT" --region "$REGION" \
  --allow-unauthenticated \
  --max-instances=1 \
  --memory 2Gi --cpu 2 \
  --no-cpu-throttling \
  --timeout 120 --port 8080 \
  --set-env-vars EMB_MODEL=text-embedding-3-small,GEN_LLM_MODEL=gpt-5.4-mini,REASONING_EFFORT=low,RESPONSES_API=1,RERANK=1,DAILY_QUOTA=300,RATE_LIMIT_PER_MIN=6,MAX_INPUT_CHARS=500,LANGFUSE_PUBLIC_KEY="$LANGFUSE_PUBLIC_KEY",LANGFUSE_HOST="$LANGFUSE_HOST",LANGFUSE_TRACING_ENVIRONMENT=community,GOOGLE_CLIENT_ID="$GOOGLE_CLIENT_ID",GCS_BUCKET="$GCS_BUCKET",INGEST_URL="$INGEST_URL" \
  --set-secrets OPENAI_API_KEY=OPENAI_API_KEY:latest,LANGFUSE_SECRET_KEY=SEC_LANGFUSE_SECRET_KEY:latest,DATABASE_URL=DATABASE_URL:latest

echo "Deployed. URL:"
gcloud run services describe sec-filing-agent --project "$PROJECT" --region "$REGION" \
  --format='value(status.url)'
