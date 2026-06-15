#!/usr/bin/env bash
# Build + deploy the SEC Filing Agent demo to Cloud Run. RUN FROM repo root.
# Uses Cloud Build (no local Docker; builds amd64 natively on GCP). Review first —
# uses the lab GCP project + keys in Secret Manager.
set -euo pipefail

PROJECT=proslm
REGION=us-central1
IMAGE=us-central1-docker.pkg.dev/proslm/proslm-repo/sec-filing-agent:v1

# Langfuse (audit-trail observability) — dedicated sec-filing-agent project. Public key
# + host are non-secret; the secret key lives in Secret Manager (create it once):
#   printf '%s' 'sk-lf-...' | gcloud secrets create SEC_LANGFUSE_SECRET_KEY \
#       --project proslm --data-file=-
LANGFUSE_PUBLIC_KEY=pk-lf-960e32be-ffe9-4fdd-9526-0a8b12e2f8cd
LANGFUSE_HOST=https://us.cloud.langfuse.com

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
  --timeout 120 --port 8080 \
  --set-env-vars EMB_MODEL=text-embedding-3-small,GEN_LLM_MODEL=gpt-4o-mini,RERANK=1,DAILY_QUOTA=150,RATE_LIMIT_PER_MIN=6,MAX_INPUT_CHARS=500,LANGFUSE_PUBLIC_KEY="$LANGFUSE_PUBLIC_KEY",LANGFUSE_HOST="$LANGFUSE_HOST",LANGFUSE_TRACING_ENVIRONMENT=production \
  --set-secrets OPENAI_API_KEY=OPENAI_API_KEY:latest,LANGFUSE_SECRET_KEY=SEC_LANGFUSE_SECRET_KEY:latest,DATABASE_URL=DATABASE_URL:latest

echo "Deployed. URL:"
gcloud run services describe sec-filing-agent --project "$PROJECT" --region "$REGION" \
  --format='value(status.url)'
