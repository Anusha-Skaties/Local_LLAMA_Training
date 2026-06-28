#!/usr/bin/env bash
# runpod_eval.sh
# --------------
# Run inside a RunPod pod to evaluate the fine-tuned QLoRA adapter.
# Downloads the adapter from HuggingFace Hub, runs inference on the
# val set, scores with ROUGE-L + cosine similarity, and logs to MLflow.
#
# Required env vars:
#   HF_TOKEN          - HuggingFace token (read access to LLaMA + your adapter repo)
#   HF_HUB_REPO_ID    - HF repo where the adapter was pushed (e.g. username/llama-blog-sft)
#   GITHUB_REPO_URL   - HTTPS URL of this repo
#
# Optional:
#   MAX_SAMPLES       - Limit evaluation samples (default: all 9)
#   MLFLOW_TRACKING_URI - Remote MLflow server URI (default: logs locally)
#   GITHUB_PAT        - GitHub PAT for private repos
#
# Usage (paste into RunPod web terminal):
#   bash /workspace/training/scripts/runpod_eval.sh
#   -- OR after a training run is already on the pod --
#   cd /workspace/training && bash scripts/runpod_eval.sh

set -euo pipefail

: "${HF_TOKEN:?HF_TOKEN is required}"
: "${HF_HUB_REPO_ID:?HF_HUB_REPO_ID is required}"
: "${GITHUB_REPO_URL:?GITHUB_REPO_URL is required}"

MAX_SAMPLES="${MAX_SAMPLES:-}"
MLFLOW_TRACKING_URI="${MLFLOW_TRACKING_URI:-sqlite:///mlflow.db}"
ADAPTER_DIR="/workspace/adapter"

echo "=============================================="
echo " LLaMA QLoRA SFT Evaluation — RunPod"
echo " Pod ID : ${RUNPOD_POD_ID:-unknown}"
echo " GPU    : $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | head -1 || echo 'N/A')"
echo " HF repo: $HF_HUB_REPO_ID"
echo " MLflow : $MLFLOW_TRACKING_URI"
echo "=============================================="

# ── Authenticate with HuggingFace ─────────────────────────────────────────────
pip install --quiet --upgrade huggingface_hub
huggingface-cli login --token "$HF_TOKEN"

# ── Clone or update repo ───────────────────────────────────────────────────────
cd /workspace

CLONE_URL="$GITHUB_REPO_URL"
if [ -n "${GITHUB_PAT:-}" ]; then
    CLONE_URL="${GITHUB_REPO_URL/https:\/\//https://x-access-token:${GITHUB_PAT}@}"
fi

if [ -d "training/.git" ]; then
    echo "Repo already cloned, pulling latest..."
    cd training && git pull && cd ..
else
    git clone "$CLONE_URL" training
fi

cd training

# ── Install evaluation dependencies ───────────────────────────────────────────
echo "Installing evaluation dependencies..."
pip install --quiet -r requirements.txt
pip install --quiet -r requirements_eval.txt

# ── Download adapter from HuggingFace Hub ─────────────────────────────────────
echo "Downloading adapter from HuggingFace Hub: $HF_HUB_REPO_ID"
python - <<EOF
from huggingface_hub import snapshot_download
import os
snapshot_download(
    repo_id=os.environ["HF_HUB_REPO_ID"],
    local_dir="$ADAPTER_DIR",
    token=os.environ["HF_TOKEN"],
)
print(f"Adapter downloaded to $ADAPTER_DIR")
EOF

ls -lh "$ADAPTER_DIR"

# ── Build optional args ────────────────────────────────────────────────────────
EXTRA_ARGS=""
if [ -n "$MAX_SAMPLES" ]; then
    EXTRA_ARGS="--max-samples $MAX_SAMPLES"
fi

# ── Run evaluation ─────────────────────────────────────────────────────────────
echo "Running evaluation..."
MLFLOW_TRACKING_URI="$MLFLOW_TRACKING_URI" \
python scripts/evaluate_model.py \
    --adapter-dir "$ADAPTER_DIR" \
    --base-model meta-llama/Llama-3.2-3B-Instruct \
    --val-file data/processed/sft/val_conversations.jsonl \
    --corpus-file data/processed/blog_corpus.jsonl \
    --mlflow-experiment "qlora-blog-eval" \
    $EXTRA_ARGS

echo "=============================================="
echo " Evaluation complete!"
echo " Reports saved to: data/meta/eval_runs/"
echo "=============================================="

# ── Self-terminate pod ─────────────────────────────────────────────────────────
POD_ID="${RUNPOD_POD_ID:-}"
if [ -n "$POD_ID" ] && [ -n "${RUNPOD_API_KEY:-}" ]; then
    echo "Terminating pod $POD_ID..."
    curl -s -X POST "https://api.runpod.io/graphql?api_key=${RUNPOD_API_KEY}" \
        -H "Content-Type: application/json" \
        -d "{\"query\":\"mutation{podTerminate(input:{podId:\\\"${POD_ID}\\\"})}\"}"\
        > /dev/null
fi
