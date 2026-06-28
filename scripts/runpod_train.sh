#!/usr/bin/env bash
# runpod_train.sh
# ---------------
# Run this script inside a RunPod pod terminal to train the QLoRA adapter.
# Clones the repo, installs deps, runs training, and pushes the adapter to HF Hub.
#
# Required env vars (set in RunPod pod env or export before running):
#   HF_TOKEN          - HuggingFace token with read access to LLaMA 3.2 + write access to push adapter
#   GITHUB_REPO_URL   - HTTPS URL of your training repo (e.g. https://github.com/user/repo.git)
#   HF_HUB_REPO_ID    - HuggingFace repo to push adapter (e.g. username/llama-blog-sft)
#
# Optional:
#   TRAIN_EPOCHS      - Number of epochs (default: 3)
#   GITHUB_PAT        - GitHub personal access token (only needed for private repos)
#
# Usage (paste into RunPod web terminal):
#   curl -fsSL https://raw.githubusercontent.com/YOUR_USER/YOUR_REPO/main/scripts/runpod_train.sh | bash
#   -- OR --
#   bash /workspace/training/scripts/runpod_train.sh

set -euo pipefail

: "${HF_TOKEN:?HF_TOKEN is required}"
: "${GITHUB_REPO_URL:?GITHUB_REPO_URL is required}"
: "${HF_HUB_REPO_ID:?HF_HUB_REPO_ID is required}"

TRAIN_EPOCHS="${TRAIN_EPOCHS:-3}"

echo "=============================================="
echo " LLaMA QLoRA SFT Training — RunPod"
echo " Pod ID : ${RUNPOD_POD_ID:-unknown}"
echo " GPU    : $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | head -1 || echo 'N/A')"
echo " Epochs : $TRAIN_EPOCHS"
echo " HF repo: $HF_HUB_REPO_ID"
echo "=============================================="

# ── Authenticate with HuggingFace ─────────────────────────────────────────────
pip install --quiet --upgrade "huggingface_hub[cli]"
if command -v hf &>/dev/null; then
    hf auth login --token "$HF_TOKEN"
else
    huggingface-cli login --token "$HF_TOKEN"
fi

# ── Clone repo ────────────────────────────────────────────────────────────────
cd /workspace

# Support private repos via GitHub PAT
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

# ── Install dependencies ───────────────────────────────────────────────────────
echo "Installing dependencies..."
pip install --quiet -r requirements.txt

# ── Run training ───────────────────────────────────────────────────────────────
echo "Starting training..."
python scripts/train_qlora_sft.py \
    --model-name meta-llama/Llama-3.2-3B-Instruct \
    --train-file data/processed/sft/train_conversations.jsonl \
    --val-file data/processed/sft/val_conversations.jsonl \
    --output-root /workspace/model_output \
    --max-seq-length 2048 \
    --num-train-epochs "$TRAIN_EPOCHS" \
    --per-device-train-batch-size 1 \
    --gradient-accumulation-steps 16 \
    --learning-rate 2e-4 \
    --eval-steps 50 \
    --save-steps 50 \
    --push-to-hub \
    --hf-hub-repo-id "$HF_HUB_REPO_ID"

echo "=============================================="
echo " Training complete!"
echo " Adapter: https://huggingface.co/$HF_HUB_REPO_ID"
echo "=============================================="

# ── Self-terminate pod (only when running inside RunPod automation) ────────────
POD_ID="${RUNPOD_POD_ID:-}"
if [ -n "$POD_ID" ] && [ -n "${RUNPOD_API_KEY:-}" ]; then
    echo "Terminating pod $POD_ID..."
    curl -s -X POST "https://api.runpod.io/graphql?api_key=${RUNPOD_API_KEY}" \
        -H "Content-Type: application/json" \
        -d "{\"query\":\"mutation{podTerminate(input:{podId:\\\"${POD_ID}\\\"})}\"}"\
        > /dev/null
fi
