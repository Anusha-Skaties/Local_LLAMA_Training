#!/usr/bin/env python3
"""
runpod_submit.py
----------------
Submit a QLoRA SFT training job to RunPod and poll until completion.
Runs locally or inside GitHub Actions.

Required env vars:
  RUNPOD_API_KEY    RunPod API key (from runpod.io/console/user/settings)
  HF_TOKEN          HuggingFace token with LLaMA read + Hub write access
  HF_HUB_REPO_ID    Repo to push the trained adapter (e.g. username/llama-blog-sft)
  GITHUB_REPO_URL   HTTPS URL of this repo (e.g. https://github.com/user/repo.git)

Optional env vars:
  GPU_TYPE          RunPod GPU (default: NVIDIA GeForce RTX 3090)
  TRAIN_EPOCHS      Number of training epochs (default: 3)
  MAX_WAIT_MINUTES  How long to poll before timing out (default: 120)
  GITHUB_PAT        GitHub PAT for private repos (optional)
"""

from __future__ import annotations

import base64
import json
import os
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

RUNPOD_GRAPHQL = "https://api.runpod.io/graphql"
SCRIPT_DIR = Path(__file__).resolve().parent


def graphql(api_key: str, query: str, variables: dict | None = None) -> dict:
    payload: dict = {"query": query}
    if variables:
        payload["variables"] = variables
    data = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(
        f"{RUNPOD_GRAPHQL}?api_key={api_key}",
        data=data,
        headers={"Content-Type": "application/json"},
    )
    try:
        with urllib.request.urlopen(req, timeout=30) as resp:
            return json.loads(resp.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        body = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"RunPod API HTTP {exc.code}: {body}") from exc


def build_startup_script(
    repo_url: str,
    hf_hub_repo_id: str,
    epochs: str,
    github_pat: str,
) -> str:
    # Build the authenticated clone URL if a PAT is provided.
    if github_pat:
        clone_expr = f'git clone "${{GITHUB_REPO_URL/https:\\/\\//https://x-access-token:{github_pat}@}}" training'
    else:
        clone_expr = 'git clone "$GITHUB_REPO_URL" training'

    return f"""#!/usr/bin/env bash
set -euo pipefail
echo "=== LLaMA QLoRA SFT — RunPod pod ${{RUNPOD_POD_ID:-unknown}} ==="
GPU=$(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | head -1 || echo N/A)
echo "GPU: $GPU"

pip install --quiet --upgrade huggingface_hub
huggingface-cli login --token "$HF_TOKEN"

cd /workspace
{clone_expr}
cd training

pip install --quiet -r requirements.txt

python scripts/train_qlora_sft.py \\
  --model-name meta-llama/Llama-3.2-3B-Instruct \\
  --train-file data/processed/sft/train_conversations.jsonl \\
  --val-file data/processed/sft/val_conversations.jsonl \\
  --output-root /workspace/model_output \\
  --max-seq-length 2048 \\
  --num-train-epochs {epochs} \\
  --per-device-train-batch-size 1 \\
  --gradient-accumulation-steps 16 \\
  --learning-rate 2e-4 \\
  --eval-steps 50 \\
  --save-steps 50 \\
  --push-to-hub \\
  --hf-hub-repo-id "$HF_HUB_REPO_ID"

echo "=== Training complete. Model pushed to https://huggingface.co/$HF_HUB_REPO_ID ==="

# Self-terminate so GHA polling detects completion.
if [ -n "${{RUNPOD_POD_ID:-}}" ] && [ -n "${{RUNPOD_API_KEY:-}}" ]; then
  curl -s -X POST "https://api.runpod.io/graphql?api_key=${{RUNPOD_API_KEY}}" \\
    -H "Content-Type: application/json" \\
    -d '{{"query":"mutation{{podTerminate(input:{{podId:\\"${{RUNPOD_POD_ID}}\\"}})}}}}'
fi
"""


def create_pod(
    api_key: str,
    gpu_type: str,
    startup_b64: str,
    env_vars: list[dict[str, str]],
) -> str:
    mutation = """
mutation CreatePod($input: PodFindAndDeployOnDemandInput!) {
  podFindAndDeployOnDemand(input: $input) {
    id
    machineId
    desiredStatus
  }
}
"""
    variables = {
        "input": {
            "cloudType": "SECURE",
            "gpuCount": 1,
            "volumeInGb": 10,
            "containerDiskInGb": 60,
            "minVcpuCount": 2,
            "minMemoryInGb": 15,
            "gpuTypeId": gpu_type,
            "name": "llama-sft",
            "imageName": "runpod/pytorch:2.1.1-py3.10-cuda12.1.1-devel-ubuntu22.04",
            # Decode and execute the base64-encoded startup script.
            "dockerArgs": f"bash -c 'echo {startup_b64} | base64 -d | bash'",
            "env": env_vars,
        }
    }
    result = graphql(api_key, mutation, variables)
    if "errors" in result:
        raise RuntimeError(f"Pod creation failed: {result['errors']}")
    pod = result["data"]["podFindAndDeployOnDemand"]
    return pod["id"]


def get_pod_status(api_key: str, pod_id: str) -> str:
    """Return desiredStatus, or 'TERMINATED' if the pod no longer exists."""
    query = """
query GetPod($input: PodFilter!) {
  pod(input: $input) {
    id
    desiredStatus
  }
}
"""
    result = graphql(api_key, query, {"input": {"podId": pod_id}})
    if "errors" in result:
        return "UNKNOWN"
    pod = result.get("data", {}).get("pod")
    if pod is None:
        return "TERMINATED"  # pod removed from API = clean exit
    return pod.get("desiredStatus", "UNKNOWN")


def main() -> int:
    api_key = os.environ.get("RUNPOD_API_KEY", "")
    hf_token = os.environ.get("HF_TOKEN", "")
    hf_hub_repo_id = os.environ.get("HF_HUB_REPO_ID", "")
    repo_url = os.environ.get("GITHUB_REPO_URL", "")

    missing = [
        k for k, v in [
            ("RUNPOD_API_KEY", api_key),
            ("HF_TOKEN", hf_token),
            ("HF_HUB_REPO_ID", hf_hub_repo_id),
            ("GITHUB_REPO_URL", repo_url),
        ]
        if not v
    ]
    if missing:
        print(f"ERROR: Missing required env vars: {', '.join(missing)}", file=sys.stderr)
        return 1

    gpu_type = os.environ.get("GPU_TYPE", "NVIDIA GeForce RTX 3090")
    epochs = os.environ.get("TRAIN_EPOCHS", "3")
    max_wait = int(os.environ.get("MAX_WAIT_MINUTES", "120"))
    github_pat = os.environ.get("GITHUB_PAT", "")

    script = build_startup_script(repo_url, hf_hub_repo_id, epochs, github_pat)
    startup_b64 = base64.b64encode(script.encode("utf-8")).decode("ascii")

    env_vars: list[dict[str, str]] = [
        {"key": "HF_TOKEN", "value": hf_token},
        {"key": "RUNPOD_API_KEY", "value": api_key},
        {"key": "HF_HUB_REPO_ID", "value": hf_hub_repo_id},
        {"key": "GITHUB_REPO_URL", "value": repo_url},
        {"key": "TRAIN_EPOCHS", "value": epochs},
        {"key": "HF_HUB_ENABLE_HF_TRANSFER", "value": "1"},
    ]

    print(f"Submitting training job to RunPod")
    print(f"  GPU     : {gpu_type}")
    print(f"  Epochs  : {epochs}")
    print(f"  HF repo : {hf_hub_repo_id}")

    pod_id = create_pod(api_key, gpu_type, startup_b64, env_vars)
    print(f"  Pod ID  : {pod_id}")
    print(f"  Dashboard: https://www.runpod.io/console/pods/{pod_id}")

    # Poll until the pod terminates (self-terminates after training).
    poll_interval = 60  # seconds
    max_polls = (max_wait * 60) // poll_interval

    print(f"\nPolling every {poll_interval}s (max {max_wait} min)...")
    for i in range(1, max_polls + 1):
        time.sleep(poll_interval)
        status = get_pod_status(api_key, pod_id)
        elapsed_min = (i * poll_interval) // 60
        print(f"  [{elapsed_min:3d}m/{max_wait}m] status={status}")

        if status in ("EXITED", "TERMINATED"):
            print(f"\nTraining complete!")
            print(f"Adapter: https://huggingface.co/{hf_hub_repo_id}")
            return 0
        if status == "DEAD":
            print("\nERROR: Pod died unexpectedly.", file=sys.stderr)
            print(f"Logs: https://www.runpod.io/console/pods/{pod_id}", file=sys.stderr)
            return 1

    print(
        f"\nTimeout after {max_wait} minutes.",
        f"Check: https://www.runpod.io/console/pods/{pod_id}",
        file=sys.stderr,
    )
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
