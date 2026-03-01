#!/usr/bin/env python3
# /// script
# requires-python = ">=3.10"
# dependencies = [
#     "basilica-sdk>=0.20.0",
# ]
# ///
"""
Ministral-3-3B-Instruct-sec: LoRA-adapted 3B model served via vLLM.

Deploys the base model mistralai/Ministral-3-3B-Instruct-2512-BF16 with a LoRA
adapter from llmtrace/Ministral-3-3B-Instruct-sec via vLLM's --enable-lora.

Model details:
  - Base: mistralai/Ministral-3-3B-Instruct-2512-BF16 (Mistral3, 3B params, BF16)
  - Adapter: llmtrace/Ministral-3-3B-Instruct-sec (LoRA r=32, alpha=64)
  - Architecture: Mistral3ForConditionalGeneration (26 layers, 3072 dim, GQA 32/8)
  - Fine-tuning: GRPO via TRL, targeting q/k/v/o/gate/up/down projections
  - Context: 256K tokens (capped to 8192 here)

GPU sizing:
  - BF16 weights: ~6 GB
  - LoRA adapter overhead: ~200 MB
  - vLLM runtime: ~1.5 GB
  - Total: ~8 GB base, rest available for KV cache
  - Targeting A100/H100 80GB for CUDA driver compatibility with vLLM v0.16.0
    (older nodes with 24GB GPUs may have incompatible driver versions)

Mistral-specific vLLM flags:
  - The 3B base model uses Mistral3 architecture (Mistral3ForConditionalGeneration)
    with a vision encoder. Same as the 14B variant, it requires:
    --tokenizer_mode mistral, --config_format mistral, --load_format mistral
  - mistral-common >= 1.8.6 must be pip-installed (not in vLLM image)

LoRA serving:
  - --enable-lora activates adapter support
  - --lora-modules maps adapter name to HuggingFace repo
  - --max-lora-rank 32 matches the adapter's r=32
  - At inference, use model="ministral-3b-sec" to route through the adapter,
    or model="mistralai/Ministral-3-3B-Instruct-2512-BF16" for the raw base model

Usage:
    export BASILICA_API_TOKEN="your-token"
    export HF_TOKEN="your-hf-token"
    cd deployment/
    uv run deploy_ministral_3b_lora.py

Reference:
    https://huggingface.co/llmtrace/Ministral-3-3B-Instruct-sec
    https://huggingface.co/mistralai/Ministral-3-3B-Instruct-2512-BF16
"""
import os

import basilica
from basilica import (
    BasilicaClient,
    CreateDeploymentRequest,
    GpuRequirementsSpec,
    HealthCheckConfig,
    ProbeConfig,
    ResourceRequirements,
)

BASE_MODEL = "mistralai/Ministral-3-3B-Instruct-2512-BF16"
ADAPTER_REPO = "llmtrace/Ministral-3-3B-Instruct-sec"
ADAPTER_NAME = "ministral-3b-sec"
INSTANCE_NAME = "ministral-3b-sec"
VLLM_IMAGE = "vllm/vllm-openai:v0.16.0"
PORT = 8000


def deploy_ministral_3b_lora() -> basilica.Deployment:
    """Deploy Ministral-3B base + LoRA adapter via vLLM."""
    client = BasilicaClient()

    vllm_args = " ".join([
        f"vllm serve {BASE_MODEL}",
        "--host 0.0.0.0",
        f"--port {PORT}",
        "--tokenizer_mode mistral",
        "--config_format mistral",
        "--load_format mistral",
        "--dtype auto",
        "--max-model-len 8192",
        "--gpu-memory-utilization 0.90",
        "--max-num-seqs 64",
        "--enable-chunked-prefill",
        "--max-num-batched-tokens 8192",
        "--disable-log-requests",
        "--enable-lora",
        f"--lora-modules {ADAPTER_NAME}={ADAPTER_REPO}",
        "--max-lora-rank 32",
        "--max-loras 2",
    ])

    startup_cmd = f"pip install --no-cache-dir mistral-common>=1.8.6 && {vllm_args}"

    gpu_spec = GpuRequirementsSpec(
        count=1,
        model=["H100", "A100"],
        min_cuda_version=None,
        min_gpu_memory_gb=80,
    )

    resources = ResourceRequirements(
        cpu="8",
        memory="48Gi",
        gpus=gpu_spec,
    )

    # 3B model loads fast: ~5s download, ~5s load, ~10s warmup
    # pip install adds ~30-60s. Startup probe gives 4 min total tolerance.
    health_check = HealthCheckConfig(
        liveness=ProbeConfig(
            path="/health",
            port=PORT,
            initial_delay_seconds=180,
            period_seconds=30,
            timeout_seconds=10,
            failure_threshold=3,
        ),
        readiness=ProbeConfig(
            path="/health",
            port=PORT,
            initial_delay_seconds=180,
            period_seconds=10,
            timeout_seconds=5,
            failure_threshold=3,
        ),
        startup=ProbeConfig(
            path="/health",
            port=PORT,
            initial_delay_seconds=0,
            period_seconds=10,
            timeout_seconds=5,
            failure_threshold=24,
        ),
    )

    hf_token = os.environ.get("HF_TOKEN", "")

    env = {
        "HF_TOKEN": hf_token,
        "HF_HUB_DOWNLOAD_TIMEOUT": "300",
        "PYTORCH_ALLOC_CONF": "expandable_segments:True",
        "VLLM_LOGGING_LEVEL": "INFO",
    }

    request = CreateDeploymentRequest(
        instance_name=INSTANCE_NAME,
        image=VLLM_IMAGE,
        replicas=1,
        port=PORT,
        command=["bash"],
        args=["-c", startup_cmd],
        env=env,
        resources=resources,
        ttl_seconds=7200,
        public=True,
        storage=None,
        health_check=health_check,
    )

    print(f"Creating Ministral-3B + LoRA deployment...")
    print(f"  Base:    {BASE_MODEL}")
    print(f"  Adapter: {ADAPTER_REPO}")
    print(f"  Image:   {VLLM_IMAGE}")
    print(f"  GPU:     1x H100/A100 (80GB)")
    print(f"  Memory:  48Gi")
    print()

    response = client._client.create_deployment(request)
    deployment = basilica.Deployment._from_response(client, response)

    print(f"Deployment created: {deployment.name}")
    print(f"URL: {deployment.url}")
    print()
    print("Waiting for model to load (~1-2 minutes)...")
    print(f"Monitor with: basilica deploy logs {INSTANCE_NAME} --follow")
    print()

    try:
        deployment.wait_until_ready(timeout=600, silent=False)
    except basilica.exceptions.DeploymentFailed:
        print()
        print("Deployment is still loading.")
        print()
        print("Check progress with:")
        print(f"  basilica deploy logs {deployment.name} --follow")
        print()
        print("When ready, the API will be available at:")
        print(f"  {deployment.url}/v1/chat/completions")
        return deployment

    return deployment


if __name__ == "__main__":
    deployment = deploy_ministral_3b_lora()

    if deployment.state == "Ready":
        print()
        print(f"Ministral-3B-sec deployment ready!")
        print(f"  Name:    {deployment.name}")
        print(f"  URL:     {deployment.url}")
        print(f"  State:   {deployment.state}")
        print()
        print("OpenAI-compatible API endpoints:")
        print(f"  Chat:    {deployment.url}/v1/chat/completions")
        print(f"  Models:  {deployment.url}/v1/models")
        print(f"  Health:  {deployment.url}/health")
        print()
        print(f"Use model=\"{ADAPTER_NAME}\" for the fine-tuned adapter,")
        print(f"or model=\"{BASE_MODEL}\" for the raw base model.")
        print()
        print("Example usage (LoRA adapter):")
        print(f'  curl {deployment.url}/v1/chat/completions \\')
        print('    -H "Content-Type: application/json" \\')
        print(f'    -d \'{{"model": "{ADAPTER_NAME}", '
              '"messages": [{"role": "user", '
              '"content": "What is prompt injection?"}], '
              '"max_tokens": 256, "temperature": 0.1}}\''
              )
