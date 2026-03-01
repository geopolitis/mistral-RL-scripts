#!/usr/bin/env python3
# /// script
# requires-python = ">=3.10"
# dependencies = [
#     "basilica-sdk>=0.20.0",
# ]
# ///
"""
Ministral-3-14B-Instruct vLLM deployment for fine-tuning evaluation workflows.

Deploys mistralai/Ministral-3-14B-Instruct-2512 via vLLM as an OpenAI-compatible
inference endpoint. Designed as a baseline evaluation server and post-fine-tuning
serving endpoint for custom-dataset fine-tuning pipelines.

Model details (from https://huggingface.co/mistralai/Ministral-3-14B-Instruct-2512):
  - Architecture: Mistral3 (13.5B LM + 0.4B vision encoder = 14B total)
  - Default precision: FP8 (~14GB VRAM for weights)
  - Context window: 256K tokens (capped to 8192 here, see below)
  - Features: vision, tool calling, multilingual (11 langs)
  - License: Apache 2.0

GPU sizing rationale:
  - FP8 model weights: ~14 GB
  - vLLM runtime + CUDA context: ~1.5-2 GB
  - KV cache at max-model-len 8192: ~1.3 GB per sequence (GQA with 8 KV heads,
    128 head dim, 40 layers = 0.16 MB/token, but vLLM pre-allocates block pools)
  - Total base footprint: ~16.5 GB before KV cache pool
  - A single A100-80GB or H100-80GB leaves ~63 GB for KV cache, supporting high
    concurrent batch evaluation throughput
  - H100 preferred: native FP8 tensor cores, 3.35 TB/s memory bandwidth vs 2.0 on A100

Mistral-specific vLLM flags (all three are MANDATORY for the official checkpoint):
  - --tokenizer_mode mistral: loads Tekken tokenizer via mistral-common lib
    (the repo does NOT contain a standard HuggingFace tokenizer.json)
  - --config_format mistral: reads Mistral's params.json instead of HF config.json
  - --load_format mistral: loads Mistral's consolidated safetensors format
  NOTE: after fine-tuning and merging LoRA into HF format, drop config_format
  and load_format but KEEP tokenizer_mode mistral.

Why max-model-len is 8192, not 256K:
  - vLLM pre-allocates KV cache for max-model-len. At 256K, a single sequence
    would consume ~42 GB of KV cache, leaving no room for batching.
  - Fine-tuning evaluation prompts are typically 2K-8K tokens.
  - 8192 is the practical sweet spot: covers evaluation workloads while leaving
    maximum VRAM for concurrent request batching.

Why mistral-common is pip-installed at startup:
  - The mistral-common Python package (Tekken tokenizer backend) is NOT bundled
    in the standard vllm/vllm-openai Docker image.
  - Without it, --tokenizer_mode mistral fails on startup.
  - Adds ~30-60s to cold start. For production, build a custom image instead.

Fine-tuning workflow this endpoint supports:
  Phase 1: Baseline eval - serve Instruct FP8, score test dataset
  Phase 2: Train LoRA with TRL/SFT Trainer on BF16 weights (separate GPU)
  Phase 3: Iterative eval - add --enable-lora flag, load adapters via API
  Phase 4: Production - merge LoRA, serve merged model (drop config/load flags)

Startup timing estimate (cold start):
  - pip install mistral-common: ~30-60s
  - Model download (14GB FP8 over 10Gbps): ~15-30s
  - Model loading + CUDA init: ~30-60s
  - CUDA graph capture: ~10-30s
  - Total: ~2-3.5 minutes (health check initial_delay_seconds=300 covers this)

Requirements:
  - 1x A100-80GB or H100-80GB GPU (single GPU, no tensor parallelism needed)
  - vLLM >= 0.12.0 (using v0.16.0 image)
  - mistral-common >= 1.8.6 (installed at container startup)

Usage:
    export BASILICA_API_TOKEN="your-token"
    export HF_TOKEN="your-hf-token"
    cd deployment/
    uv run deploy_ministral.py

Reference:
    https://huggingface.co/mistralai/Ministral-3-14B-Instruct-2512
    https://mistral.ai/news/mistral-3
    https://arxiv.org/abs/2601.08584
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

MODEL = "mistralai/Ministral-3-14B-Instruct-2512"
INSTANCE_NAME = "ministral-14b-instruct"
VLLM_IMAGE = "vllm/vllm-openai:v0.16.0"
PORT = 8000


def deploy_ministral_14b() -> basilica.Deployment:
    """Deploy Ministral-3-14B-Instruct via vLLM with Mistral-native tokenizer."""
    client = BasilicaClient()

    # vLLM serve command assembled as a single bash -c invocation because
    # mistral-common must be pip-installed before vllm can start with
    # --tokenizer_mode mistral. The package provides the Tekken tokenizer
    # backend that Mistral3 models require.
    vllm_args = " ".join([
        f"vllm serve {MODEL}",
        "--host 0.0.0.0",
        f"--port {PORT}",
        "--tokenizer_mode mistral",
        "--config_format mistral",
        "--load_format mistral",
        "--dtype auto",                    # auto-detects FP8 from checkpoint
        "--max-model-len 8192",            # capped for eval workloads (not 256K)
        "--gpu-memory-utilization 0.92",   # safe ceiling for dedicated server
        "--max-num-seqs 32",               # concurrent sequences for batch eval
        "--enable-chunked-prefill",        # prevents long prompts blocking decodes
        "--max-num-batched-tokens 8192",   # chunk size for prefill interleaving
        "--disable-log-requests",          # reduce log noise in eval runs
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

    # Health check with 5-minute initial delay to cover:
    #   pip install (~60s) + model download (~30s) + loading (~60s) + warmup (~30s)
    # The startup probe with failure_threshold=36 at period_seconds=10 gives
    # 6 minutes total tolerance before the container is killed and restarted.
    health_check = HealthCheckConfig(
        liveness=ProbeConfig(
            path="/health",
            port=PORT,
            initial_delay_seconds=300,
            period_seconds=30,
            timeout_seconds=10,
            failure_threshold=3,
        ),
        readiness=ProbeConfig(
            path="/health",
            port=PORT,
            initial_delay_seconds=300,
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
            failure_threshold=36,
        ),
    )

    hf_token = os.environ.get("HF_TOKEN", "")

    env = {
        "HF_TOKEN": hf_token,
        "HF_HUB_DOWNLOAD_TIMEOUT": "600",
        "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True",
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

    print(f"Creating Ministral-3-14B-Instruct deployment...")
    print(f"  Model:  {MODEL}")
    print(f"  Image:  {VLLM_IMAGE}")
    print(f"  GPU:    1x H100/A100 (80GB)")
    print(f"  Memory: 48Gi")
    print(f"  Max context: 8192 tokens")
    print()

    response = client._client.create_deployment(request)
    deployment = basilica.Deployment._from_response(client, response)

    print(f"Deployment created: {deployment.name}")
    print(f"URL: {deployment.url}")
    print()
    print("Waiting for model to load (~2-3 minutes)...")
    print(f"Monitor with: basilica deploy logs {INSTANCE_NAME} --follow")
    print()

    try:
        deployment.wait_until_ready(timeout=600, silent=False)
    except basilica.exceptions.DeploymentFailed:
        print()
        print("Deployment is still loading.")
        print("mistral-common install + 14GB model download can take a few minutes.")
        print()
        print("Check progress with:")
        print(f"  basilica deploy logs {deployment.name} --follow")
        print()
        print("When ready, the API will be available at:")
        print(f"  {deployment.url}/v1/chat/completions")
        return deployment

    return deployment


if __name__ == "__main__":
    deployment = deploy_ministral_14b()

    if deployment.state == "Ready":
        print()
        print(f"Ministral-3-14B-Instruct deployment ready!")
        print(f"  Name:    {deployment.name}")
        print(f"  URL:     {deployment.url}")
        print(f"  State:   {deployment.state}")
        print()
        print("OpenAI-compatible API endpoints:")
        print(f"  Chat:    {deployment.url}/v1/chat/completions")
        print(f"  Models:  {deployment.url}/v1/models")
        print(f"  Health:  {deployment.url}/health")
        print()
        print("Example usage:")
        print(f'  curl {deployment.url}/v1/chat/completions \\')
        print('    -H "Content-Type: application/json" \\')
        print(f'    -d \'{{"model": "{MODEL}", '
              '"messages": [{"role": "user", '
              '"content": "Explain gradient descent in 3 sentences."}], '
              '"max_tokens": 256, "temperature": 0.1}}\''
              )
        print()
        print("Fine-tuning evaluation workflow:")
        print("  1. Run baseline eval against this endpoint")
        print("  2. Fine-tune with LoRA (TRL/SFT Trainer, BF16, separate GPU)")
        print("  3. Redeploy with --enable-lora to load adapters")
        print("  4. Merge best adapter, redeploy merged model")
