"""
Deployment config generation: docker-compose, vLLM start.sh, k8s.yaml, systemd unit.

All configs are generated from scale + runtime recommendation.
Output: ~/.shipai/deployments/{sanitized_model_name}/
"""
from __future__ import annotations

import re
from pathlib import Path
from typing import Dict, Optional

from app.install.runtime_recommender import RuntimeRecommendation
from app.install.scale_classifier import ScaleProfile

_OUTPUT_BASE = Path.home() / ".shipai" / "deployments"


def _sanitize_name(model_id: str) -> str:
    """Convert model_id to a filesystem-safe directory name."""
    return re.sub(r"[^a-zA-Z0-9_\-]", "_", model_id)[:64]


def generate_docker_compose(
    scale: ScaleProfile,
    rec: RuntimeRecommendation,
    model_id: str,
    hf_cache_path: Optional[str] = None,
) -> str:
    """
    Generate a docker-compose.yml for the recommended runtime.

    Multi-GPU: includes device_ids for all GPUs.
    Single GPU: includes count=1.
    CPU-only: no GPU section.
    """
    model_safe = _sanitize_name(model_id)
    hf_volume = f"      - {hf_cache_path}:/root/.cache/huggingface\n" if hf_cache_path else ""

    if rec.runtime == "ollama":
        gpu_section = ""
        if rec.gpu_device_ids:
            ids_str = ", ".join(f'"{i}"' for i in rec.gpu_device_ids)
            gpu_section = f"""    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              device_ids: [{ids_str}]
              capabilities: [gpu]
"""
        return f"""# ShipAI generated docker-compose — Ollama
# Scale tier: {scale.tier} | Runtime: ollama
version: "3.8"
services:
  ollama:
    image: ollama/ollama:latest
    container_name: shipai_ollama
    restart: unless-stopped
    ports:
      - "11434:11434"
    volumes:
      - ollama_data:/root/.ollama
{hf_volume}{gpu_section}
  # Auto-pull model on startup
  model_puller:
    image: ollama/ollama:latest
    depends_on:
      - ollama
    entrypoint: ["/bin/sh", "-c", "sleep 5 && ollama pull {model_id}"]
    environment:
      - OLLAMA_HOST=http://ollama:11434

volumes:
  ollama_data:
"""

    if rec.runtime == "vllm":
        device_ids = rec.gpu_device_ids or list(range(scale.gpu_count))
        ids_str = ", ".join(f'"{i}"' for i in device_ids)
        tp = rec.tensor_parallel
        extra_args = " ".join(rec.launch_args[2:]) if len(rec.launch_args) > 2 else ""
        return f"""# ShipAI generated docker-compose — vLLM
# Scale tier: {scale.tier} | TP={tp} | Interconnect: {scale.interconnect}
version: "3.8"
services:
  vllm:
    image: vllm/vllm-openai:latest
    container_name: shipai_vllm
    restart: unless-stopped
    ports:
      - "8000:8000"
    volumes:
      - vllm_cache:/root/.cache/huggingface
{hf_volume}    command: ["--model", "{model_id}", "--tensor-parallel-size", "{tp}", {extra_args}]
    shm_size: "10gb"
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              device_ids: [{ids_str}]
              capabilities: [gpu]

volumes:
  vllm_cache:
"""

    if rec.runtime == "llamacpp":
        threads = next(
            (rec.launch_args[i + 1] for i, a in enumerate(rec.launch_args) if a == "--threads"),
            "8"
        )
        return f"""# ShipAI generated docker-compose — llama.cpp (CPU-only)
# Scale tier: {scale.tier}
version: "3.8"
services:
  llamacpp:
    image: ghcr.io/ggerganov/llama.cpp:server
    container_name: shipai_llamacpp
    restart: unless-stopped
    ports:
      - "8080:8080"
    volumes:
      - ${{HOME}}/models:/models
{hf_volume}    command: [
      "--model", "/models/{model_safe}.gguf",
      "--threads", "{threads}",
      "--ctx-size", "4096",
      "--host", "0.0.0.0",
      "--port", "8080"
    ]
    environment:
      - OMP_NUM_THREADS={threads}
"""

    # Ray + vLLM
    tp = rec.tensor_parallel
    pp = rec.pipeline_parallel
    return f"""# ShipAI generated docker-compose — Ray + vLLM (hyperscale)
# Scale tier: {scale.tier} | TP={tp} | PP={pp}
version: "3.8"
services:
  ray_head:
    image: rayproject/ray:latest-gpu
    container_name: shipai_ray_head
    command: ["ray", "start", "--head", "--dashboard-host=0.0.0.0"]
    ports:
      - "6379:6379"
      - "8265:8265"
    runtime: nvidia
    environment:
      - NVIDIA_VISIBLE_DEVICES=all

  vllm:
    image: vllm/vllm-openai:latest
    container_name: shipai_vllm_distributed
    depends_on:
      - ray_head
    ports:
      - "8000:8000"
    command: [
      "--model", "{model_id}",
      "--tensor-parallel-size", "{tp}",
      "--pipeline-parallel-size", "{pp}",
      "--distributed-executor-backend", "ray"
    ]
    runtime: nvidia
    environment:
      - NVIDIA_VISIBLE_DEVICES=all
      - RAY_ADDRESS=ray://ray_head:6379
"""


def generate_vllm_launch_script(
    rec: RuntimeRecommendation,
    model_id: str,
    scale: ScaleProfile,
) -> str:
    """Generate a shell script to launch vLLM with correct args."""
    env_lines = "\n".join(f"export {k}={v}" for k, v in rec.env_vars.items())
    args = " \\\n  ".join(rec.launch_args) if rec.launch_args else f"serve {model_id}"
    return f"""#!/bin/bash
# ShipAI generated vLLM launch script
# Scale: {scale.tier} | TP={rec.tensor_parallel} | PP={rec.pipeline_parallel}
# {rec.explanation}

set -e

{env_lines}

echo "Starting vLLM for {model_id}..."
echo "Tensor parallel: {rec.tensor_parallel}"
echo "Pipeline parallel: {rec.pipeline_parallel}"

python -m vllm.entrypoints.openai.api_server \\
  {args}

# Health check endpoint: http://localhost:8000/health
# OpenAI-compatible API: http://localhost:8000/v1
"""


def generate_k8s_deployment(
    scale: ScaleProfile,
    rec: RuntimeRecommendation,
    model_id: str,
) -> str:
    """Generate a minimal Kubernetes Deployment + Service YAML."""
    model_safe = _sanitize_name(model_id)
    gpu_count = scale.gpu_count
    runtime_image = {
        "vllm": "vllm/vllm-openai:latest",
        "ray_vllm": "vllm/vllm-openai:latest",
        "ollama": "ollama/ollama:latest",
    }.get(rec.runtime, "vllm/vllm-openai:latest")

    port = 8000 if rec.runtime != "ollama" else 11434
    args_yaml = "\n".join(f'        - "{a}"' for a in rec.launch_args) if rec.launch_args else f'        - "serve"\n        - "{model_id}"'

    return f"""# ShipAI generated Kubernetes deployment
# Scale: {scale.tier} | Runtime: {rec.runtime} | GPUs: {gpu_count}
apiVersion: apps/v1
kind: Deployment
metadata:
  name: shipai-{model_safe}
  labels:
    app: shipai-llm
    model: {model_safe}
spec:
  replicas: 1
  selector:
    matchLabels:
      app: shipai-llm
  template:
    metadata:
      labels:
        app: shipai-llm
    spec:
      containers:
        - name: llm-server
          image: {runtime_image}
          args:
{args_yaml}
          ports:
            - containerPort: {port}
          resources:
            limits:
              nvidia.com/gpu: {gpu_count}
            requests:
              nvidia.com/gpu: {gpu_count}
          volumeMounts:
            - name: hf-cache
              mountPath: /root/.cache/huggingface
      volumes:
        - name: hf-cache
          hostPath:
            path: /data/hf_cache
            type: DirectoryOrCreate
---
apiVersion: v1
kind: Service
metadata:
  name: shipai-{model_safe}-svc
spec:
  selector:
    app: shipai-llm
  ports:
    - protocol: TCP
      port: {port}
      targetPort: {port}
  type: ClusterIP
"""


def generate_systemd_unit(
    rec: RuntimeRecommendation,
    model_id: str,
) -> str:
    """Generate a systemd unit file for bare-metal Linux servers."""
    model_safe = _sanitize_name(model_id)
    env_lines = "\n".join(f"Environment={k}={v}" for k, v in rec.env_vars.items())
    if rec.runtime == "ollama":
        exec_start = "ollama serve"
        after_service = "ExecStartPost=/bin/sh -c 'sleep 3 && ollama pull {model_id}'"
    elif rec.runtime in ("vllm", "ray_vllm"):
        args = " ".join(rec.launch_args)
        exec_start = f"python -m vllm.entrypoints.openai.api_server {args}"
        after_service = ""
    else:
        args = " ".join(rec.launch_args)
        exec_start = f"llama-server {args}"
        after_service = ""

    return f"""# ShipAI generated systemd unit
# Install: sudo cp llm.service /etc/systemd/system/shipai-{model_safe}.service
#          sudo systemctl daemon-reload
#          sudo systemctl enable --now shipai-{model_safe}

[Unit]
Description=ShipAI LLM Server — {model_id}
After=network.target
Wants=network-online.target

[Service]
Type=simple
User=%i
Restart=always
RestartSec=10
{env_lines}
ExecStart={exec_start}
{after_service}
StandardOutput=journal
StandardError=journal
SyslogIdentifier=shipai-llm

[Install]
WantedBy=multi-user.target
"""


def write_deployment_configs(
    scale: ScaleProfile,
    rec: RuntimeRecommendation,
    model_id: str,
    hf_cache_path: Optional[str] = None,
    output_dir: Optional[Path] = None,
) -> Dict[str, str]:
    """
    Write all deployment configs to disk.
    Returns dict mapping config type → absolute path.
    """
    out_dir = output_dir or (_OUTPUT_BASE / _sanitize_name(model_id))
    out_dir.mkdir(parents=True, exist_ok=True)

    configs: Dict[str, str] = {}

    # docker-compose.yml
    compose = generate_docker_compose(scale, rec, model_id, hf_cache_path)
    compose_path = out_dir / "docker-compose.yml"
    compose_path.write_text(compose, encoding="utf-8")
    configs["docker_compose"] = str(compose_path)

    # start.sh (vLLM / llama.cpp launch script)
    if rec.runtime in ("vllm", "ray_vllm", "llamacpp"):
        script = generate_vllm_launch_script(rec, model_id, scale)
        script_path = out_dir / "start.sh"
        script_path.write_text(script, encoding="utf-8")
        try:
            script_path.chmod(0o755)
        except OSError:
            pass
        configs["start_sh"] = str(script_path)

    # k8s.yaml (server + hyperscale only)
    if scale.tier in ("server", "hyperscale", "multi_gpu"):
        k8s = generate_k8s_deployment(scale, rec, model_id)
        k8s_path = out_dir / "k8s.yaml"
        k8s_path.write_text(k8s, encoding="utf-8")
        configs["k8s_yaml"] = str(k8s_path)

    # systemd unit (non-Windows, non-Apple)
    import platform
    if platform.system() == "Linux":
        unit = generate_systemd_unit(rec, model_id)
        unit_path = out_dir / "llm.service"
        unit_path.write_text(unit, encoding="utf-8")
        configs["systemd_unit"] = str(unit_path)

    return configs
