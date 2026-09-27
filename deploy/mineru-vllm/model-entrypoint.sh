#!/usr/bin/env bash
set -euo pipefail

memory_args=(--gpu-memory-utilization "${MINERU_DOCKER_GPU_MEMORY:-0.10}")

if [[ -n "${MINERU_DOCKER_KV_CACHE_MEMORY_BYTES:-}" ]]; then
  if [[ ! "$MINERU_DOCKER_KV_CACHE_MEMORY_BYTES" =~ ^[1-9][0-9]*$ ]]; then
    echo "MINERU_DOCKER_KV_CACHE_MEMORY_BYTES must be a positive integer number of bytes." >&2
    exit 2
  fi
  memory_args+=(--kv-cache-memory-bytes "$MINERU_DOCKER_KV_CACHE_MEMORY_BYTES")
fi

exec mineru-kit vlm-server --engine vllm "$@" "${memory_args[@]}"
