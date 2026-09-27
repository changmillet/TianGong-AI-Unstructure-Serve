#!/usr/bin/env bash
set -euo pipefail

repo_root=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)
mode=${1:-parallel}
case "$mode" in
  parallel|parallel4|single) ;;
  *) echo "Usage: $0 [parallel|parallel4|single]" >&2; exit 2 ;;
esac

cd "$repo_root"
# Historical PM2 records can retain env values after a template removes them.
# The model port and memory budget come exclusively from this host's .env file.
unset MINERU_DOCKER_PORT MINERU_DOCKER_GPU_MEMORY MINERU_DOCKER_KV_CACHE_MEMORY_BYTES
unset MINERU_DOCKER_GPU_MEMORY_MODEL4 MINERU_DOCKER_KV_CACHE_MEMORY_BYTES_MODEL4
# Device nodes can appear after Docker/PM2 during boot. Never load host modules
# or restart the shared Docker daemon from this application launcher.
for ((attempt=0; attempt<60; attempt++)); do
  if [[ -c /dev/nvidia-uvm && -c /dev/nvidia-uvm-tools ]]; then
    break
  fi
  if ((attempt == 59)); then
    echo "CUDA UVM devices missing; check host NVIDIA driver initialization." >&2
    exit 1
  fi
  sleep 2
done
compose=(docker compose --env-file "$repo_root/.env" --project-name "mineru-vlm-$mode" --file "$repo_root/deploy/mineru-vllm/compose.mineru.yaml")
if [[ "$mode" == parallel ]]; then
  compose+=(--file "$repo_root/deploy/mineru-vllm/compose.mineru.parallel.yaml")
elif [[ "$mode" == parallel4 ]]; then
  compose+=(--file "$repo_root/deploy/mineru-vllm/compose.mineru.parallel4.yaml")
fi

# The final image may have been imported from another host. Reuse that exact
# configured tag rather than rebuilding it on every PM2 start or resurrection.
model_image=$("${compose[@]}" config --images)
if [[ -z "$model_image" || "$model_image" == *$'\n'* ]]; then
  echo "Expected exactly one configured MinerU model image." >&2
  exit 1
fi
if docker image inspect "$model_image" >/dev/null 2>&1; then
  echo "Using the configured local MinerU image without rebuild or pull."
  image_options=(--no-build --pull never)
else
  echo "Configured MinerU image is not local; building from Dockerfile."
  image_options=(--build)
fi

# Stay attached so PM2 stop/restart also stops/restarts the container.
exec "${compose[@]}" up "${image_options[@]}" --abort-on-container-exit --exit-code-from mineru-vlm mineru-vlm
