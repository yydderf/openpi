#!/bin/bash
# Interactive openpi container for workstations, mirroring scripts/docker/compose.yml
# plus Nsight Systems profiling (see deploy/workstation/entrypoint.sh).
# Build the image first: docker compose -f scripts/docker/compose.yml build

REPO_ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
NSYS_ROOT=/opt/nvidia/nsight-systems/2026.4.1
OPENPI_DATA_HOME="${OPENPI_DATA_HOME:-$HOME/.cache/openpi}"
mkdir -p "$OPENPI_DATA_HOME"

docker run -it --rm --init \
    --cap-add=SYS_ADMIN --security-opt seccomp=unconfined \
    --name openpi \
    --net host --gpus all \
    --ulimit memlock=-1 --ulimit stack=67108864 \
    -e HF_TOKEN=$HF_TOKEN \
    -e SERVER_ARGS \
    -e OPENPI_DATA_HOME=/openpi_assets \
    -e IS_DOCKER=true \
    -e NSYS_ROOT=$NSYS_ROOT \
    -v $NSYS_ROOT:$NSYS_ROOT:ro \
    -v "$REPO_ROOT:/app" \
    -v "$OPENPI_DATA_HOME:/openpi_assets" \
    -v "/srv/models/huggingface:/root/.cache/huggingface" \
    --entrypoint /app/deploy/workstation/entrypoint.sh \
    openpi_server \
    /bin/bash
