#!/bin/bash
# Usage: run_policy_server.sh [pi05|pi0|pi0fast] [jax|pytorch] [server|nsys|ncu]   (default: pi05 jax server)
#   jax:     gs://openpi-assets/checkpoints/polaris/<config>
#   pytorch: $OPENPI_DATA_HOME/openpi-assets/checkpoints/polaris/<config>_pytorch
#            (made by examples/convert_jax_model_to_pytorch.py; pi0/pi05 only)
#   server:  just serve the policy.
#   nsys:    wraps the server in Nsight Systems and captures 2 requests after NSYS_WARMUP warm-up requests.
#   ncu:     wraps the server in Nsight Compute and profiles kernels from the same 2 requests. Every kernel is
#            replayed once per metric pass, so a captured request takes minutes; clients may need a longer
#            timeout. Tune with NCU_SET (default detailed; see `ncu --list-sets`), NCU_LAUNCH_COUNT
#            (max kernels, default 50), NCU_LAUNCH_SKIP (default 0) and NCU_KERNEL_NAME (e.g. regex:gemm).
#            Runs the venv's python directly rather than through `uv run` (see below).
#   e.g. 
#   - NCU_SET=full NCU_LAUNCH_COUNT=50 NCU_KERNEL_NAME=regex:gemm deploy/workstation/run_policy_server.sh pi05 jax ncu
#   - deploy/workstation/run_policy_server.sh pi05 pytorch ncu

set -euo pipefail

policy=${1:-pi05}
checkpoint_type=${2:-jax}
mode=${3:-server}

case "$policy" in
    pi05|pi0) config=${policy}_droid_jointpos_polaris ;;
    pi0fast) config=pi0_fast_droid_jointpos_polaris ;;
    *) echo "unknown policy '$policy', expected one of: pi05, pi0, pi0fast" >&2; exit 1 ;;
esac

case "$checkpoint_type" in
    jax) checkpoint_dir=gs://openpi-assets/checkpoints/polaris/$config ;;
    pytorch)
        [ "$policy" = pi0fast ] && { echo "pi0fast has no PyTorch implementation" >&2; exit 1; }
        checkpoint_dir=${OPENPI_DATA_HOME:-/openpi_assets}/openpi-assets/checkpoints/polaris/${config}_pytorch
        [ -f "$checkpoint_dir/model.safetensors" ] || { echo "no converted checkpoint at $checkpoint_dir" >&2; exit 1; }
        ;;
    *) echo "unknown checkpoint type '$checkpoint_type', expected one of: jax, pytorch" >&2; exit 1 ;;
esac

server=(
    uv run --group profiling scripts/serve_policy.py policy:checkpoint
        --policy.config="$config"
        --policy.dir="$checkpoint_dir"
)

case "$mode" in
    server) exec "${server[@]}" ;;
    nsys|ncu) ;;
    *) echo "unknown mode '$mode', expected one of: server, nsys, ncu" >&2; exit 1 ;;
esac

# nvidia-smi reports no GPU name on Jetson; fall back to the board name.
gpu=$(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null) && gpu=${gpu%%$'\n'*} || gpu=
[ -n "$gpu" ] || gpu=$(tr -d '\0' </proc/device-tree/model 2>/dev/null || echo unknown-gpu)

export PI_NVTX=1 NSYS_WARMUP=${NSYS_WARMUP:-3} NSYS_NVTX_PROFILER_REGISTER_ONLY=0
export MODEL=${gpu// /-} POLICY=${policy}-droid-${checkpoint_type} DATETIME=$(date +"%Y%m%d_%H%M")

if [ "$mode" = ncu ]; then
    # Under ncu, `uv run` hangs the server at "Loading model..." (no GPU context, ncu never connects). Likely
    # cause: python inherits ncu's injection env through uv, but ncu does not track uv's spawn, so the injection
    # handshake in cuInit never completes. Sync the env the way `uv run` would, then launch python directly.
    uv sync --inexact --group profiling
    server=("${UV_PROJECT_ENVIRONMENT:-.venv}/bin/python" "${server[@]:4}")

    mkdir -p /app/outputs/ncu
    # The capture range is a push/pop range in the default domain, hence the trailing '/'. Kernels outside it
    # (warm-up, JIT/autotuning) run unprofiled.
    exec ncu \
        -o "/app/outputs/ncu/${MODEL}_${POLICY}_${DATETIME}_%p" --force-overwrite \
        --nvtx --nvtx-include "openpi.capture/" \
        --set "${NCU_SET:-detailed}" \
        --launch-skip "${NCU_LAUNCH_SKIP:-0}" --launch-count "${NCU_LAUNCH_COUNT:-50}" \
        ${NCU_KERNEL_NAME:+--kernel-name "$NCU_KERNEL_NAME"} \
        --graph-profiling node \
        "${server[@]}"
fi

mkdir -p /app/outputs/nsys

exec nsys profile \
    --output /app/outputs/nsys/%q{MODEL}_%q{POLICY}_%q{DATETIME}_%p --force-overwrite=true \
    --trace=cuda,nvtx,cublas,cudnn,osrt \
    --gpu-metrics-devices=cuda-visible \
    --gpu-metrics-set=ga10x \
    --gpu-metrics-frequency=10000 \
    --cuda-memory-usage=true \
    --cuda-graph-trace=node \
    --capture-range=nvtx --nvtx-capture=openpi.capture --capture-range-end=repeat:2 \
    --sample=process-tree --cpuctxsw=process-tree \
    "${server[@]}"
