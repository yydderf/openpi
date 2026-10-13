#!/bin/bash
# Usage: run_policy_server.sh [pi05|pi0|pi0fast] [jax|pytorch] [serve|profile]   (default: pi05 jax serve)
#   jax:     gs://openpi-assets/checkpoints/polaris/<config>
#   pytorch: $OPENPI_DATA_HOME/openpi-assets/checkpoints/polaris/<config>_pytorch
#            (made by examples/convert_jax_model_to_pytorch.py; pi0/pi05 only)
#   profile: wraps the server in nsys and captures 2 requests after NSYS_WARMUP warm-up requests.

set -euo pipefail

policy=${1:-pi05}
checkpoint_type=${2:-jax}
mode=${3:-serve}

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
    serve) exec "${server[@]}" ;;
    profile) ;;
    *) echo "unknown mode '$mode', expected one of: serve, profile" >&2; exit 1 ;;
esac

# nvidia-smi reports no GPU name on Jetson; fall back to the board name.
gpu=$(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null) && gpu=${gpu%%$'\n'*} || gpu=
[ -n "$gpu" ] || gpu=$(tr -d '\0' </proc/device-tree/model 2>/dev/null || echo unknown-gpu)

export PI_NVTX=1 NSYS_WARMUP=${NSYS_WARMUP:-3} NSYS_NVTX_PROFILER_REGISTER_ONLY=0
export MODEL=${gpu// /-} POLICY=${policy}-droid-${checkpoint_type} DATETIME=$(date +"%Y%m%d_%H%M")
mkdir -p /app/outputs/nsys

exec nsys profile \
    --output /app/outputs/nsys/%q{MODEL}_%q{POLICY}_%q{DATETIME}_%p --force-overwrite=true \
    --trace=cuda,nvtx,cublas,cudnn,osrt \
    --gpu-metrics-devices=cuda-visible \
    --gpu-metrics-set=ga10y \
    --gpu-metrics-frequency=2000 \
    --cuda-memory-usage=true \
    --cuda-graph-trace=node \
    --capture-range=nvtx --nvtx-capture=openpi.capture --capture-range-end=repeat:2 \
    --sample=process-tree --cpuctxsw=process-tree \
    "${server[@]}"
