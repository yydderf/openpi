#!/bin/bash
# Usage: run_policy_server.sh [pi05|pi0|pi0fast]   (default: pi05)

policy=${1:-pi05}
case "$policy" in
    pi05|pi0) name=${policy}_droid_jointpos_polaris ;;
    pi0fast) name=pi0_fast_droid_jointpos_polaris ;;
    *) echo "unknown policy '$policy', expected one of: pi05, pi0, pi0fast" >&2; exit 1 ;;
esac

PI_NVTX=1 NSYS_WARMUP=3 \
MODEL=$(nvidia-smi --query-gpu=name --format=csv,noheader | tr ' ' '-') \
POLICY=${policy}-droid DATETIME=$(date +"%Y%m%d_%H%M") \
    nsys profile \
    --output /app/outputs/nsys/%q{MODEL}_%q{POLICY}_%q{DATETIME}_%p --force-overwrite=true \
    --trace=cuda,nvtx,cublas,cudnn,osrt \
    --gpu-metrics-devices=cuda-visible \
    --gpu-metrics-set=ga10x \
    --gpu-metrics-frequency=10000 \
    --cuda-memory-usage=true \
    --cuda-graph-trace=node \
    --capture-range=cudaProfilerApi --capture-range-end=repeat:2 \
    --sample=process-tree --cpuctxsw=process-tree \
    uv run scripts/serve_policy.py policy:checkpoint \
        --policy.config=$name \
        --policy.dir=gs://openpi-assets/checkpoints/polaris/$name
