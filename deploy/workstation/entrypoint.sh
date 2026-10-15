#!/usr/bin/env bash
# Docker entrypoint script.

set -e

NSYS_ROOT="${NSYS_ROOT:-/opt/nvidia/nsight-systems/2026.3.1}"

if [ -d "$NSYS_ROOT" ]; then
  if [ -x "$NSYS_ROOT/bin/nsys" ]; then
    export PATH="$NSYS_ROOT/bin:$PATH"
  else
    NSYS_TARGET="$(ls -d "$NSYS_ROOT"/target-linux-* 2>/dev/null | head -1)"
    [ -n "$NSYS_TARGET" ] && export PATH="$NSYS_TARGET:$PATH"
  fi
fi

NCU_ROOT="${NCU_ROOT:-/opt/nvidia/NVIDIA-Nsight-Compute-2026.3}"

if [ -d "$NCU_ROOT" ]; then
    if [ -x "$NCU_ROOT/ncu" ]; then
        export PATH="$NCU_ROOT:$PATH"
    fi
fi

exec "$@"


