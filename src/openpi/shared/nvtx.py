"""Opt-in NVTX phase ranges for Nsight Systems profiling. No-op unless PI_NVTX=1.

PI0Pytorch.sample_actions is wrapped in torch.compile by default (pytorch_compile_mode="max-autotune").
Ranges placed inside it are dropped while Dynamo traces (see `nvtx_phase`), so with compile on only the
eager call-site ranges in Policy.infer show up. Set `pytorch_compile_mode=None` to see the in-model phases.
"""

import contextlib
import functools
import os

import torch

# Resolved once at import: this is a profiling-session switch, not something to
# re-read per range in a hot loop.
_ENABLED: bool = os.environ.get("PI_NVTX", "0").strip().lower() not in ("0", "", "false", "no")

# nullcontext holds no state, so one shared instance is safe across nested and
# repeated `with` blocks. Keeps the disabled path allocation-free.
_NULL: contextlib.nullcontext = contextlib.nullcontext()

_CAPTURE_REQUESTS = 2  # must match --capture-range-end=repeat:N


def nvtx_enabled() -> bool:
    return _ENABLED


def nvtx_phase(name: str):
    """Context manager marking an NVTX range. No-op when disabled.

    Also a no-op while torch.compile traces: Dynamo folds `is_compiling()` to a constant, so the
    range vanishes from the compiled graph instead of causing a graph break (which would split the
    CUDA graph and silently slow the model down).

    Never calls torch.cuda.synchronize(): synchronizing on range boundaries
    serializes CPU and GPU and manufactures exactly the idle gaps you are hunting.
    """
    if not _ENABLED or torch.compiler.is_compiling():
        return _NULL
    return torch.cuda.nvtx.range(name)


def nsys_profile(func):
    """Open a cudaProfilerApi capture range around the first WARM calls only.

    Returns `func` unchanged when PI_NVTX is off, so the serving path pays nothing.
    """
    if not _ENABLED:
        return func

    warmup = int(os.environ.get("NSYS_WARMUP", "10"))
    state = {"count": 0}

    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        index = state["count"]
        state["count"] = index + 1
        if not warmup <= index < warmup + _CAPTURE_REQUESTS:
            return func(*args, **kwargs)  # no sync, no profiler calls
        torch.cuda.synchronize()
        torch.cuda.profiler.start()
        try:
            return func(*args, **kwargs)
        finally:
            torch.cuda.profiler.stop()
            torch.cuda.synchronize()

    return wrapper
