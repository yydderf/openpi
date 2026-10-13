"""Opt-in NVTX phase ranges for Nsight Systems profiling. No-op unless PI_NVTX=1.

Ranges go through the standalone `nvtx` package (`--group profiling`), which needs neither a CUDA build of
torch nor a GPU, so the same code tags phases on the workstation (CUDA torch) and on Jetson (CPU-only
torch wheel, model running in JAX). Without that package it falls back to `torch.cuda.nvtx` when torch was
built with CUDA, and otherwise disables itself with a warning instead of raising mid-inference.

PI0Pytorch.sample_actions is wrapped in torch.compile by default (pytorch_compile_mode="max-autotune").
Ranges placed inside it are dropped while Dynamo traces (see `nvtx_phase`), so with compile on only the
eager call-site ranges in Policy.infer show up. Set `pytorch_compile_mode=None` to see the in-model phases.
"""

import contextlib
import functools
import logging
import os

import torch

logger = logging.getLogger(__name__)

# Resolved once at import: this is a profiling-session switch, not something to
# re-read per range in a hot loop.
_ENABLED: bool = os.environ.get("PI_NVTX", "0").strip().lower() not in ("0", "", "false", "no")

# nullcontext holds no state, so one shared instance is safe across nested and
# repeated `with` blocks. Keeps the disabled path allocation-free.
_NULL: contextlib.nullcontext = contextlib.nullcontext()

_CAPTURE_REQUESTS = 2  # must match --capture-range-end=repeat:N
# Must match --nvtx-capture. An NVTX capture range needs no CUDA runtime call, unlike
# cudaProfilerStart, which a CPU-only torch build cannot issue.
CAPTURE_RANGE = "openpi.capture"


def _resolve_range():
    try:
        import nvtx

        return nvtx.annotate
    except ImportError:
        pass
    if torch.backends.cuda.is_built():
        return torch.cuda.nvtx.range
    logger.warning(
        "PI_NVTX=1 but no NVTX backend: install the `profiling` group (uv sync --group profiling). "
        "NVTX ranges are disabled."
    )
    return None


_range = _resolve_range() if _ENABLED else None
_ENABLED = _range is not None


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
    return _range(name)


def _sync_torch_cuda() -> None:
    # Drains in-flight torch work so the capture starts and ends on idle-GPU boundaries. Skipped on
    # CPU-only torch; the JAX path needs nothing here since Policy.infer already returns host arrays.
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def nsys_profile(func):
    """Wrap the first WARM calls in the `CAPTURE_RANGE` NVTX range that nsys captures on.

    Run nsys with `--capture-range=nvtx --nvtx-capture=openpi.capture` and
    `NSYS_NVTX_PROFILER_REGISTER_ONLY=0` (the range name is not a registered string).

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
            return func(*args, **kwargs)  # no sync, no extra range
        _sync_torch_cuda()
        with _range(CAPTURE_RANGE):
            try:
                return func(*args, **kwargs)
            finally:
                _sync_torch_cuda()  # inside the range, so trailing GPU work is captured

    return wrapper
