"""
hp_encoders.py
─────────────────────────────────────────────────────────────────────────────
Per-benchmark hyperparameter encoders.

Each encoder takes a benchmark-specific configuration descriptor (architecture
string, channel widths, LCBench config dict, ...) and produces a fixed-length
float vector in [0, 1] of dimension ≤ 10. This is the format FT-PFN v0.0.1
expects: up to 10 dimensions, all values normalized.

The encoders here are deliberately simple. The goal is *not* to invent a new
representation — it is to stop discarding the HP information that FT-PFN's
attention mechanism is trained to use. A simple per-benchmark encoding is
strictly better than `torch.zeros(d)`, which is what the previous code passed.

Conventions
-----------
* All output vectors live in [0, 1].
* All encoders return np.ndarray with dtype float32.
* Padding to 10 dims happens at the call site (in `surrogate_api`), not here:
  encoders return their natural dimensionality and the wrapper pads as needed.

Public encoders
---------------
    encode_nb201(arch_strings)              -> (n, 6)
    encode_nats_tss(arch_strings)           -> (n, 6)
    encode_nats_sss(arch_strings)           -> (n, 5)
    encode_lcbench(config_dicts, key_order) -> (n, 7)
    encode_synthetic(n, d=3, seed=...)      -> (n, d)
"""

from __future__ import annotations

import numpy as np

from typing import Sequence

_NB201_OP_TO_INT: dict[str, int] = {
    "none":         0,
    "skip_connect": 1,
    "nor_conv_1x1": 2,
    "nor_conv_3x3": 3,
    "avg_pool_3x3": 4,
}

_NB201_N_OPS = 5

_NB201_N_EDGES = 6

def _parse_nb201_arch(arch_str: str) -> list[int]:
    """Parse a NAS-Bench-201 architecture string to 6 edge ops.

    Format: '|op1~0|+|op2~0|op3~1|+|op4~0|op5~1|op6~2|'
    Returns a list of 6 op indices ∈ [0, 4]; raises ValueError if malformed.
    """
    parts = [p for p in arch_str.split("+") if p.strip()]
    edges: list[int] = []
    for part in parts:
        items = [seg for seg in part.split("|") if seg.strip()]
        for seg in items:
            op_name = seg.split("~")[0]
            if op_name not in _NB201_OP_TO_INT:
                raise ValueError(f"Unknown NB201 op '{op_name}' in '{arch_str}'")
            edges.append(_NB201_OP_TO_INT[op_name])
    if len(edges) != _NB201_N_EDGES:
        raise ValueError(
            f"Expected {_NB201_N_EDGES} edges, got {len(edges)} in '{arch_str}'"
        )
    return edges

def encode_nb201(arch_strings: Sequence[str]) -> np.ndarray:
    """Encode NAS-Bench-201 architectures.

    Returns (n, 6) float32 in [0, 1]. Each edge op is encoded as
    op_index / (n_ops - 1) — a compact ordinal encoding that preserves the
    discrete structure while staying within FT-PFN's [0, 1] input range.

    A one-hot of (n, 30) would also work but consumes more of FT-PFN's 10-dim
    budget; we keep 6 dims so there's headroom for combination encodings later.
    """
    n = len(arch_strings)
    out = np.zeros((n, _NB201_N_EDGES), dtype=np.float32)
    denom = float(_NB201_N_OPS - 1)
    for i, s in enumerate(arch_strings):
        edges = _parse_nb201_arch(s)
        for j, op in enumerate(edges):
            out[i, j] = op / denom
    return out

encode_nats_tss = encode_nb201

_NATS_SSS_CHANNELS = (8, 16, 24, 32, 40, 48, 56, 64)

_NATS_SSS_N_LAYERS = 5

def _parse_nats_sss_arch(arch_str: str) -> list[int]:
    """Parse '8:16:32:48:64' (or similar) into 5 channel widths."""
    parts = arch_str.strip().split(":")
    if len(parts) != _NATS_SSS_N_LAYERS:
        raise ValueError(
            f"Expected {_NATS_SSS_N_LAYERS} channels, got {len(parts)} in '{arch_str}'"
        )
    return [int(p) for p in parts]

def encode_nats_sss(arch_strings: Sequence[str]) -> np.ndarray:
    """Encode NATS-Bench SSS architectures.

    Returns (n, 5) float32 in [0, 1]. Each channel width is encoded as
    (channel - min_channel) / (max_channel - min_channel) — a continuous
    normalization that respects the natural ordering of widths.
    """
    n = len(arch_strings)
    out = np.zeros((n, _NATS_SSS_N_LAYERS), dtype=np.float32)
    c_min = float(_NATS_SSS_CHANNELS[0])
    c_max = float(_NATS_SSS_CHANNELS[-1])
    span = c_max - c_min
    for i, s in enumerate(arch_strings):
        widths = _parse_nats_sss_arch(s)
        for j, w in enumerate(widths):
            out[i, j] = (float(w) - c_min) / span
    return out

LCBENCH_HP_KEYS = (
    "batch_size",
    "learning_rate",
    "max_dropout",
    "max_units",
    "momentum",
    "num_layers",
    "weight_decay",
)

_LCBENCH_RANGES: dict[str, tuple[float, float, str]] = {
    # key: (min, max, scale)  scale ∈ {'linear', 'log10'}
    "batch_size":      (16.0,    512.0,    "log10"),
    "learning_rate":   (1e-4,    1e-1,     "log10"),
    "max_dropout":     (0.0,     1.0,      "linear"),
    "max_units":       (64.0,    1024.0,   "log10"),
    "momentum":        (0.1,     0.99,     "linear"),
    "num_layers":      (1.0,     5.0,      "linear"),
    "weight_decay":    (1e-5,    1e-1,     "log10"),
}

def encode_lcbench(
    config_dicts: Sequence[dict],
    key_order: Sequence[str] = LCBENCH_HP_KEYS,
) -> np.ndarray:
    """Encode LCBench configurations from per-run config dicts.

    Each dict must contain at least the keys in `key_order`. Returns (n, 7)
    float32 in [0, 1]. Log-scale keys are first log10-transformed, then
    affine-mapped to [0, 1] using the documented LCBench ranges. Values
    outside the documented range are clipped (rare; they reflect a different
    LCBench version, not a bug).
    """
    n = len(config_dicts)
    d = len(key_order)
    out = np.zeros((n, d), dtype=np.float32)
    for i, cfg in enumerate(config_dicts):
        for j, key in enumerate(key_order):
            if key not in _LCBENCH_RANGES:
                raise ValueError(f"Unknown LCBench HP key: {key}")
            lo, hi, scale = _LCBENCH_RANGES[key]
            v = float(cfg[key])
            if scale == "log10":
                v = np.log10(max(v, 1e-30))
                lo = np.log10(lo)
                hi = np.log10(hi)
            out[i, j] = np.clip((v - lo) / (hi - lo), 0.0, 1.0)
    return out
