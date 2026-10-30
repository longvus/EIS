"""Canonical JSON encoding and SHA-256 helpers."""

from __future__ import annotations

import json

def canonical_json(payload: object) -> bytes:
    return json.dumps(
        payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True, allow_nan=False
    ).encode("ascii")
