"""The replay arrays built from the downloaded benchmarks equal the reference arrays.

Run `python build_shared_temporal_replay_arrays.py` first; this compares each built array with the SHA-256 recorded in
staging/shared_temporal_gp/v1/replay_arrays_expected.json."""
import hashlib
import json
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
STAGING = ROOT / "staging/shared_temporal_gp/v1"
EXPECTED = json.loads((STAGING / "replay_arrays_expected.json").read_text())


@pytest.mark.skipif(not (STAGING / "replay_arrays").exists(), reason="replay arrays not built")
@pytest.mark.parametrize("name", sorted(EXPECTED["arrays"]))
def test_built_array_matches_the_studies(name):
    assert hashlib.sha256((STAGING / "replay_arrays" / name).read_bytes()).hexdigest() == EXPECTED["arrays"][name]
