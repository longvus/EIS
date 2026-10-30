"""run_episode.py reproduces stored episodes: EI stopping under all three surrogates and H2 stopping episodes in
which the gate opened, on LCBench, PD1 and ODBench. Needs the benchmark data under datasets/ (see README)."""
import json
from pathlib import Path

import pytest

import run_episode

ROOT = Path(__file__).resolve().parents[1]
FIXTURES = json.loads((Path(__file__).parent / "episode_fixtures.json").read_text())
pytestmark = pytest.mark.skipif(not (ROOT / "datasets").exists(), reason="benchmark data absent")


@pytest.mark.parametrize("case", FIXTURES, ids=lambda c: f"{c['surrogate']}-{c['stopping']}-{c['benchmark']}-{c['cost']}")
def test_episode_matches_stored_record(case):
    got = run_episode.episode(case["benchmark"], case["task"], case["seed"], case["cost"], case["surrogate"], case["stopping"])
    assert got["increments"] == case["increments"]
    if "actions" in case:
        assert got["actions"] == case["actions"]
    assert got["utility"] == pytest.approx(case["utility"], abs=1e-12)
    assert got["best_score"] == pytest.approx(case["best_score"], abs=1e-12)
