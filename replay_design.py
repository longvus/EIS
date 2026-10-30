"""Pool sizes, curve caps, hyperparameter dimensions and per-episode row and candidate draws."""

from __future__ import annotations

import operator

from types import MappingProxyType

import numpy as np

from benchmark_design import LCBENCH_TASKS, N_CONFIGS, PD1_CAPS, PD1_TASKS

TASKS = MappingProxyType({"lcbench": LCBENCH_TASKS, "pd1": PD1_TASKS})

POOL_SIZES = MappingProxyType({"lcbench": 2_000, "pd1": 242})

HP_DIMS = MappingProxyType({"lcbench": 7, "pd1": 4})

def _integer(value: object, label: str) -> int:
    if isinstance(value, bool):
        raise TypeError(f"{label} must be an integer")
    try:
        return operator.index(value)
    except TypeError as error:
        raise TypeError(f"{label} must be an integer") from error

def tasks_for(benchmark: str) -> tuple[str, ...]:
    try:
        return TASKS[benchmark]
    except KeyError as error:
        raise ValueError(f"unknown benchmark: {benchmark}") from error

def _validate_task(benchmark: str, task: str) -> None:
    if task not in tasks_for(benchmark):
        raise ValueError(f"unknown {benchmark} task: {task}")

def task_cap(benchmark: str, task: str) -> int:
    _validate_task(benchmark, task)
    return 50 if benchmark == "lcbench" else int(PD1_CAPS[task])

def episode_row_ids(benchmark: str, task: str, episode_seed: int) -> tuple[int, ...]:
    """Reproduce post-loader source rows without reading features or outcomes."""

    _validate_task(benchmark, task)
    seed = _integer(episode_seed, "episode_seed")
    if seed < 0:
        raise ValueError("episode_seed must be nonnegative")
    rng = np.random.default_rng(seed)
    sampled_task = str(rng.choice([task]))
    if sampled_task != task:
        raise RuntimeError("singleton task sampling semantics changed")
    rows = rng.choice(POOL_SIZES[benchmark], size=N_CONFIGS, replace=False)
    result = tuple(int(row) for row in np.asarray(rows, dtype=np.int64))
    if len(result) != N_CONFIGS or len(set(result)) != N_CONFIGS:
        raise RuntimeError("episode source rows are not unique")
    if any(row < 0 or row >= POOL_SIZES[benchmark] for row in result):
        raise RuntimeError("episode source row lies outside the frozen pool")
    return result

def candidate_id(benchmark: str, task: str, row_id: int) -> str:
    """Identify an arm only by benchmark, task, and post-loader source row."""

    _validate_task(benchmark, task)
    row = _integer(row_id, "row_id")
    if row < 0 or row >= POOL_SIZES[benchmark]:
        raise ValueError("row_id lies outside the frozen source pool")
    return f"{benchmark}/{task}/row/{row:020d}"

def episode_candidate_ids(
    benchmark: str,
    task: str,
    episode_seed: int,
) -> tuple[str, ...]:
    identifiers = tuple(
        candidate_id(benchmark, task, row)
        for row in episode_row_ids(benchmark, task, episode_seed)
    )
    if len(set(identifiers)) != N_CONFIGS:
        raise RuntimeError("episode candidate IDs are not unique")
    return identifiers
