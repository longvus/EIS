"""Deterministic pool binding: the configurations and curves of each task, seed and pool."""

from __future__ import annotations

from dataclasses import dataclass

import hashlib

import numpy as np

from canonical_artifacts import canonical_json

from bench_io import RealCurveTable

import replay_design as design

class InitialPoolBindingError(RuntimeError):
    """The environment's first sampled pool differs from its recorded identity."""

@dataclass(frozen=True)
class PoolBinding:
    row_ids: tuple[int, ...]
    candidate_ids: tuple[str, ...]
    row_ids_sha256: str
    candidate_ids_sha256: str
    hp_pool_sha256: str
    curve_pool_sha256: str
    pool_sha256: str
    pool_size: int
    curve_len: int
    hp_dim: int
    hps: np.ndarray
    curves: np.ndarray

def _array_sha256(value: object, dtype: object) -> str:
    canonical_dtype = np.dtype(dtype).newbyteorder("<")
    array = np.ascontiguousarray(np.asarray(value, dtype=canonical_dtype))
    digest = hashlib.sha256()
    digest.update(np.asarray(array.shape, dtype="<i8").tobytes())
    digest.update(array.tobytes(order="C"))
    return digest.hexdigest()

def _value_sha256(value: object) -> str:
    return hashlib.sha256(canonical_json(value)).hexdigest()

def _pool_sha256(
    row_hash: str,
    candidate_hash: str,
    hp_hash: str,
    curve_hash: str,
) -> str:
    return _value_sha256({
        "row_ids_sha256": row_hash,
        "candidate_ids_sha256": candidate_hash,
        "hp_pool_sha256": hp_hash,
        "curve_pool_sha256": curve_hash,
    })

def _require_binding(condition: bool, message: str) -> None:
    if not condition:
        raise InitialPoolBindingError(message)

def expected_pool_binding(
    table: RealCurveTable,
    *,
    benchmark: str,
    task: str,
    episode_seed: int,
) -> PoolBinding:
    """Bind one row roster to its exact post-loader HP and curve arrays."""

    rows = design.episode_row_ids(benchmark, task, episode_seed)
    identifiers = design.episode_candidate_ids(benchmark, task, episode_seed)
    cap = design.task_cap(benchmark, task)
    pool_size = design.POOL_SIZES[benchmark]
    hp_dim = design.HP_DIMS[benchmark]
    _require_binding(table.task_names == [task], "runner table is not exactly one task")
    _require_binding(set(table.tasks) == {task}, "curve-table task keys differ")
    _require_binding(table.hps is not None, "production table has no HP rows")
    _require_binding(set(table.hps) == {task}, "HP-table task keys differ")

    source_curves = np.asarray(table.tasks[task])
    source_hps = np.asarray(table.hps[task])
    _require_binding(
        source_curves.shape == (pool_size, cap),
        "source curve pool size or dimensions differ",
    )
    _require_binding(
        source_hps.shape == (pool_size, hp_dim),
        "source HP pool size or dimensions differ",
    )
    _require_binding(source_curves.dtype == np.dtype(np.float32), "curves are not float32")
    _require_binding(source_hps.dtype == np.dtype(np.float32), "HP rows are not float32")
    indices = np.asarray(rows, dtype=np.int64)
    expected_curves = np.ascontiguousarray(source_curves[indices])
    expected_hps = np.ascontiguousarray(source_hps[indices])
    _require_binding(
        np.isfinite(expected_curves).all()
        and np.isfinite(expected_hps).all()
        and np.all((expected_curves >= 0.0) & (expected_curves <= 1.0))
        and np.all((expected_hps >= 0.0) & (expected_hps <= 1.0)),
        "bound source rows are nonfinite or outside [0,1]",
    )
    row_hash = _array_sha256(rows, np.int64)
    candidate_hash = _value_sha256(list(identifiers))
    hp_hash = _array_sha256(expected_hps, np.float32)
    curve_hash = _array_sha256(expected_curves, np.float32)
    return PoolBinding(
        row_ids=rows,
        candidate_ids=identifiers,
        row_ids_sha256=row_hash,
        candidate_ids_sha256=candidate_hash,
        hp_pool_sha256=hp_hash,
        curve_pool_sha256=curve_hash,
        pool_sha256=_pool_sha256(row_hash, candidate_hash, hp_hash, curve_hash),
        pool_size=pool_size,
        curve_len=cap,
        hp_dim=hp_dim,
        hps=expected_hps,
        curves=expected_curves,
    )
