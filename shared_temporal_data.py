"""Bound replay inputs; score arrays are held by the evaluator, not the policy."""

from functools import lru_cache

import json

from types import SimpleNamespace

import numpy as np

from shared_temporal_design import STAGING, SharedTemporalSnapshot, snapshot, sha

@lru_cache(maxsize=16)
def model_snapshot(benchmark,mode):
    if benchmark!="odbench":
        return snapshot(benchmark,mode)
    payload=json.loads((STAGING/"odbench_geometry.json").read_bytes())
    return SharedTemporalSnapshot(tuple(payload["geometry"]["lengthscales"]),payload["encoder_id"],mode)

@lru_cache(maxsize=8)
def binding(benchmark,task,seed):
    if benchmark!="odbench":
        import pool_binding as runner
        from hpo_bench import RealCurveTable
        directory=STAGING/"replay_arrays"
        curves=np.load(directory/f"{benchmark}__{task}__curves.npy",mmap_mode="r",allow_pickle=False)
        hps=np.load(directory/f"{benchmark}__{task}__hps.npy",mmap_mode="r",allow_pickle=False)
        table=RealCurveTable(tasks={task:curves},hps={task:hps},pfn_total_epochs=curves.shape[1]+1)
        return runner.expected_pool_binding(table,benchmark=benchmark,task=task,episode_seed=seed)
    from cfbo_benchmark_io import CACHE_PATHS
    geometry=json.loads((STAGING/"odbench_geometry.json").read_bytes())
    rng=np.random.default_rng(seed)
    if str(rng.choice([task]))!=task:
        raise ValueError("task sampling mismatch")
    rows=tuple(int(r) for r in rng.choice(500,size=32,replace=False))
    directory=STAGING/"replay_arrays"
    hp=np.ascontiguousarray(np.load(directory/f"odbench__{task}__hps.npy",mmap_mode="r",allow_pickle=False)[list(rows)])
    curves=np.ascontiguousarray(np.load(directory/f"odbench__{task}__curves.npy",mmap_mode="r",allow_pickle=False)[list(rows)])
    if curves.shape!=(32,20) or curves.dtype!=np.float32 or hp.dtype!=np.float32:
        raise ValueError("ODBench pool dimensions/encoding differ")
    if not np.isfinite(curves).all() or np.any(curves<0) or np.any(curves>1):
        raise ValueError("invalid ODBench score support")
    ids=tuple(f"odbench/{task}/row/{r:020d}" for r in rows)
    pool_hash=sha(dict(ids=ids,rows=rows,hps=hp.tolist(),curves=curves.tolist()))
    return SimpleNamespace(row_ids=rows,candidate_ids=ids,hps=hp,curves=curves,pool_sha256=pool_hash,curve_len=20)
