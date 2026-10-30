"""
bench_io.py
─────────────────────────────────────────────────────────────────────────────
Project-level adapter between run scripts and real_task_eval.py.

All scripts in this project import benchmark classes and loaders from here,
not directly from real_task_eval.  If real_task_eval changes, only this file
needs updating.

Re-exports
----------
RealCurveTable          — learning-curve store (LCBench / NAS-Bench-201)
RealTaskHPOEnvironment  — HPO environment backed by real curves
load_ftpfn_surrogate    — raw loader → (pfn_wrapper, pfn_api)
D_HP, EVAL_SEED, N_CONFIGS  — shared constants

Convenience loaders
-------------------
load_lcbench(path, *, held_out_only, tasks) → RealCurveTable
load_pd1(path, *, task)                     → RealCurveTable  (one PD1 task)
pd1_task_names(path)                        → list[str]
load_pfn(table, device)                     → FTPFNSurrogate
load_lcbench_and_pfn(path, device, ...)     → (table, FTPFNSurrogate)

Benchmarks differ in fidelity unit: an LCBench tick is one epoch (a pass over the
data), a PD1 tick is 1000 optimizer steps. λ is the price of one tick either way.
"""

from __future__ import annotations

from hpo_bench import RealCurveTable
