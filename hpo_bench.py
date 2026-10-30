"""
hpo_bench.py
─────────────────────────────────────────────────────────────────────────────
Core benchmark classes extracted from real_task_eval.py.

Contains RealCurveTable, RealTaskHPOEnvironment, load_ftpfn_surrogate,
evaluate_with_ceiling, _baselines_run_key, and shared constants.
No dependency on real_task_eval — safe to import standalone.

real_task_eval.py imports from here for backward compatibility.
bench_io.py imports from here as the project-level adapter.
"""

from __future__ import annotations

import bz2

import copy

import json

import pickle

import warnings

from pathlib import Path

from typing import Optional

import numpy as np

from bucket_constants import LCBENCH_HELD_OUT_TASKS

class RealCurveTable:
    """
    Precomputed learning curve store, grouped by task.

    Attributes
    ----------
    tasks : dict[str, np.ndarray]
        {task_name: ndarray(n_configs, curve_len)}  values in [0, 1]
    hps : dict[str, np.ndarray] | None
        {task_name: ndarray(n_configs, d_hp)}  values in [0, 1].
        Per-config hyperparameter vectors aligned with `tasks` along axis 0.
        When None, callers fall back to zero HP vectors (legacy behavior;
        this is what `torch.zeros(3)` was doing before, and is now the
        explicit "no HP info" path).
    pfn_total_epochs : int
        Epoch count used to normalise t for FT-PFN queries.
        May differ from curve_len (e.g. LCBench: curve_len=51, pfn_total=52).
    task_names : list[str]
        Names of available eval tasks.
    """

    def __init__(
        self,
        tasks:            dict[str, np.ndarray],
        pfn_total_epochs: int,
        hps:              dict[str, np.ndarray] | None = None,
    ):
        if not tasks:
            raise ValueError("RealCurveTable requires at least one task")

        # Enforce the invariant that all tasks share the same curve length.
        # Mixed-length tasks (e.g. synthetic with multiple horizons) are
        # right-padded by repeating the final value — the natural
        # extrapolation for a learning curve that has stopped training.
        lengths = {name: arr.shape[1] for name, arr in tasks.items()}
        max_len = max(lengths.values())
        if len(set(lengths.values())) > 1:
            padded: dict[str, np.ndarray] = {}
            for name, arr in tasks.items():
                if arr.shape[1] < max_len:
                    pad_width = max_len - arr.shape[1]
                    last_col = arr[:, -1:]
                    arr = np.concatenate(
                        [arr, np.repeat(last_col, pad_width, axis=1)],
                        axis=1,
                    )
                padded[name] = arr
            tasks = padded

        # Enforce FT-PFN input contract: learning-curve values must be finite
        # and normalized to [0, 1].  A few benchmark dumps store accuracies as
        # percentages; when the range is clearly [0, 100], normalize here rather
        # than feeding invalid values into the surrogate.
        normalized_tasks: dict[str, np.ndarray] = {}
        for name, arr in tasks.items():
            arr = np.asarray(arr, dtype=np.float32)
            if arr.ndim != 2:
                raise ValueError(f"tasks['{name}'] must be 2-D, got shape {arr.shape}")
            if not np.all(np.isfinite(arr)):
                raise ValueError(f"tasks['{name}'] contains NaN/Inf values")
            amin = float(arr.min())
            amax = float(arr.max())
            if amin < -1e-6:
                raise ValueError(f"tasks['{name}'] has values below 0: min={amin}")
            if amax > 1.0 + 1e-6:
                if amax <= 100.0 + 1e-6:
                    warnings.warn(
                        f"tasks['{name}'] appears to contain percentage accuracies; "
                        "dividing by 100 to satisfy FT-PFN's [0,1] contract.",
                        RuntimeWarning,
                    )
                    arr = arr / 100.0
                else:
                    raise ValueError(
                        f"tasks['{name}'] has values above 1 after normalization check: max={amax}"
                    )
            normalized_tasks[name] = np.clip(arr, 0.0, 1.0).astype(np.float32, copy=False)
        tasks = normalized_tasks

        # Validate that hps (if provided) aligns with tasks and obeys FT-PFN's
        # normalized hyperparameter contract.
        if hps is not None:
            normalized_hps: dict[str, np.ndarray] = {}
            for name, curves in tasks.items():
                if name not in hps:
                    raise ValueError(
                        f"hps missing entry for task '{name}'; either provide "
                        f"hps for every task or pass hps=None."
                    )
                hp_arr = np.asarray(hps[name], dtype=np.float32)
                if hp_arr.shape[0] != curves.shape[0]:
                    raise ValueError(
                        f"hps['{name}'] has {hp_arr.shape[0]} rows but "
                        f"tasks['{name}'] has {curves.shape[0]} configs."
                    )
                if not np.all(np.isfinite(hp_arr)):
                    raise ValueError(f"hps['{name}'] contains NaN/Inf values")
                if hp_arr.size and (float(hp_arr.min()) < -1e-6 or float(hp_arr.max()) > 1.0 + 1e-6):
                    raise ValueError(
                        f"hps['{name}'] must be normalized to [0,1]; "
                        f"range=({float(hp_arr.min())}, {float(hp_arr.max())})"
                    )
                normalized_hps[name] = np.clip(hp_arr, 0.0, 1.0).astype(np.float32, copy=False)
            hps = normalized_hps

        self.tasks            = tasks
        self.hps              = hps
        self.pfn_total_epochs = pfn_total_epochs
        self.task_names       = list(tasks.keys())

    @property
    def curve_len(self) -> int:
        """Length of each stored curve (number of usable training epochs)."""
        return next(iter(self.tasks.values())).shape[1]

    @property
    def hp_dim(self) -> int | None:
        """Dimension of HP vectors, or None if this table was built without HPs."""
        if self.hps is None:
            return None
        return next(iter(self.hps.values())).shape[1]

    def sample_episode_configs(
        self,
        n_configs: int,
        rng:       np.random.Generator,
    ) -> tuple[str, np.ndarray, np.ndarray | None]:
        """
        Sample a random task and n_configs random configs.

        Returns
        -------
        task_name : str
        curves    : np.ndarray (n_configs, curve_len)
        hps       : np.ndarray (n_configs, d_hp) or None
                    None iff this table was built without HPs (legacy path).
        """
        task_name  = str(rng.choice(self.task_names))
        pool       = self.tasks[task_name]          # (n_pool, curve_len)
        n_pool     = len(pool)
        replace    = n_pool < n_configs
        idx        = rng.choice(n_pool, size=n_configs, replace=replace)
        curves = pool[idx]
        hps = self.hps[task_name][idx] if self.hps is not None else None
        return task_name, curves, hps

    def split(
        self,
        train_fraction: float = 0.6,
        valid_fraction: float = 0.2,
        seed: int = 12345,
    ) -> tuple["RealCurveTable", "RealCurveTable", "RealCurveTable"]:
        """
        Deterministically split each task's configs into train/valid/test pools.

        Splitting per task preserves the task mix while preventing the scheduler
        from training and testing on the same config rows. HP vectors (if
        present) follow the same row indices as their curves.
        """
        if not (0.0 < train_fraction < 1.0):
            raise ValueError("train_fraction must be in (0, 1)")
        if not (0.0 <= valid_fraction < 1.0):
            raise ValueError("valid_fraction must be in [0, 1)")
        if train_fraction + valid_fraction >= 1.0:
            raise ValueError("train_fraction + valid_fraction must be < 1")

        train_tasks: dict[str, np.ndarray] = {}
        valid_tasks: dict[str, np.ndarray] = {}
        test_tasks: dict[str, np.ndarray] = {}
        train_hps: dict[str, np.ndarray] | None = {} if self.hps is not None else None
        valid_hps: dict[str, np.ndarray] | None = {} if self.hps is not None else None
        test_hps: dict[str, np.ndarray] | None = {} if self.hps is not None else None

        for task_offset, task_name in enumerate(sorted(self.task_names)):
            curves = self.tasks[task_name]
            rng = np.random.default_rng(seed + task_offset)
            perm = rng.permutation(len(curves))
            n_train = int(np.floor(len(curves) * train_fraction))
            n_valid = int(np.floor(len(curves) * valid_fraction))
            train_idx = perm[:n_train]
            valid_idx = perm[n_train:n_train + n_valid]
            test_idx = perm[n_train + n_valid:]
            train_tasks[task_name] = curves[train_idx]
            valid_tasks[task_name] = curves[valid_idx]
            test_tasks[task_name] = curves[test_idx]
            if self.hps is not None:
                hp_arr = self.hps[task_name]
                train_hps[task_name] = hp_arr[train_idx]
                valid_hps[task_name] = hp_arr[valid_idx]
                test_hps[task_name]  = hp_arr[test_idx]

        return (
            RealCurveTable(train_tasks, self.pfn_total_epochs, hps=train_hps),
            RealCurveTable(valid_tasks, self.pfn_total_epochs, hps=valid_hps),
            RealCurveTable(test_tasks, self.pfn_total_epochs, hps=test_hps),
        )

    def select_split(
        self,
        part: str,
        train_fraction: float = 0.6,
        valid_fraction: float = 0.2,
        seed: int = 12345,
    ) -> "RealCurveTable":
        if part == "all":
            return self
        train, valid, test = self.split(train_fraction, valid_fraction, seed)
        return {"train": train, "valid": valid, "test": test}[part]

    def select_tasks(self, task_names: list[str] | tuple[str, ...]) -> "RealCurveTable":
        selected = [str(name) for name in task_names]
        missing = [name for name in selected if name not in self.tasks]
        if missing:
            raise ValueError(
                f"unknown task(s) {missing}; available tasks are {sorted(self.tasks)}"
            )
        tasks = {name: self.tasks[name] for name in selected}
        hps = None if self.hps is None else {name: self.hps[name] for name in selected}
        return RealCurveTable(tasks=tasks, pfn_total_epochs=self.pfn_total_epochs, hps=hps)

    @classmethod
    def from_nasbench201(
        cls,
        cifar10_cifar100_path: Path,
        imagenet16_path: Optional[Path] = None,
        arch_strings_path: Optional[Path] = None,
    ) -> "RealCurveTable":
        """
        Load NAS-Bench-201 as three independent tasks: cifar10, cifar100, imagenet16.

        cifar10_cifar100_path : .npy file of shape (31250, 200) — first 15625
            rows are CIFAR-10, last 15625 are CIFAR-100.
        imagenet16_path : optional .npy of shape (15625, 200) for ImageNet16-120.
        arch_strings_path : optional .txt file with one NB201 arch string per
            line (15625 lines), aligned with the curve rows. When provided,
            per-config HP vectors are encoded via `encode_nb201` and stored
            on the table so FT-PFN sees real architecture geometry instead of
            zero HP vectors.

        pfn_total_epochs is set to 200 (full NAS-Bench-201 training length).
        """
        from hp_encoders import encode_nb201

        combined = np.load(str(cifar10_cifar100_path))  # (31250, 200)
        n = len(combined) // 2
        tasks: dict[str, np.ndarray] = {
            "cifar10":  combined[:n].astype(np.float32),
            "cifar100": combined[n:].astype(np.float32),
        }
        if imagenet16_path is not None and Path(imagenet16_path).exists():
            tasks["imagenet16"] = np.load(str(imagenet16_path)).astype(np.float32)

        hps: dict[str, np.ndarray] | None = None
        if arch_strings_path is not None and Path(arch_strings_path).exists():
            with open(arch_strings_path) as f:
                arch_strings = [line.strip() for line in f if line.strip()]
            if len(arch_strings) != n:
                raise ValueError(
                    f"arch_strings_path has {len(arch_strings)} entries but "
                    f"NB201 expects {n} architectures (one per CIFAR-10 row, "
                    f"shared with CIFAR-100 and ImageNet16-120)."
                )
            arch_hps = encode_nb201(arch_strings).astype(np.float32)  # (15625, 6)
            hps = {name: arch_hps.copy() for name in tasks}

        n_configs = next(iter(tasks.values())).shape[0]
        curve_len = next(iter(tasks.values())).shape[1]
        hp_status = "with real HPs" if hps is not None else "with zero HPs (legacy)"
        print(f"  [NAS-Bench-201] {len(tasks)} tasks, "
              f"{n_configs} configs/task, {curve_len} epochs/config  "
              f"(pfn_total_epochs=200, {hp_status})")
        return cls(tasks=tasks, pfn_total_epochs=200, hps=hps)

    @classmethod
    def from_nats_tss(
        cls,
        tss_dir: Path,
        cache_path: Optional[Path] = None,
        hp: str = "200",
    ) -> "RealCurveTable":
        """
        Load NATS-Bench TSS simple files as NAS-Bench-201-style tasks.

        Per-architecture HP vectors are encoded via `encode_nats_tss` (which
        reuses the NB201 encoder, since TSS shares NB201's search space) and
        stored alongside curves in the cache.
        """
        from hp_encoders import encode_nats_tss

        tss_dir = Path(tss_dir)
        if cache_path is None:
            cache_path = tss_dir.parent / f"{tss_dir.name}_curves_hp{hp}_v3.npz"

        if Path(cache_path).exists():
            loaded = np.load(str(cache_path))
            # Cache layout: '<task>' -> curves, '<task>__hps' -> hps
            tasks = {}
            hps: dict[str, np.ndarray] = {}
            for name in loaded.files:
                if name.endswith("__hps"):
                    hps[name[:-5]] = loaded[name].astype(np.float32)
                else:
                    tasks[name] = loaded[name].astype(np.float32)
            hps_or_none: dict[str, np.ndarray] | None = hps if hps else None
            print(f"  [NATS-Bench TSS] loaded cache {cache_path}")
        else:
            files = sorted(tss_dir.glob("*.pickle.pbz2"))
            if not files:
                raise FileNotFoundError(f"No NATS TSS pickle files found in {tss_dir}")

            specs = {
                "cifar10_valid": ("cifar10-valid", ("x-valid",)),
                "cifar100": ("cifar100", ("x-valid", "ori-test")),
                "imagenet16": ("ImageNet16-120", ("x-valid", "ori-test")),
            }
            curves_by_task: dict[str, list[np.ndarray]] = {k: [] for k in specs}
            archs_by_task: dict[str, list[str]] = {k: [] for k in specs}

            for i, path in enumerate(files, start=1):
                with bz2.BZ2File(path, "rb") as f:
                    record = pickle.load(f)
                if hp not in record:
                    continue
                arch_str = record.get("arch_str", None)
                all_results = record[hp].get("all_results", {})
                for task_name, (dataset_key, metric_prefixes) in specs.items():
                    seed_curves = []
                    for (dataset, _seed), result in all_results.items():
                        if dataset != dataset_key:
                            continue
                        eval_acc = result.get("eval_acc1es", {})
                        for metric_prefix in metric_prefixes:
                            vals = [
                                eval_acc[f"{metric_prefix}@{epoch}"]
                                for epoch in range(200)
                                if f"{metric_prefix}@{epoch}" in eval_acc
                            ]
                            if len(vals) == 200:
                                seed_curves.append(np.asarray(vals, dtype=np.float32) / 100.0)
                                break
                    if seed_curves:
                        curves_by_task[task_name].append(np.mean(seed_curves, axis=0))
                        archs_by_task[task_name].append(arch_str if arch_str is not None else "")

                if i % 2500 == 0:
                    print(f"    loaded {i}/{len(files)} TSS architectures ...", flush=True)

            tasks = {}
            hps_dict: dict[str, np.ndarray] = {}
            for task_name, curves in curves_by_task.items():
                if curves:
                    tasks[task_name] = np.stack(curves).astype(np.float32)
                    arch_strs = archs_by_task[task_name]
                    if all(s for s in arch_strs):  # only encode if every entry has arch
                        try:
                            hps_dict[task_name] = encode_nats_tss(arch_strs).astype(np.float32)
                        except ValueError as e:
                            print(f"    [NATS-TSS] HP encoding failed for {task_name}: {e}; "
                                  f"falling back to no HPs for this task")
            if not tasks:
                raise ValueError(f"No NATS TSS curves could be loaded from {tss_dir}")
            hps_or_none = hps_dict if (hps_dict and len(hps_dict) == len(tasks)) else None

            cache_path.parent.mkdir(parents=True, exist_ok=True)
            save_dict = dict(tasks)
            if hps_or_none is not None:
                for k, v in hps_or_none.items():
                    save_dict[f"{k}__hps"] = v
            np.savez_compressed(str(cache_path), **save_dict)
            print(f"  [NATS-Bench TSS] wrote cache {cache_path}")

        n_configs = next(iter(tasks.values())).shape[0]
        curve_len = next(iter(tasks.values())).shape[1]
        hp_status = "with real HPs" if hps_or_none is not None else "with zero HPs (legacy)"
        print(f"  [NATS-Bench TSS] {len(tasks)} tasks, "
              f"{n_configs} configs/task, {curve_len} epochs/config  "
              f"(pfn_total_epochs=200, {hp_status})")
        return cls(tasks=tasks, pfn_total_epochs=200, hps=hps_or_none)

    @classmethod
    def from_nats_sss(
        cls,
        sss_dir: Path,
        cache_path: Optional[Path] = None,
        hp: str = "90",
    ) -> "RealCurveTable":
        """
        Load NATS-Bench SSS simple files.

        Per-architecture HP vectors are encoded via `encode_nats_sss` (5-dim
        normalized channel widths) and stored alongside curves in the cache.
        """
        from hp_encoders import encode_nats_sss

        sss_dir = Path(sss_dir)
        if cache_path is None:
            cache_path = sss_dir.parent / f"{sss_dir.name}_curves_hp{hp}_v2.npz"

        if Path(cache_path).exists():
            loaded = np.load(str(cache_path))
            tasks = {}
            hps: dict[str, np.ndarray] = {}
            for name in loaded.files:
                if name.endswith("__hps"):
                    hps[name[:-5]] = loaded[name].astype(np.float32)
                else:
                    tasks[name] = loaded[name].astype(np.float32)
            hps_or_none: dict[str, np.ndarray] | None = hps if hps else None
            print(f"  [NATS-Bench SSS] loaded cache {cache_path}")
        else:
            files = sorted(sss_dir.glob("*.pickle.pbz2"))
            if not files:
                raise FileNotFoundError(f"No NATS SSS pickle files found in {sss_dir}")

            specs = {
                "cifar10_valid": ("cifar10-valid", "x-valid"),
                "cifar100": ("cifar100", "x-valid"),
                "imagenet16": ("ImageNet16-120", "x-valid"),
            }
            curves_by_task: dict[str, list[np.ndarray]] = {k: [] for k in specs}
            archs_by_task: dict[str, list[str]] = {k: [] for k in specs}
            curve_len = int(hp)

            for i, path in enumerate(files, start=1):
                with bz2.BZ2File(path, "rb") as f:
                    record = pickle.load(f)
                if hp not in record:
                    continue
                arch_str = record.get("arch_str", None)
                all_results = record[hp].get("all_results", {})
                for task_name, (dataset_key, metric_prefix) in specs.items():
                    seed_curves = []
                    for (dataset, _seed), result in all_results.items():
                        if dataset != dataset_key:
                            continue
                        eval_acc = result.get("eval_acc1es", {})
                        vals = [
                            eval_acc[f"{metric_prefix}@{epoch}"]
                            for epoch in range(curve_len)
                            if f"{metric_prefix}@{epoch}" in eval_acc
                        ]
                        if len(vals) == curve_len:
                            seed_curves.append(np.asarray(vals, dtype=np.float32) / 100.0)
                    if seed_curves:
                        curves_by_task[task_name].append(np.mean(seed_curves, axis=0))
                        archs_by_task[task_name].append(arch_str if arch_str is not None else "")

                if i % 5000 == 0:
                    print(f"    loaded {i}/{len(files)} SSS architectures ...", flush=True)

            tasks = {}
            hps_dict: dict[str, np.ndarray] = {}
            for task_name, curves in curves_by_task.items():
                if curves:
                    tasks[task_name] = np.stack(curves).astype(np.float32)
                    arch_strs = archs_by_task[task_name]
                    if all(s for s in arch_strs):
                        try:
                            hps_dict[task_name] = encode_nats_sss(arch_strs).astype(np.float32)
                        except ValueError as e:
                            print(f"    [NATS-SSS] HP encoding failed for {task_name}: {e}; "
                                  f"falling back to no HPs for this task")
            if not tasks:
                raise ValueError(f"No NATS SSS curves could be loaded from {sss_dir}")
            hps_or_none = hps_dict if (hps_dict and len(hps_dict) == len(tasks)) else None

            cache_path.parent.mkdir(parents=True, exist_ok=True)
            save_dict = dict(tasks)
            if hps_or_none is not None:
                for k, v in hps_or_none.items():
                    save_dict[f"{k}__hps"] = v
            np.savez_compressed(str(cache_path), **save_dict)
            print(f"  [NATS-Bench SSS] wrote cache {cache_path}")

        n_configs = next(iter(tasks.values())).shape[0]
        curve_len = next(iter(tasks.values())).shape[1]
        hp_status = "with real HPs" if hps_or_none is not None else "with zero HPs (legacy)"
        print(f"  [NATS-Bench SSS] {len(tasks)} tasks, "
              f"{n_configs} configs/task, {curve_len} epochs/config  "
              f"(pfn_total_epochs=90, {hp_status})")
        return cls(tasks=tasks, pfn_total_epochs=90, hps=hps_or_none)

    @classmethod
    def from_jahs(
        cls,
        cache_path: Path,
        pfn_total_epochs: int = 200,
    ) -> "RealCurveTable":
        """
        Load a JAHS-Bench-201 curve cache produced by build_jahs_cache.py.

        The cache stores one curve array per JAHS task with shape
        (n_configs, n_epochs), normalised to [0, 1]. Newer caches may also
        include '<task>__hps' arrays with aligned normalized HP vectors.
        """
        cache_path = Path(cache_path)
        if not cache_path.exists():
            raise FileNotFoundError(
                f"JAHS cache not found: {cache_path}. "
                "Build it first with build_jahs_cache.py."
            )
        loaded = np.load(str(cache_path))
        tasks = {}
        hps: dict[str, np.ndarray] = {}
        for name in loaded.files:
            if name.startswith("_"):
                continue
            if name.endswith("__hps"):
                hps[name[:-5]] = loaded[name].astype(np.float32)
            else:
                tasks[name] = loaded[name].astype(np.float32)
        if not tasks:
            raise ValueError(f"No JAHS task arrays found in {cache_path}")
        n_configs = next(iter(tasks.values())).shape[0]
        curve_len = next(iter(tasks.values())).shape[1]
        hps_or_none: dict[str, np.ndarray] | None = hps if hps else None
        hp_status = "with real HPs" if hps_or_none is not None else "with zero HPs (legacy)"
        print(f"  [JAHS-Bench-201] loaded cache {cache_path}")
        print(f"  [JAHS-Bench-201] {len(tasks)} tasks, "
              f"{n_configs} configs/task, {curve_len} epochs/config  "
              f"(pfn_total_epochs={pfn_total_epochs}, {hp_status})")
        return cls(tasks=tasks, pfn_total_epochs=pfn_total_epochs, hps=hps_or_none)

    @classmethod
    def from_pd1(
        cls,
        cache_path: Path,
        task: str | None = None,
        drop_final_dup: bool = False,
        pfn_total_epochs: int | None = None,
    ) -> "RealCurveTable":
        """
        Load one PD1 image-classification task from a build_pd1_cache.py cache.

        PD1 is the second real-curve benchmark alongside LCBench, and differs on
        two axes at once: image classification rather than tabular, and a
        fidelity unit of **1000 optimizer steps** rather than one epoch.  PD1
        evaluates on a fixed step schedule, so one curve index is a constant
        compute quantum (1000 steps x 256 examples for every task in this suite)
        and buys a task-dependent number of epochs -- 3.9 on svhn, 5.7 on cifar,
        0.4 on imagenet.  Nothing in the stopping rule depends on which a tick
        is; lambda is simply the price of one more tick.  When reporting PD1,
        lambda is a cost per 1000 training steps and tau counts ticks, not epochs.

        **Loads a single task**, because PD1 tasks have different native T
        (36-53 for the default suite).  `RealCurveTable.__init__` reconciles
        mixed-length tasks by right-padding with the final value, which here
        would fabricate ticks: a padded cifar100 (T=36 -> 53) would show 17 flat
        epochs an arm never trained, let lambda*tau charge for them, and corrupt
        the mean-stop metric.  Callers that want several tasks build one table
        per task -- which is already the per-task convention of the run drivers,
        where `table.curve_len` drives the per-arm cap and the N*T budget.

        Parameters
        ----------
        cache_path : Path to datasets/pd1/pd1_all_curves_15tasks.npz
        task : task name; if None and the cache holds exactly one task, that one
            is used, otherwise a ValueError lists the available names.
        drop_final_dup : drop PD1's final tick.  Every PD1 interval is exactly
            1000 steps *except the last*, which is a short end-of-training
            remainder (62 steps on mnist, 734 on cifar10, 945 on fashion_mnist).
            That tick is therefore not a full compute quantum, and charging it a
            full lambda overcharges by up to 16x.  Dropping it makes every
            retained tick exactly 1000 steps, so lambda prices a uniform
            quantum.  Default False keeps the raw record, matching
            `from_lcbench`'s default; the run drivers pass True for both
            benchmarks (LCBench drops its duplicated epoch 51 the same way).
        pfn_total_epochs : t-normalisation for FT-PFN queries.  Defaults to
            curve_len + 1, mirroring LCBench (curve_len 51 -> 52): tick 0 is the
            un-evaluated initialisation, so recorded ticks are 1..T.
        """
        cache_path = Path(cache_path)
        if not cache_path.exists():
            raise FileNotFoundError(
                f"PD1 cache not found: {cache_path}. Build it first with build_pd1_cache.py."
            )
        loaded = np.load(str(cache_path))
        names = [n for n in loaded.files if not n.startswith("_") and not n.endswith("__hps")]
        if not names:
            raise ValueError(f"No PD1 task arrays found in {cache_path}")

        if task is None:
            if len(names) != 1:
                raise ValueError(
                    f"from_pd1 loads one task at a time (PD1 tasks have different native "
                    f"lengths; mixing them would pad short tasks with fabricated ticks). "
                    f"Pass task=<name>. Available: {sorted(names)}"
                )
            task = names[0]
        if task not in names:
            raise ValueError(f"Unknown PD1 task '{task}'. Available: {sorted(names)}")

        curves = loaded[task].astype(np.float32)
        if drop_final_dup:
            curves = curves[:, :-1]
        tasks = {task: curves}
        hps = ({task: loaded[f"{task}__hps"].astype(np.float32)}
               if f"{task}__hps" in loaded.files else None)

        curve_len = curves.shape[1]
        if pfn_total_epochs is None:
            pfn_total_epochs = curve_len + 1

        meta = {}
        if "_meta" in loaded.files:
            try:
                meta = json.loads(str(loaded["_meta"])).get("tasks", {}).get(task, {})
            except Exception:
                meta = {}
        ept = meta.get("epochs_per_tick")
        unit = (f"1 tick = 1000 steps ~ {ept:.2f} epochs" if isinstance(ept, (int, float))
                else "1 tick = 1000 steps")
        print(f"  [PD1] loaded cache {cache_path}")
        print(f"  [PD1] task '{task}': {curves.shape[0]} configs, {curve_len} ticks/config  "
              f"({unit}; pfn_total_epochs={pfn_total_epochs}, "
              f"{'with real HPs' if hps else 'with zero HPs'})")
        return cls(tasks=tasks, pfn_total_epochs=pfn_total_epochs, hps=hps)

    @classmethod
    def pd1_task_names(cls, cache_path: Path) -> list[str]:
        """Task names available in a PD1 cache, without loading any curves."""
        loaded = np.load(str(Path(cache_path)))
        return sorted(n for n in loaded.files
                      if not n.startswith("_") and not n.endswith("__hps"))

    @classmethod
    def from_nasbench_graph(
        cls,
        cache_path: Path,
        pfn_total_epochs: int | None = None,
    ) -> "RealCurveTable":
        """
        Load a NAS-Bench-Graph curve cache produced by build_nasbench_graph_cache.py.

        The cache stores validation-performance curves in [0, 1]. For most
        graph datasets this is validation accuracy; for proteins it is ROC-AUC.
        Newer caches include '<task>__hps' arrays with aligned normalized
        architecture vectors decoded from the NAS-Bench-Graph hash.
        """
        cache_path = Path(cache_path)
        if not cache_path.exists():
            raise FileNotFoundError(
                f"NAS-Bench-Graph cache not found: {cache_path}. "
                "Build it first with build_nasbench_graph_cache.py."
            )
        loaded = np.load(str(cache_path))
        tasks = {}
        hps: dict[str, np.ndarray] = {}
        for name in loaded.files:
            if name.startswith("_"):
                continue
            if name.endswith("__hps"):
                hps[name[:-5]] = loaded[name].astype(np.float32)
            else:
                tasks[name] = loaded[name].astype(np.float32)
        if not tasks:
            raise ValueError(f"No NAS-Bench-Graph task arrays found in {cache_path}")

        if pfn_total_epochs is None:
            meta_path = cache_path.with_suffix(".json")
            if meta_path.exists():
                with meta_path.open() as f:
                    meta = json.load(f)
                pfn_total_epochs = int(meta.get("epochs") or next(iter(tasks.values())).shape[1])
            else:
                pfn_total_epochs = int(next(iter(tasks.values())).shape[1])

        n_configs = next(iter(tasks.values())).shape[0]
        curve_len = next(iter(tasks.values())).shape[1]
        hps_or_none: dict[str, np.ndarray] | None = hps if hps else None
        hp_status = "with real HPs" if hps_or_none is not None else "with zero HPs (legacy)"
        print(f"  [NAS-Bench-Graph] loaded cache {cache_path}")
        print(f"  [NAS-Bench-Graph] {len(tasks)} tasks, "
              f"{n_configs} configs/task, {curve_len} epochs/config  "
              f"(pfn_total_epochs={pfn_total_epochs}, {hp_status})")
        return cls(tasks=tasks, pfn_total_epochs=int(pfn_total_epochs), hps=hps_or_none)

    @classmethod
    def from_lcbench(
        cls,
        lcbench_dir:  Path,
        held_out_only: bool = True,
        drop_final_dup: bool = False,
    ) -> "RealCurveTable":
        """
        Load LCBench from data_2k_lw.json.

        With held_out_only=True (default), only the 5 tasks unseen during
        FT-PFN training are loaded — ensuring clean out-of-distribution eval.

        Epoch 0 (random initialisation) is always skipped.  By default curves
        are epochs 1–51 (curve_len=51).  With drop_final_dup=True the final
        logged epoch is ALSO dropped — LCBench duplicates it from the
        penultimate epoch, so the YAHPO-Gym-recommended usable range is epochs
        2–51, i.e. the 50 real training epochs (curve_len=50).  Values are
        normalised to [0, 1] by dividing val_accuracy by 100.

        pfn_total_epochs is set to 52 to match the t-normalisation used when
        FT-PFN was trained on LCBench (t = epoch / 52).

        Per-config 7-dim HP vectors (encoded via `encode_lcbench`) are stored
        on the table when the JSON config blocks are present, which is the
        normal case for LCBench dumps.
        """
        from hp_encoders import encode_lcbench, LCBENCH_HP_KEYS

        json_files = sorted(lcbench_dir.glob("*.json"))
        if not json_files:
            raise FileNotFoundError(f"No JSON files found in {lcbench_dir}")

        tasks: dict[str, np.ndarray] = {}
        configs_by_task: dict[str, list[dict]] = {}
        for jf in json_files:
            with jf.open() as f:
                data = json.load(f)
            for task_name, task_data in data.items():
                if held_out_only and task_name not in LCBENCH_HELD_OUT_TASKS:
                    continue
                curves = []
                cfgs: list[dict] = []
                # Skip epoch 0 (random init); with drop_final_dup also drop the
                # duplicated final epoch (LCBench logs 52 pts, last ≈ penultimate).
                sl = slice(1, -1) if drop_final_dup else slice(1, None)
                for run_data in task_data.values():
                    val_acc = run_data.get("log", {}).get("Train/val_accuracy", [])
                    if len(val_acc) > 5:
                        arr = np.array(val_acc[sl], dtype=np.float32) / 100.0
                        curves.append(arr)
                        cfgs.append(run_data.get("config", {}))
                if curves:
                    tasks[task_name] = np.stack(curves)  # (n_configs, 50 or 51)
                    configs_by_task[task_name] = cfgs

        if not tasks:
            raise ValueError(
                "No LCBench tasks loaded. "
                "Check that held_out_only tasks exist in the JSON file."
            )

        # Trim all tasks to the same (minimum) curve length
        min_len = min(v.shape[1] for v in tasks.values())
        tasks   = {k: v[:, :min_len] for k, v in tasks.items()}

        # Build HP table where every run's config has all expected keys.
        hps: dict[str, np.ndarray] = {}
        all_have_hps = True
        for task_name, cfgs in configs_by_task.items():
            if all(all(k in cfg for k in LCBENCH_HP_KEYS) for cfg in cfgs):
                try:
                    hps[task_name] = encode_lcbench(cfgs).astype(np.float32)
                except (KeyError, ValueError) as e:
                    print(f"    [LCBench] HP encoding failed for {task_name}: {e}")
                    all_have_hps = False
                    break
            else:
                all_have_hps = False
                break
        hps_or_none = hps if all_have_hps else None

        tag = "held-out" if held_out_only else "all"
        hp_status = "with real HPs" if hps_or_none is not None else "with zero HPs (legacy)"
        print(f"  [LCBench] {len(tasks)} {tag} tasks, "
              f"{sum(len(v) for v in tasks.values())} configs total, "
              f"{min_len} usable epochs/config  "
              f"(pfn_total_epochs=52, {hp_status})")

        return cls(tasks=tasks, pfn_total_epochs=52, hps=hps_or_none)
