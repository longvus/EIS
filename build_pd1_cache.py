"""Build the PD1 curve cache datasets/pd1/pd1_all_curves_15tasks.npz from the PD1 release.

PD1 (Wang et al., 2024) evaluates on a fixed step schedule: consecutive curve points are 1000
optimizer steps apart. The cache holds the 15 tasks with at least 13 native checkpoints and their
shared pool of 242 configurations.

    python build_pd1_cache.py
"""

from __future__ import annotations

import argparse
import collections
import gzip
import json
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parent
PD1_DIR = REPO / "datasets" / "pd1"
OUT = PD1_DIR / "pd1_image_curves.npz"

# The 4 tuned HPs. PD1 varies only the Nesterov-momentum optimizer knobs; every
# other pipeline HP is fixed to a task-specific default. Encoding mirrors the
# reference implementation's param_space/log_indicator (cfbo_ref/benchmark/pd1.py):
# log-scale lr and (1 - momentum), linear for the two schedule shape params.
HP_SPEC = [
    ("hps.lr_hparams.initial_value", True,  lambda v: v),
    ("hps.lr_hparams.power", False, lambda v: v),
    ("hps.lr_hparams.decay_steps_factor", False, lambda v: v),
    ("hps.opt_hparams.momentum", True, lambda v: 1.0 - v),  # -> one_minus_momentum
]

# 9 batch-256 image tasks, native T in [36, 53].
IMAGE9 = [
    "cifar10_wide_resnet_batch_size_256",
    "cifar100_wide_resnet_batch_size_256",
    "svhn_no_extra_wide_resnet_batch_size_256",
    "mnist_max_pooling_cnn_relu_batch_size_256",
    "mnist_max_pooling_cnn_tanh_batch_size_256",
    "mnist_simple_cnn_relu_batch_size_256",
    "fashion_mnist_max_pooling_cnn_relu_batch_size_256",
    "fashion_mnist_max_pooling_cnn_tanh_batch_size_256",
    "fashion_mnist_simple_cnn_relu_batch_size_256",
]
IMAGENET = ["imagenet_resnet_batch_size_512", "imagenet_resnet_batch_size_256"]

# The three non-image matched tasks. Native T = 38 / 1414 / 22. Adding them
# shrinks the shared-complete pool from 254 to 242, so a cache built with
# --include-nonimage is NOT pool-compatible with the image-only caches and
# must be written to its own --out path.
NONIMAGE = ["lm1b_transformer_batch_size_2048",
            "translate_wmt_xformer_translate_batch_size_64",
            "uniref50_transformer_batch_size_128"]

# svhn at batch 1024 (native T=13). Held out of the image-only suites only
# because it is a second batch size of a task already present at 256; that is
# not a methodological objection, and it costs nothing to include -- the
# 15-task shared pool is the *same* 242 configs as the 14-task pool.
SVHN_1024 = ["svhn_no_extra_wide_resnet_batch_size_1024"]

# Excluded from every suite: native T = 5-7 ticks. Five to seven points leave
# almost no curve for the surrogate to extrapolate, so an episode there is
# governed by the per-arm cap rather than by the max_i EI_i <= lambda rule the
# experiments are testing. Including them the way the CFBO reference does --
# linear interpolation onto a common 50-tick grid -- would manufacture curve
# shape that was never measured. Adding all eight would also shrink the shared
# pool 242 -> 240.
SHORT_HORIZON_EXCLUDED = [
    "cifar10_wide_resnet_batch_size_2048",              # T=7
    "cifar100_wide_resnet_batch_size_2048",             # T=5
    "fashion_mnist_max_pooling_cnn_relu_batch_size_2048",   # T=5
    "fashion_mnist_max_pooling_cnn_tanh_batch_size_2048",   # T=5
    "fashion_mnist_simple_cnn_relu_batch_size_2048",    # T=6
    "mnist_max_pooling_cnn_relu_batch_size_2048",       # T=5
    "mnist_max_pooling_cnn_tanh_batch_size_2048",       # T=5
    "mnist_simple_cnn_relu_batch_size_2048",            # T=7
]

# The deployed suite: every matched test task whose native horizon is long
# enough to exercise a stopping rule (T >= 13). This is the DEFAULT.
DEPLOYED15 = IMAGE9 + IMAGENET + NONIMAGE + SVHN_1024

# Legacy image-only subsets, kept so the caches behind the published 9- and
# 11-task PD1 artifacts stay reproducible. They are NOT pool-compatible with
# DEPLOYED15: image-only pools are 254 configs, the 15-task pool is 242.
SUITES = {
    "full15":  DEPLOYED15,
    "image11": IMAGE9 + IMAGENET,
    "image9":  IMAGE9,
}
DEFAULT_OUT = {
    "full15":  "pd1_all_curves_15tasks.npz",
    "image11": "pd1_image_curves_with_imagenet.npz",
    "image9":  "pd1_image_curves.npz",
}


def taskname(r: dict) -> str:
    act = r.get("hps.activation_fn") or r.get("hps.activation_function")
    act = f"_{act}" if act else ""
    return f'{r["dataset"]}_{r["model"]}{act}_batch_size_{r.get("hps.batch_size")}'


def hp_key(r: dict) -> tuple:
    """Identity of a matched Halton point (raw values, before encoding)."""
    return tuple(round(float(r[col]), 12) for col, _, _ in HP_SPEC)


def encode_hps(keys: list[tuple]) -> np.ndarray:
    """(n, 4) min-max scaled to [0,1], log-scaling lr and one_minus_momentum."""
    raw = np.array(keys, dtype=np.float64)
    cols = []
    for j, (_, is_log, xform) in enumerate(HP_SPEC):
        v = xform(raw[:, j])
        if is_log:
            v = np.log(np.clip(v, 1e-12, None))
        cols.append(v)
    X = np.stack(cols, axis=1)
    lo, hi = X.min(axis=0), X.max(axis=0)
    span = np.where(hi - lo > 0, hi - lo, 1.0)
    return ((X - lo) / span).astype(np.float32)


def load_rows(paths: list[Path]) -> list[dict]:
    rows = []
    for p in paths:
        with gzip.open(p, "rt") as f:
            for line in f:
                rows.append(json.loads(line))
    return rows


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--suite", choices=sorted(SUITES), default="full15",
                    help="full15 (default): all matched test tasks with native "
                         "T>=13, shared pool 242. image11 / image9 reproduce the "
                         "legacy image-only caches (pool 254).")
    ap.add_argument("--include-phase0", action="store_true",
                    help="also read matched phase0 (100 configs); phase1 (400) is the "
                         "scaled-up run and is the default source")
    ap.add_argument("--out", type=Path, default=None,
                    help="defaults to the chosen suite's canonical filename")
    args = ap.parse_args()

    suite = list(SUITES[args.suite])
    if args.out is None:
        args.out = PD1_DIR / DEFAULT_OUT[args.suite]

    paths = [PD1_DIR / "pd1_matched_phase1_results.jsonl.gz"]
    if args.include_phase0:
        paths.insert(0, PD1_DIR / "pd1_matched_phase0_results.jsonl.gz")
    for p in paths:
        if not p.exists():
            raise FileNotFoundError(
                f"{p} not found. Fetch PD1 with:\n"
                f"  curl -L -o pd1.tar.gz http://storage.googleapis.com/gresearch/pint/pd1.tar.gz\n"
                f"  tar xzf pd1.tar.gz && cp pd1/pd1_matched_*.jsonl.gz {PD1_DIR}/"
            )

    print(f"  [PD1] reading {', '.join(p.name for p in paths)}")
    rows = load_rows(paths)
    by_task: dict[str, list[dict]] = collections.defaultdict(list)
    for r in rows:
        by_task[taskname(r)].append(r)

    missing = [t for t in suite if t not in by_task]
    if missing:
        raise ValueError(f"tasks absent from PD1 dump: {missing}")

    # Per task: native T = the longest recorded curve; complete = curves of that length
    # that did not diverge. Partial curves are early deaths (len 1-8), not truncated
    # late-bloomers, so "complete" and "did not diverge" coincide.
    complete: dict[str, dict[tuple, np.ndarray]] = {}
    meta_tasks: dict[str, dict] = {}
    for t in suite:
        rs = by_task[t]
        T = max((len(r["valid/error_rate"]) if r.get("valid/error_rate") else 0) for r in rs)
        ok: dict[tuple, np.ndarray] = {}
        for r in rs:
            c = r.get("valid/error_rate")
            if c and len(c) == T and r.get("status") == "done":
                ok[hp_key(r)] = 1.0 - np.asarray(c, dtype=np.float64)  # accuracy
        complete[t] = ok
        ref = next(r for r in rs if r.get("valid/error_rate") and len(r["valid/error_rate"]) == T)
        n, bs = ref.get("hps.train_size"), ref.get("hps.batch_size")
        steps_per_epoch = (n / bs) if n and bs else float("nan")
        meta_tasks[t] = {
            "T": int(T),
            "n_trials": len(rs),
            "n_complete": len(ok),
            "train_size": n,
            "batch_size": bs,
            "steps_per_epoch": round(steps_per_epoch, 2),
            "epochs_per_tick": round(1000.0 / steps_per_epoch, 3),
            "total_epochs": int(ref["epoch"][-1]),
        }

    shared = sorted(set.intersection(*[set(ok) for ok in complete.values()]))
    if not shared:
        raise ValueError("no config is complete on every task in the suite")
    print(f"  [PD1] shared-complete pool across {len(suite)} tasks: {len(shared)} configs")

    hps = encode_hps(shared)
    payload: dict[str, np.ndarray] = {}
    for t in suite:
        curves = np.stack([complete[t][k] for k in shared]).astype(np.float32)
        if not np.all(np.isfinite(curves)):
            raise ValueError(f"{t}: non-finite accuracy")
        if curves.min() < -1e-6 or curves.max() > 1 + 1e-6:
            raise ValueError(f"{t}: accuracy outside [0,1]: [{curves.min()}, {curves.max()}]")
        payload[t] = curves
        payload[f"{t}__hps"] = hps
        m = meta_tasks[t]
        print(f"    {t:52s} T={m['T']:3d} n={curves.shape[0]:3d} "
              f"acc=[{curves.min():.3f},{curves.max():.3f}] "
              f"{m['epochs_per_tick']:.2f} ep/tick  ({m['total_epochs']} ep total)")

    payload["_meta"] = np.asarray(json.dumps({
        "source": [p.name for p in paths],
        "benchmark": "pd1",
        "fidelity_unit": "1000 optimizer steps (NOT one epoch)",
        "target": "1 - valid/error_rate",
        "pool": "configs complete on every task in the suite; diverged trials dropped",
        "n_shared_configs": len(shared),
        "hp_spec": [c for c, _, _ in HP_SPEC],
        "tasks": meta_tasks,
    }, indent=2))

    args.out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.out, **payload)
    print(f"  [PD1] wrote {args.out}  ({args.out.stat().st_size / 1e6:.1f} MB)")


if __name__ == "__main__":
    main()
