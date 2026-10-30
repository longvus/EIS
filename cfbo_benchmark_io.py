"""Build and load the released CFBO TaskSet and ODBench test slices.

The CFBO TaskSet JSON labels its only curve as balanced accuracy, but the values
are unbounded losses. We reproduce CFBO's logistic loss transform, scaled by the
mean first checkpoint over the full candidate pool; this is not target min-max
scaling. Hyperparameter coordinates are separately normalized from their known
candidate values. ODBench keeps its native bounded validation performance.
"""

from __future__ import annotations

import hashlib

import json

import math

import os

from pathlib import Path

import tempfile

from typing import Mapping, Sequence

import numpy as np

ROOT = Path(__file__).resolve().parent

SOURCE_DIR = ROOT / "cfbo_ref" / "data"

CACHE_DIR = ROOT / "datasets" / "cfbo_benchmarks"

SOURCE_PATHS = {
    "taskset": SOURCE_DIR / "taskset_chosen.json",
    "odbench": SOURCE_DIR / "od_datasets.json",
}

SOURCE_SHA256 = {
    "taskset": "50447ad45341e093bc48594a7b1724682e9467ba44960537c1b6047a508e3b99",
    "odbench": "79007183736d2d8aada70a6e30a5f124e165c7c1bf839780d046bd907bbd1e64",
}

CACHE_PATHS = {
    name: CACHE_DIR / f"{name}_cfbo_test_curves.npz"
    for name in SOURCE_PATHS
}

TASKS = {
    "taskset": (
        "rnn_text_classification_family_seed8",
        "rnn_text_classification_family_seed82",
        "rnn_text_classification_family_seed89",
        "word_rnn_language_model_family_seed84",
        "word_rnn_language_model_family_seed98",
        "word_rnn_language_model_family_seed99",
        "char_rnn_language_model_family_seed84",
        "char_rnn_language_model_family_seed94",
        "char_rnn_language_model_family_seed96",
    ),
    "odbench": (
        "d8_a1",
        "d8_a2",
        "d8_a3",
        "d9_a1",
        "d9_a2",
        "d9_a3",
        "d10_a1",
        "d10_a2",
        "d10_a3",
    ),
}

HP_SPECS = {
    "taskset": (
        ("learning_rate", True),
        ("beta1", False),
        ("beta2", False),
        ("epsilon", True),
        ("l1", True),
        ("l2", True),
        ("linear_decay", True),
        ("exponential_decay", True),
    ),
    "odbench": (
        ("batch_size", True),
        ("learning_rate", True),
        ("momentum", False),
        ("weight_decay", True),
    ),
}

EXPECTED_CONFIGS = {"taskset": 1000, "odbench": 500}

EXPECTED_RAW_POINTS = {"taskset": 51, "odbench": 21}

CURVE_FIELDS = {
    "taskset": "Train/val_balanced_accuracy",
    "odbench": "Train/val_accuracy",
}

PFN_TOTAL_EPOCHS = {"taskset": 50, "odbench": 20}

OD_DATASETS = {
    "d8": "x-ray-rheumatology",
    "d9": "secondary-chains",
    "d10": "avatar-recognition-nuexe",
}

OD_ARCHITECTURES = {
    "a1": "Faster R-CNN ResNet-50/FPN",
    "a2": "Faster R-CNN HRNetV2p-W18",
    "a3": "SSD-Lite MobileNetV2",
}

def sha256_path(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()

def _ordered_entries(task_data: Mapping[str, object], expected: int) -> list[dict]:
    try:
        ids = sorted((int(key), key) for key in task_data)
    except ValueError as error:
        raise ValueError("configuration IDs must be integers") from error
    if [index for index, _ in ids] != list(range(expected)):
        raise ValueError(f"expected configuration IDs 0..{expected - 1}")
    return [task_data[key] for _, key in ids]

def _encode_hps(
    entries: Sequence[Mapping[str, object]],
    spec: Sequence[tuple[str, bool]],
) -> tuple[np.ndarray, dict[str, list[float]]]:
    columns = []
    observed = {}
    for key, use_log in spec:
        values = np.asarray(
            [float(entry["config"][key]) for entry in entries], dtype=np.float64
        )
        if not np.isfinite(values).all():
            raise ValueError(f"non-finite hyperparameter values for {key}")
        observed[key] = [float(values.min()), float(values.max())]
        if use_log:
            if np.any(values <= 0.0):
                raise ValueError(f"log-scaled hyperparameter {key} is not positive")
            values = np.log(values)
        span = float(values.max() - values.min())
        if span <= 0.0:
            raise ValueError(f"hyperparameter {key} is constant")
        columns.append((values - values.min()) / span)
    encoded = np.stack(columns, axis=1).astype(np.float32)
    return encoded, observed

def _taskset_loss_to_performance(raw: np.ndarray) -> tuple[np.ndarray, float]:
    """Reproduce pfn_normalize(0, inf, 0, mean(initial), minimize=True)."""
    raw = np.asarray(raw, dtype=np.float64)
    if raw.ndim != 2 or raw.shape[1] < 2:
        raise ValueError("TaskSet raw curves must be a 2-D initial-plus-curve array")
    if not np.isfinite(raw).all() or np.any(raw < 0.0):
        raise ValueError("TaskSet losses must be finite and nonnegative")
    initial_mean = float(raw[:, 0].mean())
    if not math.isfinite(initial_mean) or initial_mean <= 0.0:
        raise ValueError("TaskSet mean initial loss must be positive")

    linear = 2.0 * raw[:, 1:] / initial_mean - 1.0
    scale = 1.0 + math.exp(-1.0)
    performance = scale * np.exp(-np.logaddexp(0.0, linear))
    return np.clip(performance, 0.0, 1.0).astype(np.float32), initial_mean

def _extract_benchmark(benchmark: str, source: Mapping[str, object]) -> dict[str, object]:
    if benchmark not in TASKS:
        raise ValueError(f"unknown benchmark: {benchmark}")
    missing = sorted(set(TASKS[benchmark]) - set(source))
    if missing:
        raise ValueError(f"{benchmark} source is missing tasks: {missing}")

    payload: dict[str, object] = {}
    task_metadata = {}
    reference_hps = None
    field = CURVE_FIELDS[benchmark]
    for task in TASKS[benchmark]:
        entries = _ordered_entries(source[task], EXPECTED_CONFIGS[benchmark])
        hps, observed = _encode_hps(entries, HP_SPECS[benchmark])
        if reference_hps is None:
            reference_hps = hps
        elif not np.array_equal(reference_hps, hps):
            raise ValueError(f"{benchmark} candidate roster differs for task {task}")

        raw = np.asarray([entry["log"][field] for entry in entries], dtype=np.float64)
        expected_shape = (EXPECTED_CONFIGS[benchmark], EXPECTED_RAW_POINTS[benchmark])
        if raw.shape != expected_shape:
            raise ValueError(f"{task} curve shape {raw.shape} != {expected_shape}")
        if not np.isfinite(raw).all():
            raise ValueError(f"{task} contains non-finite curve values")

        if benchmark == "taskset":
            curves, initial_mean = _taskset_loss_to_performance(raw)
            metric = "CFBO-normalized validation loss (higher is better)"
            transform = "published CFBO pfn_normalize using mean first checkpoint"
        else:
            if np.any(raw < 0.0) or np.any(raw > 1.0):
                raise ValueError(f"{task} validation performance is outside [0,1]")
            curves = raw[:, 1:].astype(np.float32)
            initial_mean = float(raw[:, 0].mean())
            metric = "released validation performance (higher is better)"
            transform = "drop shared initial value; otherwise preserve native [0,1] units"

        payload[task] = curves
        payload[f"{task}__hps"] = hps
        task_metadata[task] = {
            "n_configs": int(curves.shape[0]),
            "curve_len": int(curves.shape[1]),
            "raw_initial_mean": initial_mean,
            "performance_min": float(curves.min()),
            "performance_max": float(curves.max()),
            "observed_hp_bounds": observed,
            "metric": metric,
            "transform": transform,
        }
        if benchmark == "odbench":
            dataset_id, architecture_id = task.split("_")
            task_metadata[task].update({
                "rf100_dataset": OD_DATASETS[dataset_id],
                "architecture": OD_ARCHITECTURES[architecture_id],
            })

    payload["_meta"] = np.asarray(json.dumps({
        "schema": "cfbo-fixed-test-curves/v1",
        "benchmark": benchmark,
        "source_path": str(SOURCE_PATHS[benchmark].relative_to(ROOT)),
        "source_sha256": SOURCE_SHA256[benchmark],
        "tasks": list(TASKS[benchmark]),
        "curve_field": field,
        "curve_field_warning": (
            "field is mislabeled; values are minimizing losses"
            if benchmark == "taskset"
            else "paper does not identify the exact object-detection metric"
        ),
        "hp_fields": [key for key, _ in HP_SPECS[benchmark]],
        "log_scaled_hp_fields": [key for key, use_log in HP_SPECS[benchmark] if use_log],
        "pfn_total_epochs": PFN_TOTAL_EPOCHS[benchmark],
        "task_metadata": task_metadata,
    }, sort_keys=True, separators=(",", ":")))
    return payload

def build_cache(benchmark: str, output_path: Path | None = None) -> Path:
    source_path = SOURCE_PATHS[benchmark]
    if not source_path.is_file():
        raise FileNotFoundError(f"CFBO source table not found: {source_path}")
    actual_hash = sha256_path(source_path)
    if actual_hash != SOURCE_SHA256[benchmark]:
        raise ValueError(
            f"{benchmark} source SHA-256 mismatch: {actual_hash} != "
            f"{SOURCE_SHA256[benchmark]}"
        )
    with source_path.open(encoding="utf-8") as handle:
        source = json.load(handle)
    payload = _extract_benchmark(benchmark, source)

    output_path = Path(output_path or CACHE_PATHS[benchmark])
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        prefix=f".{output_path.stem}.", suffix=".npz", dir=output_path.parent, delete=False
    ) as handle:
        temporary = Path(handle.name)
    try:
        np.savez_compressed(temporary, **payload)
        os.replace(temporary, output_path)
    finally:
        temporary.unlink(missing_ok=True)
    return output_path
