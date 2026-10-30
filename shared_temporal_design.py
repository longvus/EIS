"""Fixed specification of the Gaussian surrogates: modes, kernel parameters, costs, tasks and seeds.

Lengthscales come from the hyperparameter geometry only; nothing is fitted to scores.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass

import hashlib

import json

import math

from pathlib import Path

from benchmark_design import LCBENCH_TASKS, PD1_TASKS

ROOT = Path(__file__).resolve().parent

STAGING = ROOT / "staging/shared_temporal_gp/v1"

GEOMETRY = ROOT / "staging/hp_geometry/v1/hp_geometry_v1.json"

GEOMETRY_SHA256 = "998ca942e7775518cd1d4f9b7edf3899de9a5b678d40d42e5f965ac1e2a5bf4f"

MODES = ("b_local", "b_asymptote", "b_hp_temporal", "b_pool_temporal")

WEIGHTS = {
    "b_local": ((1., 0., 0.), (1., 0., 0.)),
    "b_asymptote": ((.5, .5, 0.), (1., 0., 0.)),
    "b_hp_temporal": ((.5, .5, 0.), (.5, .5, 0.)),
    "b_pool_temporal": ((.5, .5, 0.), (.5, .25, .25)),
}

TASKS = {"lcbench": LCBENCH_TASKS, "pd1": PD1_TASKS}

def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("ascii")

def sha(value):
    return hashlib.sha256(canonical(value)).hexdigest()

def file_sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(8 << 20), b""):
            digest.update(block)
    return digest.hexdigest()

@dataclass(frozen=True)
class SharedTemporalSnapshot:
    lengthscales: tuple[float, ...]
    configuration_encoder_id: str
    mode: str = "b_pool_temporal"
    basis_scales: tuple[int, ...] = (1, 8, 64, 512)
    mean: float = .5
    asymptote_variance: float = 1 / 32
    temporal_variance: float = 1 / 32
    noise_base: float = 2.**-16
    noise_decay: float = 2.**-10
    jitter: float = 2.**-30
    sigma_floor: float = 2.**-16
    support_lo: float = 0.
    support_hi: float = 1.
    decision_variance: str = "observed"

    def __post_init__(self):
        if self.mode not in MODES:
            raise ValueError("unknown sharing mode")
        ls = tuple(float(v) for v in self.lengthscales)
        if not ls or any(not math.isfinite(v) or v <= 0 for v in ls):
            raise ValueError("positive finite lengthscales required")
        object.__setattr__(self, "lengthscales", ls)
        if not isinstance(self.configuration_encoder_id, str) or not self.configuration_encoder_id:
            raise ValueError("encoder ID required")
        if tuple(self.basis_scales) != (1, 8, 64, 512):
            raise ValueError("v1 basis scales are fixed")
        object.__setattr__(self, "basis_scales", (1, 8, 64, 512))
        for name, expected in (("mean", .5), ("asymptote_variance", 1/32),
                ("temporal_variance", 1/32), ("noise_base", 2.**-16),
                ("noise_decay", 2.**-10), ("jitter", 2.**-30),
                ("sigma_floor", 2.**-16), ("support_lo", 0.), ("support_hi", 1.)):
            value = float(getattr(self, name))
            if value.hex() != float(expected).hex():
                raise ValueError(f"v1 fixes {name}")
            object.__setattr__(self, name, value)
        if self.decision_variance != "observed":
            raise ValueError("v1 uses fresh-observation variance")

    @property
    def arms_are_autonomous(self):
        return self.mode == "b_local"

    def to_dict(self):
        values = asdict(self)
        values.update(schema="shared-temporal-snapshot/v1",
            bounded_transition="sequential_truncated_normal",
            basis_formula="(scale+1)/(scale+t)",
            noise_formula="noise_base+noise_decay/(1+t)^2+jitter",
            coefficient_order="arm-major: asymptote,b1,b2,b3,b4",
            sharing_weights=[list(w) for w in WEIGHTS[self.mode]])
        return {key: value.hex() if isinstance(value, float) else
                [v.hex() for v in value] if key == "lengthscales" else value
                for key, value in values.items()}

    @classmethod
    def from_dict(cls, payload):
        if payload.get("schema") != "shared-temporal-snapshot/v1":
            raise ValueError("snapshot schema differs")
        values = {k: payload[k] for k in cls.__dataclass_fields__}
        values["lengthscales"] = tuple(float.fromhex(v) for v in values["lengthscales"])
        for key in ("mean", "asymptote_variance", "temporal_variance", "noise_base",
                    "noise_decay", "jitter", "sigma_floor", "support_lo", "support_hi"):
            values[key] = float.fromhex(values[key])
        result = cls(**values)
        if canonical(result.to_dict()) != canonical(payload):
            raise ValueError("noncanonical or inconsistent snapshot")
        return result

    def sha256(self):
        return sha(self.to_dict())

def snapshot(benchmark, mode):
    if file_sha(GEOMETRY) != GEOMETRY_SHA256:
        raise ValueError("HP geometry hash differs")
    data = json.loads(GEOMETRY.read_bytes())
    if data["outcomes_used"] is not False:
        raise ValueError("geometry used outcomes")
    item = data["benchmarks"][benchmark]
    return SharedTemporalSnapshot(tuple(item["geometry"]["lengthscales"]), item["encoder_id"], mode)
