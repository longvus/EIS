"""Directed-rounding enclosure of the selected root's two-step advantage (the gate's certificate)."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field, fields

from fractions import Fraction

import math

import operator

from typing import Sequence

import gmpy2 as g

from mpfr_truncated_ei import IntervalRecord, NumericalUnresolved, SignStatus, bound_truncated_normal_ei

from interval_linalg import normal_mass_interval

from mpfr_interval import MPInterval

from gp_types import SurrogateQuery

@dataclass(frozen=True)
class SharedMPFRCertificateConfig:
    mpfr_precision_bits: int = 256
    root_ei_tolerance: float = 2e-10
    max_root_ei_cells: int = 64
    outer_cells: int = 32
    child_ei_tolerance: float = 2e-7
    max_child_ei_cells: int = 4

    def __post_init__(self) -> None:
        for name in (
            "mpfr_precision_bits",
            "max_root_ei_cells",
            "outer_cells",
            "max_child_ei_cells",
        ):
            value = getattr(self, name)
            if isinstance(value, bool):
                raise TypeError(f"{name} must be an integer")
            try:
                normalized = operator.index(value)
            except TypeError as exc:
                raise TypeError(f"{name} must be an integer") from exc
            object.__setattr__(self, name, normalized)
        if self.mpfr_precision_bits < 80:
            raise ValueError("MPFR precision must be at least 80 bits")
        for name in ("root_ei_tolerance", "child_ei_tolerance"):
            value = float(getattr(self, name))
            if not math.isfinite(value) or value <= 0.0:
                raise ValueError("EI tolerances must be finite and positive")
            object.__setattr__(self, name, value)
        if min(self.max_root_ei_cells, self.outer_cells, self.max_child_ei_cells) < 1:
            raise ValueError("certificate resource limits must be positive")

def _interval_from_dict(payload: object) -> IntervalRecord:
    if type(payload) is not dict or set(payload) != {"lower", "upper"}:
        raise ValueError("shared certificate interval schema differs")
    lower = payload["lower"]
    upper = payload["upper"]
    if (
        isinstance(lower, bool)
        or not isinstance(lower, (int, float))
        or isinstance(upper, bool)
        or not isinstance(upper, (int, float))
    ):
        raise ValueError("shared certificate interval endpoints are not numeric")
    return IntervalRecord(float(lower), float(upper))

@dataclass(frozen=True)
class MPFRJointParameters:
    queries: tuple[SurrogateQuery, ...]
    means: tuple[MPInterval, ...]
    latent_covariance: tuple[tuple[MPInterval, ...], ...]
    observation_covariance: tuple[tuple[MPInterval, ...], ...]

@dataclass(frozen=True)
class SharedSelectedRootBound:
    candidate_id: str
    root_mean: IntervalRecord
    root_sigma: IntervalRecord
    root_ei: IntervalRecord
    child_premium: IntervalRecord
    advantage: IntervalRecord
    gate_margin: IntervalRecord
    gate_status: SignStatus
    child_candidate_ids: tuple[str, ...]
    outer_cells: int
    child_ei_cells: int
    inner_tolerance_unmet: bool
    coefficient_enclosure_sha256: str

    def to_dict(self) -> dict[str, object]:
        result = asdict(self)
        result["gate_status"] = self.gate_status.value
        result["child_candidate_ids"] = list(self.child_candidate_ids)
        return result

    @classmethod
    def from_dict(cls, payload: dict[str, object]) -> "SharedSelectedRootBound":
        if type(payload) is not dict or set(payload) != {field.name for field in fields(cls)}:
            raise ValueError("shared selected-root certificate schema differs")
        child_ids = payload["child_candidate_ids"]
        if not isinstance(child_ids, list):
            raise ValueError("shared selected-root child roster differs")
        values = dict(payload)
        for name in ("root_mean", "root_sigma", "root_ei", "child_premium", "advantage", "gate_margin"):
            values[name] = _interval_from_dict(values[name])
        values["gate_status"] = SignStatus(values["gate_status"])
        values["child_candidate_ids"] = tuple(child_ids)
        return cls(**values)

@dataclass(frozen=True)
class _MPFRChildLaw:
    candidate_id: str
    query_epoch: float
    base_mean: MPInterval
    mean_slope: MPInterval
    observation_variance: MPInterval
    sigma: MPInterval

@dataclass(frozen=True)
class _CompiledSelectedRoot:
    candidate_id: str
    root_epoch: float
    root_mean: MPInterval
    root_observation_variance: MPInterval
    root_sigma: MPInterval
    children: tuple[_MPFRChildLaw, ...]
    coefficient_hash: str

@dataclass(frozen=True)
class _OuterCell:
    mass: MPInterval
    premium: MPInterval
    inner_cells: int
    inner_resolved: bool

@dataclass(frozen=True)
class _RawSelectedRootBound:
    compiled: _CompiledSelectedRoot
    root_ei: MPInterval
    child_premium: MPInterval
    advantage: MPInterval
    outer_cells: int
    child_ei_cells: int
    inner_tolerance_unmet: bool
    diagnostics: dict[str, int] = field(default_factory=dict)

def _point(value: float | int, precision: int) -> MPInterval:
    if isinstance(value, bool):
        raise TypeError("boolean is not a numeric interval point")
    if isinstance(value, int):
        exact = g.mpfr(
            value,
            precision=max(precision, abs(value).bit_length() + 2),
        )
        return MPInterval(exact, exact, precision)
    return MPInterval.point(value, precision)

def _bounds(lower: float, upper: float, precision: int) -> MPInterval:
    return MPInterval.bounds(lower, upper, precision)

def _truncated_mass(
    mu: MPInterval,
    sigma: MPInterval,
    left: float,
    right: float,
) -> MPInterval:
    if sigma.lo <= 0:
        raise NumericalUnresolved("truncated-normal scale is not proved positive")
    precision = mu.precision
    denominator = normal_mass_interval(
        (_point(0, precision) - mu) / sigma,
        (_point(1, precision) - mu) / sigma,
    )
    if denominator.lo <= 0:
        raise NumericalUnresolved("truncated-normal normalizer is unresolved")
    numerator = normal_mass_interval(
        (_point(left, precision) - mu) / sigma,
        (_point(right, precision) - mu) / sigma,
    )
    return (numerator / denominator).maximum(0).minimum(1)

def _fraction_lower(value: Fraction) -> float:
    result = float(value)
    return math.nextafter(result, -math.inf) if Fraction.from_float(result) > value else result

def _fraction_upper(value: Fraction) -> float:
    result = float(value)
    return math.nextafter(result, math.inf) if Fraction.from_float(result) < value else result

def _simplex_expectation(cells: Sequence[_OuterCell], precision: int) -> MPInterval:
    mass_lower = [Fraction.from_float(cell.mass.to_float()[0]) for cell in cells]
    mass_upper = [Fraction.from_float(cell.mass.to_float()[1]) for cell in cells]
    value_lower = [Fraction.from_float(cell.premium.to_float()[0]) for cell in cells]
    value_upper = [Fraction.from_float(cell.premium.to_float()[1]) for cell in cells]
    if sum(mass_lower, Fraction(0)) > 1 or sum(mass_upper, Fraction(0)) < 1:
        raise NumericalUnresolved("outer probability intervals do not contain a simplex")

    def extreme(values: Sequence[Fraction], reverse: bool) -> Fraction:
        masses = list(mass_lower)
        remaining = Fraction(1) - sum(masses, Fraction(0))
        for index in sorted(
            range(len(values)), key=lambda item: values[item], reverse=reverse
        ):
            addition = min(
                remaining,
                max(Fraction(0), mass_upper[index] - masses[index]),
            )
            masses[index] += addition
            remaining -= addition
            if remaining == 0:
                break
        if remaining != 0:
            raise NumericalUnresolved("outer simplex allocation did not reach unit mass")
        return sum(
            (mass * value for mass, value in zip(masses, values)),
            Fraction(0),
        )

    lower = max(0.0, _fraction_lower(extreme(value_lower, reverse=False)))
    upper = max(0.0, _fraction_upper(extreme(value_upper, reverse=True)))
    return _bounds(lower, upper, precision)

def _bound_selected_root(
    compiled: _CompiledSelectedRoot,
    root_ei,
    incumbent: float,
    lambda_: float,
    config: SharedMPFRCertificateConfig,
) -> _RawSelectedRootBound:
    precision = config.mpfr_precision_bits
    lambda_iv = _point(lambda_, precision)
    incumbent_iv = _point(incumbent, precision)
    if not 0 <= incumbent <= 1 or compiled.root_sigma.lo <= 0:
        raise NumericalUnresolved("invalid incumbent or nonpositive root scale")
    if any(child.sigma.lo <= 0 for child in compiled.children):
        raise NumericalUnresolved("child scale is not proved positive")
    stats = dict(endpoint_evaluations=0, endpoint_cache_hits=0,
                 wholecell_evaluations=0, unique_ei_cells=0,
                 unresolved_evaluations=0, pruned_children=0,
                 referenced_ei_cells=0)
    memo = {}

    def evaluate(index, left, right=None):
        key = (index, left)
        if right is None and key in memo:
            stats["endpoint_cache_hits"] += 1
            return memo[key]
        fantasy = _point(left, precision) if right is None else _bounds(left, right, precision)
        child = compiled.children[index]
        result = bound_truncated_normal_ei(
            child.base_mean + child.mean_slope * (fantasy - compiled.root_mean),
            child.sigma, incumbent_iv.maximum(fantasy),
            tolerance=config.child_ei_tolerance, max_cells=config.max_child_ei_cells,
        )
        # Reject malformed backend results before any screening comparison.
        if not all(g.is_finite(x) for x in (result.value.lo, result.value.hi)) or result.value.lo > result.value.hi:
            raise NumericalUnresolved("nonfinite or inverted child EI enclosure")
        stats["endpoint_evaluations" if right is None else "wholecell_evaluations"] += 1
        stats["unique_ei_cells"] += result.cells
        stats["unresolved_evaluations"] += int(not result.resolved)
        if right is None:
            memo[key] = result
        return result

    active = []
    for index, child in enumerate(compiled.children):
        slope = child.mean_slope
        # For restricted-normal EI, 0 <= d_mu EI <= P(X>b), d_b EI=-P(X>b).
        # These signs hold for every fixed tuple in the coefficient enclosure.
        mode = "decreasing" if slope.hi <= 0 else "unimodal" if slope.lo >= 0 and slope.hi <= 1 else "fallback"
        if mode != "fallback":
            peak = evaluate(index, 0.0 if mode == "decreasing" else incumbent)
            if peak.value.hi <= lambda_iv.lo:
                stats["pruned_children"] += 1
                continue
        active.append((index, mode))
    if not compiled.children:
        child_premium = _point(0, precision)
        outer_cell_count = 0
        child_cell_count = 0
        inner_unresolved = False
    else:
        edges = {
            0.0,
            1.0,
            incumbent,
            *(index / config.outer_cells for index in range(1, config.outer_cells)),
        }
        ordered = sorted(value for value in edges if 0.0 <= value <= 1.0)
        cells = []
        for left, right in zip(ordered, ordered[1:]):
            if left >= right:
                continue
            mass = _truncated_mass(
                compiled.root_mean, compiled.root_sigma, left, right
            )
            best_premium = _point(0, precision)
            inner_cells = 0
            inner_resolved = True
            for index, mode in active:
                if mode == "fallback":
                    a = evaluate(index, left, right)
                    value, used = a.value, (a,)
                else:
                    a, z = evaluate(index, left), evaluate(index, right)
                    increasing = mode == "unimodal" and right <= incumbent
                    value = (MPInterval(a.value.lo, z.value.hi, precision) if increasing
                             else MPInterval(z.value.lo, a.value.hi, precision))
                    used = (a, z)
                premium = (value - lambda_iv).maximum(0)
                best_premium = best_premium.maximum(premium)
                inner_cells += sum(item.cells for item in used)
                inner_resolved = inner_resolved and all(item.resolved for item in used)
            cells.append(_OuterCell(mass, best_premium, inner_cells, inner_resolved))
        child_premium = _simplex_expectation(cells, precision)
        outer_cell_count = len(cells)
        stats["referenced_ei_cells"] = sum(cell.inner_cells for cell in cells)
        child_cell_count = stats["unique_ei_cells"]
        inner_unresolved = bool(stats["unresolved_evaluations"])
    advantage = root_ei.value - lambda_iv + child_premium
    return _RawSelectedRootBound(
        compiled, root_ei.value, child_premium, advantage,
        outer_cell_count, child_cell_count, inner_unresolved, stats,
    )

def _selected_bound_for_eta(raw, lambda_, eta, precision):
    compiled = raw.compiled
    advantage = raw.advantage
    lambda_iv = _point(lambda_, precision)
    # Keep this expression on raw MPFR endpoints, not serialized float bounds.
    gate_margin = advantage - _point(eta, precision) * lambda_iv
    gate_status = (
        SignStatus.FAIL
        if gate_margin.lo > 0
        else SignStatus.PASS
        if gate_margin.hi <= 0
        else SignStatus.UNKNOWN
    )
    return SharedSelectedRootBound(
        candidate_id=compiled.candidate_id,
        root_mean=IntervalRecord.from_interval(compiled.root_mean),
        root_sigma=IntervalRecord.from_interval(compiled.root_sigma),
        root_ei=IntervalRecord.from_interval(raw.root_ei),
        child_premium=IntervalRecord.from_interval(raw.child_premium),
        advantage=IntervalRecord.from_interval(advantage),
        gate_margin=IntervalRecord.from_interval(gate_margin),
        gate_status=gate_status,
        child_candidate_ids=tuple(child.candidate_id for child in compiled.children),
        outer_cells=raw.outer_cells,
        child_ei_cells=raw.child_ei_cells,
        inner_tolerance_unmet=raw.inner_tolerance_unmet,
        coefficient_enclosure_sha256=compiled.coefficient_hash,
    )
