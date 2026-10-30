"""Rigorous MPFR enclosure of the expected improvement of a Gaussian truncated to [0,1]."""

from __future__ import annotations

from dataclasses import dataclass

from enum import Enum

import math

import gmpy2 as g

from interval_linalg import normal_mass_interval, normal_pdf_interval

from mpfr_interval import MPInterval

class SignStatus(str, Enum):
    FAIL = "FAIL"
    PASS = "PASS"
    UNKNOWN = "UNKNOWN"

class NumericalUnresolved(ArithmeticError):
    pass

@dataclass(frozen=True)
class IntervalRecord:
    lower: float
    upper: float

    def __post_init__(self) -> None:
        if math.isnan(self.lower) or math.isnan(self.upper) or self.lower > self.upper:
            raise ValueError("invalid serialized interval")

    @classmethod
    def from_interval(cls, value: MPInterval) -> "IntervalRecord":
        return cls(*value.to_float())

    @property
    def width(self) -> float:
        return self.upper - self.lower

@dataclass(frozen=True)
class _EIEnclosure:
    value: MPInterval
    resolved: bool
    cells: int

def _point(value: float | int, precision: int) -> MPInterval:
    return MPInterval.point(value, precision)

def _clip(value: MPInterval, lower: float, upper: float) -> MPInterval:
    lo = max(value.lo, g.mpfr(lower, precision=1))
    hi = min(value.hi, g.mpfr(upper, precision=1))
    if lo > hi:
        raise NumericalUnresolved("interval does not intersect its structural bounds")
    return MPInterval(lo, hi, value.precision)

def _truncated_mass(
    mu: MPInterval,
    sigma: MPInterval,
    left: float,
    right: float,
) -> MPInterval:
    precision = mu.precision
    if sigma.lo <= 0:
        raise NumericalUnresolved("truncated-normal scale is not proved positive")
    denominator = normal_mass_interval(
        (_point(0.0, precision) - mu) / sigma,
        (_point(1.0, precision) - mu) / sigma,
    )
    if denominator.lo <= 0:
        raise NumericalUnresolved("truncated-normal normalizer is unresolved")
    numerator = normal_mass_interval(
        (_point(left, precision) - mu) / sigma,
        (_point(right, precision) - mu) / sigma,
    )
    return (numerator / denominator).maximum(0).minimum(1)

def _survival(
    threshold: float,
    mu: MPInterval,
    sigma: MPInterval,
) -> MPInterval:
    if threshold <= 0.0:
        return _point(1, mu.precision)
    if threshold >= 1.0:
        return _point(0, mu.precision)
    return _truncated_mass(mu, sigma, threshold, 1.0)

def bound_truncated_normal_ei(
    mu: MPInterval,
    sigma: MPInterval,
    incumbent: MPInterval,
    *,
    tolerance: float,
    max_cells: int,
) -> _EIEnclosure:
    """Enclose EI through monotone survival-integral Riemann sums."""

    precision = mu.precision
    c_lo = max(0.0, float(incumbent.to_float()[0]))
    c_hi = min(1.0, float(incumbent.to_float()[1]))
    if c_lo > c_hi:
        raise NumericalUnresolved("incumbent interval misses [0,1]")
    if c_lo >= 1.0:
        return _EIEnclosure(_point(0, precision), True, 0)
    threshold = incumbent.maximum(0).minimum(1)
    denominator = normal_mass_interval(
        (_point(0.0, precision) - mu) / sigma,
        (_point(1.0, precision) - mu) / sigma,
    )
    if denominator.lo <= 0:
        raise NumericalUnresolved("truncated-normal EI normalizer is unresolved")
    standardized_threshold = (threshold - mu) / sigma
    standardized_upper = (_point(1.0, precision) - mu) / sigma
    tail_mass = normal_mass_interval(standardized_threshold, standardized_upper)
    closed_numerator = sigma * (
        normal_pdf_interval(standardized_threshold)
        - normal_pdf_interval(standardized_upper)
        - standardized_threshold * tail_mass
    )
    support_upper = math.nextafter(max(0.0, 1.0 - c_lo), math.inf)
    closed = _clip(closed_numerator / denominator, 0.0, support_upper)
    closed_bounds = closed.to_float()
    if closed_bounds[1] - closed_bounds[0] <= tolerance:
        return _EIEnclosure(closed, True, 0)

    points = sorted(set((c_lo, c_hi, 1.0)))
    cells = [(left, right) for left, right in zip(points, points[1:]) if left < right]
    if not cells:
        return _EIEnclosure(_point(0, precision), True, 0)
    if len(cells) > max_cells:
        return _EIEnclosure(closed, False, 0)
    cache: dict[float, MPInterval] = {}

    def survival(value: float) -> MPInterval:
        if value not in cache:
            cache[value] = _survival(value, mu, sigma)
        return cache[value]

    def totals() -> tuple[MPInterval, list[float]]:
        lower = _point(0, precision)
        upper = _point(0, precision)
        gaps = []
        for left, right in cells:
            width = _point(right, precision) - _point(left, precision)
            lower_piece = (
                _point(0, precision)
                if right <= c_hi
                else width * MPInterval(
                    survival(right).lo, survival(right).lo, precision
                )
            )
            upper_piece = width * MPInterval(
                survival(left).hi, survival(left).hi, precision
            )
            lower = lower + lower_piece
            upper = upper + upper_piece
            gaps.append(max(0.0, (upper_piece - lower_piece).to_float()[1]))
        riemann = _clip(
            MPInterval(lower.lo, upper.hi, precision), 0.0, support_upper
        )
        intersection_lower = max(riemann.lo, closed.lo)
        intersection_upper = min(riemann.hi, closed.hi)
        if intersection_lower > intersection_upper:
            raise NumericalUnresolved("independent EI enclosures have empty intersection")
        return MPInterval(intersection_lower, intersection_upper, precision), gaps

    while True:
        value, gaps = totals()
        if value.to_float()[1] - value.to_float()[0] <= tolerance:
            return _EIEnclosure(value, True, len(cells))
        if len(cells) >= max_cells:
            return _EIEnclosure(value, False, len(cells))
        index = max(range(len(cells)), key=lambda item: gaps[item])
        left, right = cells[index]
        midpoint = left + (right - left) * 0.5
        if not left < midpoint < right:
            return _EIEnclosure(value, False, len(cells))
        cells[index:index + 1] = [(left, midpoint), (midpoint, right)]
