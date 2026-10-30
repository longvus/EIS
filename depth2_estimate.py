"""Two-step (depth-two) advantage estimates and the quadrature rule of the point planner."""

from __future__ import annotations

import copy

from dataclasses import asdict, dataclass, fields

from functools import lru_cache

import math

import numpy as np

from numpy.polynomial.legendre import leggauss

from scipy.special import erf, log_ndtr

from gp_types import AffineFantasy, SurrogateNumericalError

from surrogate_api import _logdiffexp, _truncnorm_interval_excess

@dataclass(frozen=True)
class SharedDepth2RootEstimate:
    candidate_id: str
    arm_position: int
    history_length: int
    can_take_second_step: bool
    child_candidate_ids: tuple[str, ...]
    root_ei: float
    root_net: float
    pooled_child_premium: float
    same_arm_child_premium: float
    pooled_advantage: float
    same_arm_advantage: float
    pooled_lower_stability_endpoint: float
    pooled_upper_stability_endpoint: float
    same_arm_lower_stability_endpoint: float
    same_arm_upper_stability_endpoint: float
    order_values: tuple[tuple[int, float, float], ...]

    def to_dict(self) -> dict[str, object]:
        result = asdict(self)
        result["child_candidate_ids"] = list(self.child_candidate_ids)
        result["order_values"] = [
            {
                "order": order,
                "pooled_child_premium": pooled,
                "same_arm_child_premium": same_arm,
            }
            for order, pooled, same_arm in self.order_values
        ]
        return result

    @classmethod
    def from_dict(cls, payload: dict[str, object]) -> "SharedDepth2RootEstimate":
        if type(payload) is not dict or set(payload) != {field.name for field in fields(cls)}:
            raise ValueError("shared depth-two root schema differs")
        child_ids = payload["child_candidate_ids"]
        orders = payload["order_values"]
        if not isinstance(child_ids, list) or not isinstance(orders, list):
            raise ValueError("shared depth-two root vectors differ")
        parsed_orders = []
        for item in orders:
            if type(item) is not dict or set(item) != {
                "order",
                "pooled_child_premium",
                "same_arm_child_premium",
            }:
                raise ValueError("shared depth-two quadrature schema differs")
            parsed_orders.append((
                item["order"],
                item["pooled_child_premium"],
                item["same_arm_child_premium"],
            ))
        values = copy.deepcopy(payload)
        values["child_candidate_ids"] = tuple(child_ids)
        values["order_values"] = tuple(parsed_orders)
        return cls(**values)

@dataclass(frozen=True)
class SharedDepth2Estimate:
    lambda_: float
    incumbent: float
    root_ei_max: float | None
    root_ei_best_candidate_id: str | None
    boundary_point_satisfied: bool
    selected_candidate_id: str | None
    pooled_stop_regret_estimate: float
    selected_root_same_arm_stop_regret_estimate: float
    pooled_lower_stability_endpoint: float
    pooled_upper_stability_endpoint: float
    numerically_positive: bool
    numerically_nonpositive: bool
    root_estimates: tuple[SharedDepth2RootEstimate, ...]
    metadata: dict[str, object]

    def to_dict(self) -> dict[str, object]:
        result = asdict(self)
        result["root_estimates"] = [estimate.to_dict() for estimate in self.root_estimates]
        return result

    @classmethod
    def from_dict(cls, payload: dict[str, object]) -> "SharedDepth2Estimate":
        if type(payload) is not dict or set(payload) != {field.name for field in fields(cls)}:
            raise ValueError("shared depth-two estimate schema differs")
        roots = payload["root_estimates"]
        metadata = payload["metadata"]
        if not isinstance(roots, list) or type(metadata) is not dict:
            raise ValueError("shared depth-two estimate vectors differ")
        values = copy.deepcopy(payload)
        values["root_estimates"] = tuple(SharedDepth2RootEstimate.from_dict(item) for item in roots)
        return cls(**values)

@lru_cache(maxsize=32)
def _quadrature_rule(order: int) -> tuple[np.ndarray, np.ndarray]:
    nodes, weights = leggauss(order)
    levels, weights = (nodes + 1.0) * 0.5, weights * 0.5
    levels.setflags(write=False)
    weights.setflags(write=False)
    return levels, weights

def _child_ei_batch(
    affine: AffineFantasy, fantasies: np.ndarray, incumbent: float,
) -> np.ndarray:
    """Batch special functions, retaining scalar EI's libm and tail arithmetic."""
    if not np.all(np.isfinite(fantasies)):
        raise ValueError("fantasy must be finite")
    lo, hi = affine.support_lo, affine.support_hi
    if np.any((fantasies < lo) | (fantasies > hi)):
        raise ValueError("fantasy lies outside the bounded planning support")
    variances = np.diag(affine.child_covariance())
    if np.any(variances < affine.sigma_floor**2):
        raise SurrogateNumericalError(
            "child observation variance is below the frozen distribution floor"
        )
    scales = np.array([math.sqrt(float(value)) for value in variances])
    means = affine.child_base_mean + affine.child_mean_slope * (
        fantasies[:, None] - affine.root_mean
    )
    best = np.broadcast_to(np.maximum(incumbent, fantasies)[:, None], means.shape)
    scales = np.broadcast_to(scales, means.shape)
    result = np.zeros(means.shape)
    regular = np.isfinite(scales) & (scales > 0.0)
    for index in zip(*np.nonzero(~regular)):
        result[index] = max(min(max(float(means[index]), lo), hi) - float(best[index]), 0.0)
    live = regular & (best < hi)
    m, s, b = means[live], scales[live], best[live]
    lower, upper = (lo - m) / s, (hi - m) / s
    gain_lower = (np.maximum(b, lo) - m) / s
    values = np.maximum(lo - b, 0.0)
    gain = gain_lower < upper

    # Match _log_gaussian_mass branch-for-branch. Only scipy's elementwise
    # special functions are batched; numpy exp/log can differ from scalar libm.
    left = np.stack((lower[gain], gain_lower[gain]))
    right = np.broadcast_to(upper[gain], left.shape)
    negative, positive = right <= 0.0, left >= 0.0
    tail = negative | positive
    x = np.where(negative[tail], right[tail], -left[tail])
    y = np.where(negative[tail], left[tail], -right[tail])
    log_mass = np.empty(left.shape)
    log_mass[tail] = [
        _logdiffexp(float(a), float(b)) for a, b in zip(log_ndtr(x), log_ndtr(y))
    ]
    root_two = math.sqrt(2.0)
    log_mass[~tail] = [
        math.log(0.5 * (float(a) + float(b)))
        for a, b in zip(erf(-left[~tail] / root_two), erf(right[~tail] / root_two))
    ]
    for index, denominator, numerator in zip(np.flatnonzero(gain), *log_mass):
        tail_ratio = math.exp(min(0.0, float(numerator) - float(denominator)))
        tail_excess = float(s[index]) * _truncnorm_interval_excess(
            float(gain_lower[index]), float(upper[index])
        )
        value = max(lo - float(b[index]), 0.0) + tail_ratio * tail_excess
        values[index] = min(max(hi - float(b[index]), 0.0), max(value, 0.0))
    result[live] = values
    return result
