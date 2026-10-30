"""
surrogate_api.py
─────────────────────────────────────────────────────────────────────────────
Bucket-distribution interface and surrogate models for the LSMC/PB pipeline.
Provides BucketDist and FTPFNSurrogate (FT-PFN, ifBO).

Design principles
─────────────────
  1. Pure functions. No mutation of external state.
  2. Expose the full categorical posterior, not just its moments.
  3. Mirror predict()'s effective_cutoff handling so inference stays
     in-distribution relative to training.
  4. Batched: one forward pass handles many (history, query_epoch) pairs.

The core object is BucketDist — an immutable container holding the B-bucket
posterior weights plus the edges they refer to. Moments (mean, std),
quantiles, CDF, and inverse-CDF are all derived from this.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

import math

import torch

from scipy.special import erf, log_ndtr, ndtr, ndtri_exp

from scipy.stats import truncnorm

_LOG_HALF = math.log(0.5)

_LOG_2PI = math.log(2.0 * math.pi)

_SQRT_TWO = math.sqrt(2.0)

def _logdiffexp(log_x: float, log_y: float) -> float:
    """Return ``log(exp(log_x) - exp(log_y))`` for ``log_x > log_y``."""
    delta = min(log_y - log_x, math.nextafter(0.0, -math.inf))
    if delta < _LOG_HALF:
        return log_x + math.log1p(-math.exp(delta))
    return log_x + math.log(-math.expm1(delta))

def _log_gaussian_mass(lower: float, upper: float) -> float:
    """Stable log probability of a standard Gaussian interval."""
    if upper <= 0.0:
        return _logdiffexp(float(log_ndtr(upper)), float(log_ndtr(lower)))
    if lower >= 0.0:
        return _logdiffexp(float(log_ndtr(-lower)), float(log_ndtr(-upper)))
    mass = 0.5 * (
        float(erf(-lower / _SQRT_TWO)) + float(erf(upper / _SQRT_TWO))
    )
    return math.log(mass)

def _positive_truncnorm_excess(lower: float, upper: float) -> float:
    """Return E[Z-lower | lower < Z < upper] for nonnegative bounds."""
    width = upper - lower
    scaled_span = lower * width + 0.5 * width * width

    # A local series avoids subtracting endpoint densities of order 1/width.
    if scaled_span <= 4.0:
        aw = lower * width
        width2 = width * width
        mass_sum = 1.0
        moment_sum = 0.5
        previous, term = 1.0, -aw
        for order in range(1, 100):
            mass_add = term / (order + 1)
            moment_add = term / (order + 2)
            mass_sum += mass_add
            moment_sum += moment_add
            if (
                order >= 4
                and abs(term) <= 2e-16 * max(abs(mass_sum), 1.0)
                and abs(previous) <= 2e-16 * max(abs(moment_sum), 1.0)
            ):
                break
            following = -(aw * term + width2 * previous) / (order + 1)
            previous, term = term, following
        return min(width, max(0.0, width * moment_sum / mass_sum))

    # Once the upper endpoint is negligible, the inverse-Mills expansion
    # evaluates the small excess directly instead of subtracting two O(lower)
    # quantities. The threshold keeps the asymptotic series well conditioned.
    if lower >= 20.0 and scaled_span >= 36.0:
        inverse = 1.0 / lower
        inverse2 = inverse * inverse
        series = (
            1.0
            - 2.0 * inverse2
            + 10.0 * inverse2**2
            - 74.0 * inverse2**3
            + 706.0 * inverse2**4
            - 8162.0 * inverse2**5
            + 110410.0 * inverse2**6
        )
        return min(width, max(0.0, inverse * series))

    log_mass = _log_gaussian_mass(lower, upper)
    density_lower = math.exp(-0.5 * lower * lower - 0.5 * _LOG_2PI - log_mass)
    density_upper = math.exp(-0.5 * upper * upper - 0.5 * _LOG_2PI - log_mass)
    return min(width, max(0.0, density_lower - density_upper - lower))

def _truncnorm_interval_excess(lower: float, upper: float) -> float:
    """Return E[Z-lower | lower < Z < upper] without tail cancellation."""
    if lower >= 0.0:
        return _positive_truncnorm_excess(lower, upper)
    if upper <= 0.0:
        reflected = _positive_truncnorm_excess(-upper, -lower)
        return min(upper - lower, max(0.0, upper - lower - reflected))

    log_mass = _log_gaussian_mass(lower, upper)
    density_lower = math.exp(-0.5 * lower * lower - 0.5 * _LOG_2PI - log_mass)
    density_upper = math.exp(-0.5 * upper * upper - 0.5 * _LOG_2PI - log_mass)
    mean = density_lower - density_upper
    return min(upper - lower, max(0.0, mean - lower))

_CRPS_NODES, _CRPS_WEIGHTS = np.polynomial.legendre.leggauss(128)

_CRPS_LEVELS = (_CRPS_NODES + 1.0) / 2.0

_CRPS_WEIGHTS = _CRPS_WEIGHTS / 2.0

@dataclass(frozen=True)
class TruncGaussDist:
    """Deployed one-step law: the Gaussian truncated to ``[lo, hi]``.

    All planning and calibration operations use the continuous law. Optional
    bucket fields are accepted for compatibility; the deployed path does not
    construct them.

    ``mu`` and ``sigma`` are the exact predictive parameters, carried from the GP
    rather than recovered from the weights.
    """

    mu:        float = 0.0
    sigma:     float = 1.0
    lo:        float = 0.0
    hi:        float = 1.0
    probs:     torch.Tensor | None = field(default=None, repr=False, compare=False)
    edges:     torch.Tensor | None = field(default=None, repr=False, compare=False)
    midpoints: torch.Tensor | None = field(default=None, repr=False, compare=False)

    def _standard_bounds(self) -> tuple[float, float, float]:
        sd = float(self.sigma)
        lower = (float(self.lo) - float(self.mu)) / sd
        upper = (float(self.hi) - float(self.mu)) / sd
        return lower, upper, _log_gaussian_mass(lower, upper)

    def moments(self) -> tuple[float, float]:
        """Exact continuous mean and standard deviation."""
        sd = float(self.sigma)
        lo, hi = float(self.lo), float(self.hi)
        if not math.isfinite(sd) or sd <= 0.0:
            return min(max(float(self.mu), lo), hi), 0.0
        lower, upper, _ = self._standard_bounds()
        mean_z, var_z = truncnorm.stats(lower, upper, moments="mv")
        mean = min(max(float(self.mu) + sd * float(mean_z), lo), hi)
        std = sd * math.sqrt(max(float(var_z), 0.0))
        return mean, min(std, (hi - lo) / 2.0)

    def mean(self) -> float:
        return self.moments()[0]

    def variance(self) -> float:
        std = self.moments()[1]
        return std * std

    def std(self) -> float:
        return self.moments()[1]

    def ei(self, best_f: float) -> float:
        s = float(self.sigma)
        m, lo, hi = float(self.mu), float(self.lo), float(self.hi)
        best = float(best_f)
        if not math.isfinite(s) or s <= 0.0:
            return max(min(max(m, lo), hi) - best, 0.0)
        if best >= hi:
            return 0.0

        threshold = max(best, lo)
        lower = (lo - m) / s
        upper = (hi - m) / s
        gain_lower = (threshold - m) / s
        if gain_lower >= upper:
            return max(lo - best, 0.0)

        log_denominator = _log_gaussian_mass(lower, upper)
        log_tail_mass = _log_gaussian_mass(gain_lower, upper)
        tail_ratio = math.exp(min(0.0, log_tail_mass - log_denominator))
        tail_excess = s * _truncnorm_interval_excess(gain_lower, upper)
        value = max(lo - best, 0.0) + tail_ratio * tail_excess
        return float(min(max(hi - best, 0.0), max(value, 0.0)))

    def cdf(self, value: float) -> float:
        value = float(value)
        if value <= self.lo:
            return 0.0
        if value >= self.hi:
            return 1.0
        lower, upper, log_denominator = self._standard_bounds()
        z = (value - float(self.mu)) / float(self.sigma)
        return float(min(1.0, max(0.0, math.exp(
            _log_gaussian_mass(lower, z) - log_denominator
        ))))

    def pi(self, best_f: float) -> float:
        best = float(best_f)
        if best < self.lo:
            return 1.0
        if best >= self.hi:
            return 0.0
        _, upper, log_denominator = self._standard_bounds()
        z = (best - float(self.mu)) / float(self.sigma)
        return float(min(1.0, max(0.0, math.exp(
            _log_gaussian_mass(z, upper) - log_denominator
        ))))

    def prob_greater_than(self, threshold: float) -> float:
        return self.pi(threshold)

    def log_prob(self, value: float) -> float:
        """Continuous truncated-Gaussian log density."""
        value = float(value)
        if value < self.lo or value > self.hi:
            return -math.inf
        sd = float(self.sigma)
        if not math.isfinite(sd) or sd <= 0.0:
            return 0.0 if value == min(max(float(self.mu), self.lo), self.hi) else -math.inf
        _, _, log_mass = self._standard_bounds()
        z = (value - float(self.mu)) / sd
        return -0.5 * z * z - 0.5 * _LOG_2PI - math.log(sd) - log_mass

    def crps(self, value: float) -> float:
        """CRPS of the continuous truncated Gaussian.

        The ordinary branch uses the closed-form Gini term. Extreme tails fall
        back to fixed Gauss-Legendre integration over the exact inverse CDF to
        avoid cancellation; this quadrature is not an experiment hyperparameter.
        """
        value = float(value)
        mean = self.mean()
        absolute_error = 2.0 * self.ei(value) + value - mean
        lower, upper, log_mass = self._standard_bounds()
        half_pair = math.nan
        if log_mass > -20.0:
            mass = math.exp(log_mass)
            phi_lower = math.exp(-0.5 * lower * lower - 0.5 * _LOG_2PI)
            phi_upper = math.exp(-0.5 * upper * upper - 0.5 * _LOG_2PI)
            root_mass = float(ndtr(_SQRT_TWO * upper) - ndtr(_SQRT_TWO * lower))
            numerator = (
                root_mass / math.sqrt(math.pi)
                - mass * (phi_lower + phi_upper)
            )
            candidate = float(self.sigma) * numerator / (mass * mass)
            if math.isfinite(candidate) and 0.0 <= candidate <= (self.hi - self.lo) / 2.0:
                half_pair = candidate
        if not math.isfinite(half_pair):
            quantiles = self.quantiles(_CRPS_LEVELS)
            half_pair = float(np.sum(
                _CRPS_WEIGHTS * quantiles * (2.0 * _CRPS_LEVELS - 1.0)
            ))
        return float(max(absolute_error - half_pair, 0.0))

    def quantiles(self, p) -> np.ndarray:
        """Vectorized inverse CDF of the continuous truncated-Gaussian law.

        ``BucketDist.quantile`` is intentionally an inverse CDF for its
        piecewise-uniform bucket approximation.  This subclass evaluates EI
        and samples from the continuous truncated Gaussian, so rollout fantasy
        nodes must use the same law rather than silently switching models.
        """

        lo, hi = float(self.lo), float(self.hi)
        p = np.clip(np.asarray(p, dtype=np.float64), 0.0, 1.0)
        sd = float(self.sigma)
        if not math.isfinite(sd) or sd <= 0.0:
            return np.full_like(p, min(max(float(self.mu), lo), hi))
        m = float(self.mu)
        lower = (lo - m) / sd
        upper = (hi - m) / sd
        z = np.empty_like(p)
        z[p <= 0.0] = lower
        z[p >= 1.0] = upper
        interior = (p > 0.0) & (p < 1.0)
        if np.any(interior):
            levels = p[interior]
            log_mass = _log_gaussian_mass(lower, upper)
            if lower < 0.0:
                log_cdf = np.logaddexp(
                    float(log_ndtr(lower)), np.log(levels) + log_mass
                )
                z[interior] = ndtri_exp(log_cdf)
            else:
                log_survival = np.logaddexp(
                    float(log_ndtr(-upper)), np.log1p(-levels) + log_mass
                )
                z[interior] = -ndtri_exp(log_survival)
        return np.clip(m + sd * z, lo, hi)

    def quantile(self, p: float) -> float:
        """Inverse CDF of the continuous truncated-Gaussian planning law."""

        return float(self.quantiles(np.asarray([p], dtype=np.float64))[0])

    def sample(self, n: int, rng: np.random.Generator) -> np.ndarray:
        """Inverse-CDF draws from the truncated Gaussian itself.

        The inherited piecewise-uniform sampler draws from the discretization of
        this law rather than the law, which would leave the value-function
        simulator and the acquisition disagreeing at ``O(h^2)``. Sampling the
        exact law keeps a single transition model throughout.
        """
        return self.quantiles(rng.uniform(size=n))

    def stratified_sample(self, n: int, rng: np.random.Generator) -> np.ndarray:
        levels = (np.arange(n, dtype=np.float64) + rng.random(n)) / n
        return self.quantiles(levels)

    def expect(self, f_at_midpoints: torch.Tensor) -> float:
        if self.probs is None:
            raise RuntimeError(
                "bucket expectations are unavailable for the continuous planning law"
            )
        return float((self.probs * f_at_midpoints).sum())

    def top_c(self, c: int) -> tuple[torch.Tensor, torch.Tensor, float]:
        if self.probs is None:
            raise RuntimeError(
                "top-c bucket pruning is unavailable for the continuous planning law"
            )
        c = min(c, self.probs.numel())
        top_vals, top_idx = torch.topk(self.probs, c)
        return top_idx, top_vals / top_vals.sum(), float(1.0 - top_vals.sum())
