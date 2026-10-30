"""Query, joint-prediction and affine-fantasy types shared by the Gaussian surrogates and the certificate."""

from __future__ import annotations

from dataclasses import dataclass

import math

from typing import Literal

import numpy as np

from surrogate_api import TruncGaussDist

VarianceKind = Literal["latent", "observed"]

class SurrogateNumericalError(RuntimeError):
    """The frozen Gaussian law could not be evaluated numerically."""

def _finite_float(value: float, name: str) -> float:
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{name} must be finite")
    return result

def _positive_float(value: float, name: str) -> float:
    result = _finite_float(value, name)
    if result <= 0.0:
        raise ValueError(f"{name} must be positive")
    return result

@dataclass(frozen=True)
class SurrogateQuery:
    candidate_id: str
    epoch: float

    def __post_init__(self) -> None:
        if not isinstance(self.candidate_id, str) or not self.candidate_id:
            raise ValueError("candidate_id must be a nonempty string")
        epoch = _positive_float(self.epoch, "query epoch")
        object.__setattr__(self, "epoch", epoch)

@dataclass(frozen=True)
class JointPrediction:
    queries: tuple[SurrogateQuery, ...]
    mean: np.ndarray
    latent_covariance: np.ndarray
    observation_covariance: np.ndarray

    def covariance(self, kind: VarianceKind) -> np.ndarray:
        if kind == "latent":
            return self.latent_covariance
        if kind == "observed":
            return self.observation_covariance
        raise ValueError("kind must be 'latent' or 'observed'")

@dataclass(frozen=True)
class AffineFantasy:
    """Root law and all-child Gaussian summaries after a scalar fantasy."""

    root_query: SurrogateQuery
    child_queries: tuple[SurrogateQuery, ...]
    root_distribution: TruncGaussDist
    root_update_variance: float
    root_mean: float
    child_base_mean: np.ndarray
    child_mean_slope: np.ndarray
    child_latent_covariance: np.ndarray
    child_observation_covariance: np.ndarray
    decision_variance: VarianceKind
    sigma_floor: float
    support_lo: float
    support_hi: float

    def child_mean(self, fantasy: float) -> np.ndarray:
        value = _finite_float(fantasy, "fantasy")
        if value < self.support_lo or value > self.support_hi:
            raise ValueError("fantasy lies outside the bounded planning support")
        return self.child_base_mean + self.child_mean_slope * (value - self.root_mean)

    def child_covariance(self, kind: VarianceKind | None = None) -> np.ndarray:
        selected = self.decision_variance if kind is None else kind
        if selected == "latent":
            return self.child_latent_covariance
        if selected == "observed":
            return self.child_observation_covariance
        raise ValueError("kind must be 'latent' or 'observed'")

    def child_distributions(
        self,
        fantasy: float,
        kind: VarianceKind | None = None,
    ) -> list[TruncGaussDist]:
        means = self.child_mean(fantasy)
        covariance = self.child_covariance(kind)
        distributions = []
        for index in range(len(self.child_queries)):
            variance = float(covariance[index, index])
            if variance < self.sigma_floor**2:
                raise SurrogateNumericalError(
                    "child observation variance is below the frozen distribution floor"
                )
            distributions.append(
                TruncGaussDist(
                    mu=float(means[index]),
                    sigma=math.sqrt(variance),
                    lo=self.support_lo,
                    hi=self.support_hi,
                )
            )
        return distributions
