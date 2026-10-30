"""Fixed finite-basis Gaussian learning-curve surrogates (GP-a, GP-at, GP-atJ), conditioned on the pool.

The only observation-dependent operation is Gaussian conditioning; kernel parameters are fixed.
"""

from __future__ import annotations

from functools import lru_cache

import math

import numpy as np

from scipy.linalg import cho_factor, cho_solve

from shared_temporal_design import SharedTemporalSnapshot, WEIGHTS, sha

from gp_types import SurrogateQuery, JointPrediction, AffineFantasy

from surrogate_api import TruncGaussDist

class SharedTemporalNumericalError(RuntimeError):
    pass

def readonly(value):
    result = np.array(value, dtype=np.float64, copy=True)
    result.setflags(write=False)
    return result

def basis(epochs, snapshot):
    t = np.asarray(epochs, dtype=float).reshape(-1)
    if np.any(~np.isfinite(t)) or np.any(t < 1):
        raise ValueError("epochs must be finite and at least one")
    scales = np.asarray(snapshot.basis_scales, dtype=float)
    return np.column_stack((np.ones(len(t)), (scales + 1) / (t[:, None] + scales)))

def noise(epochs, snapshot):
    t = np.asarray(epochs, dtype=float)
    if np.any(~np.isfinite(t)) or np.any(t < 1):
        raise ValueError("epochs must be finite and at least one")
    return snapshot.noise_base + snapshot.noise_decay / (1 + t)**2 + snapshot.jitter

def coefficient_prior(features, snapshot):
    n = len(features)
    delta = (features[:, None] - features[None, :]) / np.asarray(snapshot.lengthscales)
    z = np.sqrt(5. * np.sum(delta * delta, axis=-1))
    matern = (1 + z + z*z/3) * np.exp(-z)
    matrices = []
    for weights in WEIGHTS[snapshot.mode]:
        matrices.append(weights[0] * np.eye(n) + weights[1] * matern + weights[2] * np.ones((n, n)))
    result = np.zeros((5*n, 5*n))
    for r in range(5):
        positions = np.arange(n) * 5 + r
        variance = snapshot.asymptote_variance if r == 0 else snapshot.temporal_variance / 4
        result[np.ix_(positions, positions)] = variance * matrices[r != 0]
    return result

@lru_cache(maxsize=16)
def _prior_cache(snapshot, feature_rows):
    prior = coefficient_prior(np.asarray(feature_rows), snapshot)
    precision = cho_solve(cho_factor(prior, lower=True), np.eye(len(prior)))
    return readonly(prior), readonly(precision)

class SharedTemporalPosterior:
    """Immutable public history; privately cached finite coefficient summaries."""

    def __init__(self, snapshot, candidate_ids, features, histories, *, epochs=None):
        if not isinstance(snapshot, SharedTemporalSnapshot):
            raise TypeError("SharedTemporalSnapshot required")
        ids = tuple(candidate_ids)
        if not ids or any(not isinstance(i, str) or not i for i in ids) or len(set(ids)) != len(ids):
            raise ValueError("unique nonempty candidate IDs required")
        x = np.asarray(features, dtype=float)
        if x.shape != (len(ids), len(snapshot.lengthscales)) or not np.isfinite(x).all() or np.any(x < 0) or np.any(x > 1):
            raise ValueError("finite normalized feature matrix required")
        if len(histories) != len(ids) or (epochs is not None and len(epochs) != len(ids)):
            raise ValueError("histories/epochs must align with candidates")
        hs, ts = [], []
        for arm, raw in enumerate(histories):
            values = tuple(float(v) for v in raw)
            times = tuple(float(t) for t in (range(1, len(values)+1) if epochs is None else epochs[arm]))
            if len(times) != len(values) or any(not math.isfinite(t) or t < 1 for t in times) or any(b <= a for a, b in zip(times, times[1:])):
                raise ValueError("epochs must be increasing and align with histories")
            if any(not math.isfinite(v) or not 0 <= v <= 1 for v in values):
                raise ValueError("scores outside [0,1]")
            hs.append(values)
            ts.append(times)
        self.snapshot, self.candidate_ids = snapshot, ids
        self.features, self.histories, self.epochs = readonly(x), tuple(hs), tuple(ts)
        self._index = {v: i for i, v in enumerate(ids)}
        self._updates = 0
        self._initialize()

    @property
    def arms_are_autonomous(self):
        return self.snapshot.arms_are_autonomous

    def _initialize(self):
        prior, precision = _prior_cache(self.snapshot, tuple(map(tuple, self.features)))
        self.prior_covariance = prior
        p = precision.copy()
        h = np.zeros(len(p))
        for arm, (times, values) in enumerate(zip(self.epochs, self.histories)):
            if not values:
                continue
            phi = basis(times, self.snapshot)
            inv_noise = 1 / noise(times, self.snapshot)
            sl = slice(arm*5, arm*5+5)
            p[sl, sl] += phi.T @ (inv_noise[:, None] * phi)
            h[sl] += phi.T @ (inv_noise * (np.asarray(values) - self.snapshot.mean))
        if not any(self.histories):
            covariance, mean = prior, h
        else:
            factor = cho_factor(p, lower=True)
            mean = cho_solve(factor, h)
            covariance = cho_solve(factor, np.eye(len(p)))
        self.coefficient_precision, self.coefficient_rhs = readonly(p), readonly(h)
        self.coefficient_mean = readonly(mean)
        self.coefficient_covariance = readonly((covariance + covariance.T) / 2)

    def query_features(self, queries):
        queries = tuple(queries)
        design = np.zeros((len(queries), len(self.candidate_ids)*5))
        for row, query in enumerate(queries):
            if not isinstance(query, SurrogateQuery) or query.candidate_id not in self._index:
                raise ValueError("invalid query")
            arm = self._index[query.candidate_id]
            design[row, 5*arm:5*arm+5] = basis([query.epoch], self.snapshot)[0]
        return design

    def joint_parameters(self, queries):
        queries = tuple(queries)
        design = self.query_features(queries)
        mean = self.snapshot.mean + design @ self.coefficient_mean
        latent = design @ self.coefficient_covariance @ design.T
        latent = (latent + latent.T) / 2
        if latent.size and np.min(np.diag(latent)) < -1e-12:
            raise SharedTemporalNumericalError("negative latent variance")
        observed = latent.copy()
        observed[np.diag_indices(len(queries))] += noise([q.epoch for q in queries], self.snapshot)
        if not np.isfinite(mean).all() or not np.isfinite(observed).all():
            raise SharedTemporalNumericalError("nonfinite query summary")
        return JointPrediction(queries, readonly(mean), readonly(latent), readonly(observed))

    def predict(self, queries):
        joint = self.joint_parameters(queries)
        values = np.diag(joint.observation_covariance)
        if np.any(values < self.snapshot.sigma_floor**2):
            raise SharedTemporalNumericalError("query scale below fixed floor")
        return [TruncGaussDist(mu=float(m), sigma=math.sqrt(float(v))) for m, v in zip(joint.mean, values)]

    def _check_new_query(self, query):
        self.query_features((query,))
        arm = self._index[query.candidate_id]
        if self.epochs[arm] and query.epoch <= self.epochs[arm][-1]:
            raise ValueError("observation must follow arm history")
        return arm

    def compile_fantasy(self, root_query, child_queries):
        root_arm = self._check_new_query(root_query)
        children = tuple(child_queries)
        for query in children:
            arm = self._check_new_query(query)
            if arm == root_arm and query.epoch <= root_query.epoch:
                raise ValueError("same-arm child must follow root")
        joint = self.joint_parameters((root_query, *children))
        var = float(joint.observation_covariance[0, 0])
        if var < self.snapshot.sigma_floor**2:
            raise SharedTemporalNumericalError("invalid root variance")
        cross = joint.latent_covariance[1:, 0]
        latent = joint.latent_covariance[1:, 1:] - np.outer(cross, cross) / var
        observed = joint.observation_covariance[1:, 1:] - np.outer(cross, cross) / var
        if np.any(np.diag(observed) < self.snapshot.sigma_floor**2):
            raise SharedTemporalNumericalError("invalid child variance")
        return AffineFantasy(root_query, children,
            TruncGaussDist(mu=float(joint.mean[0]), sigma=math.sqrt(var)), var,
            float(joint.mean[0]), readonly(joint.mean[1:]), readonly(cross/var),
            readonly(latent), readonly(observed), "observed", self.snapshot.sigma_floor, 0., 1.)

    def condition(self, query, value):
        arm = self._check_new_query(query)
        value = float(value)
        if not math.isfinite(value) or not 0 <= value <= 1:
            raise ValueError("observation outside [0,1]")
        histories, epochs = list(self.histories), list(self.epochs)
        histories[arm] = (*histories[arm], value)
        epochs[arm] = (*epochs[arm], query.epoch)
        # A predetermined rebuild limits floating accumulation; ideal summaries
        # of rank-one conditioning and batch conditioning are identical.
        if (self._updates + 1) % 128 == 0:
            return SharedTemporalPosterior(self.snapshot, self.candidate_ids, self.features, histories, epochs=epochs)
        result = object.__new__(SharedTemporalPosterior)
        result.__dict__ = self.__dict__.copy()
        result.histories, result.epochs = tuple(histories), tuple(epochs)
        result._updates = self._updates + 1
        phi = self.query_features((query,))[0]
        cross = self.coefficient_covariance @ phi
        nugget = float(noise(query.epoch, self.snapshot))
        var = float(phi @ cross + nugget)
        mu = self.snapshot.mean + float(phi @ self.coefficient_mean)
        result.coefficient_mean = readonly(self.coefficient_mean + cross * ((value - mu) / var))
        covariance = self.coefficient_covariance - np.outer(cross, cross) / var
        result.coefficient_covariance = readonly((covariance + covariance.T) / 2)
        result.coefficient_precision = readonly(self.coefficient_precision + np.outer(phi, phi) / nugget)
        result.coefficient_rhs = readonly(self.coefficient_rhs + phi * ((value-self.snapshot.mean) / nugget))
        return result

    def state_sha256(self, caps, remaining_allocations):
        return sha(dict(schema="shared-temporal-state/v1", snapshot=self.snapshot.to_dict(),
            candidate_ids=self.candidate_ids, features=self.features.tolist(), histories=self.histories,
            epochs=self.epochs, caps=list(caps), remaining_allocations=remaining_allocations))
