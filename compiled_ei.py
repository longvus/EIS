"""Optional compiled EI kernel (build with build_compiled_ei.py); hashes and ABI are checked at import."""

import hashlib

import importlib.util

import json

import math

from pathlib import Path

import sysconfig

import numpy as np

import scipy

from scipy.special import erf, log_ndtr

from gp_types import AffineFantasy, SurrogateNumericalError

def _load_kernel(root):
    build = root / 'build' / 'compiled_ei'
    name = '_compiled_ei_kernel'
    binary = build / (name + sysconfig.get_config_var('EXT_SUFFIX'))
    try:
        manifest = json.loads((build / 'manifest.json').read_text())
        if (manifest['schema'] != 1 or manifest['binary'] != binary.name
                or manifest['soabi'] != sysconfig.get_config_var('SOABI')
                or manifest['numpy'] != np.__version__ or manifest['scipy'] != scipy.__version__
                or manifest['flags'] != ['-O2', '-shared', '-fPIC', '-fno-fast-math', '-ffp-contract=off', '-fno-builtin-pow']):
            raise ImportError('compiled EI ABI/runtime/build mismatch')
        for path in (root / 'compiled_ei.py', root / 'compiled_ei_kernel.c', binary):
            if hashlib.sha256(path.read_bytes()).hexdigest() != manifest['sha256'][path.name]:
                raise ImportError('compiled EI hash mismatch: ' + str(path))
        spec = importlib.util.spec_from_file_location(name, binary.resolve())
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module
    except (OSError, ValueError, KeyError, TypeError) as error:
        raise ImportError('compiled EI artifact missing or invalid; build offline') from error

_kernel = _load_kernel(Path(__file__).resolve().parent)

def child_ei_batch(
    affine: AffineFantasy, fantasies: np.ndarray, incumbent: float,
) -> np.ndarray:
    """Same API/guards as the frozen planner's _child_ei_batch; no fallback."""
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

    # Preserve the reference's SciPy vector operations and scalar libm order.
    left = np.stack((lower[gain], gain_lower[gain]))
    right = np.broadcast_to(upper[gain], left.shape)
    negative, positive = right <= 0.0, left >= 0.0
    tail = negative | positive
    x = np.where(negative[tail], right[tail], -left[tail])
    y = np.where(negative[tail], left[tail], -right[tail])
    log_mass = np.empty(left.shape)
    log_mass[tail] = np.frombuffer(_kernel.mass(
        np.ascontiguousarray(log_ndtr(x), dtype=np.float64),
        np.ascontiguousarray(log_ndtr(y), dtype=np.float64),
        np.ones(x.shape, dtype=np.float64)), dtype=np.float64)
    root_two = math.sqrt(2.0)
    a, bb = erf(-left[~tail] / root_two), erf(right[~tail] / root_two)
    log_mass[~tail] = np.frombuffer(_kernel.mass(
        np.ascontiguousarray(a, dtype=np.float64),
        np.ascontiguousarray(bb, dtype=np.float64),
        np.zeros(a.shape, dtype=np.float64)), dtype=np.float64)
    values[gain] = np.frombuffer(_kernel.inner(
        *(np.ascontiguousarray(v, dtype=np.float64) for v in
          (log_mass[0], log_mass[1], s[gain], gain_lower[gain], upper[gain], b[gain])),
        np.full(np.count_nonzero(gain), lo, dtype=np.float64),
        np.full(np.count_nonzero(gain), hi, dtype=np.float64)), dtype=np.float64)
    result[live] = values
    return result
