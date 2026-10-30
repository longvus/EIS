"""Verified interval linear algebra in MPFR: Cholesky factorization, triangular solves and dot products."""

from __future__ import annotations

from functools import lru_cache

from typing import Sequence

import gmpy2 as g

from mpfr_interval import MPInterval

@lru_cache(maxsize=None)
def _contexts(precision: int) -> tuple[g.context, g.context]:
    if precision < 80:
        raise ValueError("MPFR interval precision must be at least 80 bits")
    common = {
        "precision": precision,
        "subnormalize": False,
        "allow_complex": False,
    }
    return (
        g.context(round=g.RoundDown, **common),
        g.context(round=g.RoundUp, **common),
    )

def _mass_points(left: g.mpfr, right: g.mpfr, precision: int) -> MPInterval:
    """Stable standard-normal mass between two exact MPFR points."""

    if left >= right:
        return MPInterval.point(0, precision)
    half = MPInterval.point(0.5, precision)
    one = MPInterval.point(1, precision)
    sqrt_two = MPInterval.point(2, precision).sqrt()
    left_iv = MPInterval(left, left, precision)
    right_iv = MPInterval(right, right, precision)
    zero = g.mpfr(0)
    if right <= zero:
        result = half * (
            ((-right_iv) / sqrt_two).erfc()
            - ((-left_iv) / sqrt_two).erfc()
        )
    elif left >= zero:
        result = half * (
            (left_iv / sqrt_two).erfc()
            - (right_iv / sqrt_two).erfc()
        )
    else:
        result = (
            one
            - half * (right_iv / sqrt_two).erfc()
            - half * ((-left_iv) / sqrt_two).erfc()
        )
    return result.maximum(0).minimum(1)

def normal_mass_interval(left: MPInterval, right: MPInterval) -> MPInterval:
    """Enclose nonnegative Gaussian mass for ordered, possibly overlapping bounds.

    A definitely reversed pair is rejected.  If endpoint uncertainty leaves the
    order unresolved, the lower mass is zero and the upper mass spans the widest
    ordered endpoint pair.
    """

    if left.precision != right.precision:
        raise ValueError("mixed MPFR interval precisions")
    if left.lo > right.hi:
        raise ValueError("normal-mass bounds are definitely reversed")
    precision = left.precision
    lower = (
        MPInterval.point(0, precision)
        if left.hi >= right.lo
        else _mass_points(left.hi, right.lo, precision)
    )
    upper = _mass_points(left.lo, right.hi, precision)
    return MPInterval(lower.lo, upper.hi, precision).maximum(0).minimum(1)

def normal_pdf_interval(value: MPInterval) -> MPInterval:
    """Enclose the standard-normal density over ``value``."""

    down, up = _contexts(value.precision)
    pi = MPInterval(down.const_pi(), up.const_pi(), value.precision)
    normalizer = (MPInterval.point(2, value.precision) * pi).sqrt()
    return (-(value.square()) / MPInterval.point(2, value.precision)).exp() / normalizer

def verified_cholesky(
    matrix: Sequence[Sequence[MPInterval]],
) -> tuple[tuple[MPInterval, ...], ...]:
    """Return an interval Cholesky factor, rejecting unproved pivots."""

    rows = tuple(tuple(row) for row in matrix)
    size = len(rows)
    if any(len(row) != size for row in rows):
        raise ValueError("Cholesky matrix must be square")
    if size == 0:
        return ()
    precision = rows[0][0].precision
    symmetric = [list(row) for row in rows]
    for row in range(size):
        for column in range(row):
            lower = max(rows[row][column].lo, rows[column][row].lo)
            upper = min(rows[row][column].hi, rows[column][row].hi)
            if lower > upper:
                raise ValueError("Cholesky matrix is not interval-symmetric")
            value = MPInterval(lower, upper, precision)
            symmetric[row][column] = value
            symmetric[column][row] = value
    rows = tuple(tuple(row) for row in symmetric)
    zero = MPInterval.point(0, precision)
    factor = [[zero for _ in range(size)] for _ in range(size)]
    for row in range(size):
        for column in range(row + 1):
            value = rows[row][column]
            for index in range(column):
                value = value - factor[row][index] * factor[column][index]
            if row == column:
                if value.lo <= 0:
                    raise ArithmeticError(
                        f"interval Cholesky pivot {row} is not proved positive"
                    )
                factor[row][column] = value.sqrt()
            else:
                factor[row][column] = value / factor[column][column]
    return tuple(tuple(row) for row in factor)

def forward_substitute(
    lower: Sequence[Sequence[MPInterval]],
    right: Sequence[MPInterval],
) -> tuple[MPInterval, ...]:
    factor = tuple(tuple(row) for row in lower)
    size = len(factor)
    if len(right) != size or any(len(row) != size for row in factor):
        raise ValueError("triangular solve dimensions do not match")
    result: list[MPInterval] = []
    for row in range(size):
        value = right[row]
        for column in range(row):
            value = value - factor[row][column] * result[column]
        result.append(value / factor[row][row])
    return tuple(result)

def backward_substitute_transpose(
    lower: Sequence[Sequence[MPInterval]],
    right: Sequence[MPInterval],
) -> tuple[MPInterval, ...]:
    factor = tuple(tuple(row) for row in lower)
    size = len(factor)
    if len(right) != size or any(len(row) != size for row in factor):
        raise ValueError("triangular solve dimensions do not match")
    result: list[MPInterval | None] = [None] * size
    for row in range(size - 1, -1, -1):
        value = right[row]
        for column in range(row + 1, size):
            solved = result[column]
            assert solved is not None
            value = value - factor[column][row] * solved
        result[row] = value / factor[row][row]
    return tuple(value for value in result if value is not None)

def solve_spd(
    cholesky: Sequence[Sequence[MPInterval]],
    right: Sequence[MPInterval],
) -> tuple[MPInterval, ...]:
    return backward_substitute_transpose(
        cholesky, forward_substitute(cholesky, right)
    )
