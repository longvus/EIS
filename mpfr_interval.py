"""Small directed-rounding MPFR interval primitives.

Binary64 inputs are imported exactly. Each primitive operation is evaluated
separately with MPFR round-toward-negative and round-toward-positive modes.
"""

from __future__ import annotations

from dataclasses import dataclass

from functools import lru_cache

import math

import gmpy2 as g

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

def _exact(value: float | int) -> g.mpfr:
    if isinstance(value, bool):
        raise TypeError("boolean is not a numeric interval endpoint")
    if isinstance(value, int):
        return g.mpfr(value)
    value = float(value)
    if math.isnan(value):
        raise ValueError("NaN interval endpoint")
    return g.mpfr(value, precision=1)

def _float_down(value: g.mpfr) -> float:
    rounded = float(value)
    if math.isinf(rounded):
        return rounded
    if _exact(rounded) > value:
        rounded = math.nextafter(rounded, -math.inf)
    return rounded

def _float_up(value: g.mpfr) -> float:
    rounded = float(value)
    if math.isinf(rounded):
        return rounded
    if _exact(rounded) < value:
        rounded = math.nextafter(rounded, math.inf)
    return rounded

@dataclass(frozen=True)
class MPInterval:
    lo: g.mpfr
    hi: g.mpfr
    precision: int = 256

    def __post_init__(self) -> None:
        if g.is_nan(self.lo) or g.is_nan(self.hi) or self.lo > self.hi:
            raise ValueError("invalid MPFR interval")
        _contexts(self.precision)

    @classmethod
    def point(cls, value: float | int, precision: int = 256) -> "MPInterval":
        exact = _exact(value)
        return cls(exact, exact, precision)

    @classmethod
    def bounds(
        cls, lower: float, upper: float, precision: int = 256
    ) -> "MPInterval":
        return cls(_exact(lower), _exact(upper), precision)

    def _coerce(self, other: "MPInterval | float | int") -> "MPInterval":
        if isinstance(other, MPInterval):
            if other.precision != self.precision:
                raise ValueError("mixed MPFR interval precisions")
            return other
        return MPInterval.point(other, self.precision)

    def __neg__(self) -> "MPInterval":
        down, up = _contexts(self.precision)
        zero = g.mpfr(0)
        return MPInterval(
            down.sub(zero, self.hi), up.sub(zero, self.lo), self.precision
        )

    def __add__(self, other: "MPInterval | float | int") -> "MPInterval":
        other = self._coerce(other)
        down, up = _contexts(self.precision)
        return MPInterval(
            down.add(self.lo, other.lo), up.add(self.hi, other.hi), self.precision
        )

    __radd__ = __add__

    def __sub__(self, other: "MPInterval | float | int") -> "MPInterval":
        other = self._coerce(other)
        down, up = _contexts(self.precision)
        return MPInterval(
            down.sub(self.lo, other.hi), up.sub(self.hi, other.lo), self.precision
        )

    def __rsub__(self, other: "MPInterval | float | int") -> "MPInterval":
        return self._coerce(other).__sub__(self)

    def __mul__(self, other: "MPInterval | float | int") -> "MPInterval":
        other = self._coerce(other)
        down, up = _contexts(self.precision)
        lower = [
            down.mul(left, right)
            for left in (self.lo, self.hi)
            for right in (other.lo, other.hi)
        ]
        upper = [
            up.mul(left, right)
            for left in (self.lo, self.hi)
            for right in (other.lo, other.hi)
        ]
        return MPInterval(min(lower), max(upper), self.precision)

    __rmul__ = __mul__

    def __truediv__(self, other: "MPInterval | float | int") -> "MPInterval":
        other = self._coerce(other)
        if other.lo <= 0 <= other.hi:
            raise ZeroDivisionError("interval denominator contains zero")
        down, up = _contexts(self.precision)
        lower = [
            down.div(left, right)
            for left in (self.lo, self.hi)
            for right in (other.lo, other.hi)
        ]
        upper = [
            up.div(left, right)
            for left in (self.lo, self.hi)
            for right in (other.lo, other.hi)
        ]
        return MPInterval(min(lower), max(upper), self.precision)

    def __rtruediv__(self, other: "MPInterval | float | int") -> "MPInterval":
        return self._coerce(other).__truediv__(self)

    def square(self) -> "MPInterval":
        down, up = _contexts(self.precision)
        upper = max(up.mul(self.lo, self.lo), up.mul(self.hi, self.hi))
        if self.lo <= 0 <= self.hi:
            lower = g.mpfr(0)
        else:
            lower = min(down.mul(self.lo, self.lo), down.mul(self.hi, self.hi))
        return MPInterval(lower, upper, self.precision)

    def sqrt(self) -> "MPInterval":
        if self.lo < 0:
            raise ValueError("sqrt interval crosses below zero")
        down, up = _contexts(self.precision)
        return MPInterval(down.sqrt(self.lo), up.sqrt(self.hi), self.precision)

    def exp(self) -> "MPInterval":
        down, up = _contexts(self.precision)
        return MPInterval(down.exp(self.lo), up.exp(self.hi), self.precision)

    def log(self) -> "MPInterval":
        if self.lo <= 0:
            raise ValueError("log interval is not positive")
        down, up = _contexts(self.precision)
        return MPInterval(down.log(self.lo), up.log(self.hi), self.precision)

    def erfc(self) -> "MPInterval":
        down, up = _contexts(self.precision)
        return MPInterval(down.erfc(self.hi), up.erfc(self.lo), self.precision)

    def maximum(self, other: "MPInterval | float | int") -> "MPInterval":
        other = self._coerce(other)
        return MPInterval(
            max(self.lo, other.lo), max(self.hi, other.hi), self.precision
        )

    def minimum(self, other: "MPInterval | float | int") -> "MPInterval":
        other = self._coerce(other)
        return MPInterval(
            min(self.lo, other.lo), min(self.hi, other.hi), self.precision
        )

    def hull(self, other: "MPInterval") -> "MPInterval":
        other = self._coerce(other)
        return MPInterval(min(self.lo, other.lo), max(self.hi, other.hi), self.precision)

    def to_float(self) -> tuple[float, float]:
        return _float_down(self.lo), _float_up(self.hi)
