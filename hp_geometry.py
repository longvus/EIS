"""Median-gap lengthscale geometry of the hyperparameter arrays, computed from configurations only."""

from __future__ import annotations

import json

import mmap

import re

class GeometryLockError(ValueError):
    """Raised when a source pool cannot support the geometry rule."""

_JSON_STRING = re.compile(
    rb'"(?:[^"\\\x00-\x1f]|\\(?:["\\/bfnrt]|u[0-9a-fA-F]{4}))*"'
)

_JSON_STRUCTURAL = re.compile(rb'["{}\[\]]')

_JSON_NUMBER_PATTERN = rb"-?(?:0|[1-9][0-9]*)(?:\.[0-9]+)?(?:[eE][+-]?[0-9]+)?"

_JSON_SCALAR_PATTERN = (
    rb"(?:" + _JSON_NUMBER_PATTERN + rb"|true|false|null|NaN|Infinity|-Infinity)"
)

_JSON_SCALAR = re.compile(_JSON_SCALAR_PATTERN)

_JSON_PRIMITIVE_LIST = re.compile(
    rb"\s*(?:"
    + _JSON_SCALAR_PATTERN
    + rb"(?:\s*,\s*"
    + _JSON_SCALAR_PATTERN
    + rb")*)?\s*"
)

_JSON_WHITESPACE = b" \t\r\n"

def _skip_json_whitespace(data: mmap.mmap, position: int) -> int:
    while position < len(data) and data[position] in _JSON_WHITESPACE:
        position += 1
    return position

def _json_string(data: mmap.mmap, position: int) -> tuple[str, int]:
    match = _JSON_STRING.match(data, position)
    if match is None:
        raise GeometryLockError(f"expected JSON string at byte {position}")
    return str(json.loads(match.group(0))), match.end()

def _parse_json_array(data: mmap.mmap, start: int) -> int:
    position = _skip_json_whitespace(data, start)
    if position >= len(data) or data[position] != ord("["):
        raise GeometryLockError(f"expected JSON array at byte {position}")
    position = _skip_json_whitespace(data, position + 1)
    if position < len(data) and data[position] == ord("]"):
        return position + 1

    # Numeric curve arrays dominate LCBench. Validate each primitive-only body
    # with one C regex match, without converting or retaining any scalar value.
    structural = _JSON_STRUCTURAL.search(data, position)
    if structural is None:
        raise GeometryLockError("unterminated JSON array")
    if data[structural.start()] == ord("]"):
        if _JSON_PRIMITIVE_LIST.fullmatch(data, position, structural.start()) is None:
            raise GeometryLockError(f"invalid JSON array at byte {start}")
        return structural.end()

    while True:
        position = _skip_json_value(data, position)
        position = _skip_json_whitespace(data, position)
        if position >= len(data):
            raise GeometryLockError("unterminated JSON array")
        if data[position] == ord("]"):
            return position + 1
        if data[position] != ord(","):
            raise GeometryLockError(f"expected array comma at byte {position}")
        position = _skip_json_whitespace(data, position + 1)

def _skip_json_value(data: mmap.mmap, position: int) -> int:
    position = _skip_json_whitespace(data, position)
    if position >= len(data):
        raise GeometryLockError("missing JSON value")
    if data[position] == ord("{"):
        return _parse_json_object(
            data,
            position,
            lambda key, value_start: _skip_json_value(data, value_start),
        )
    if data[position] == ord("["):
        return _parse_json_array(data, position)
    if data[position] == ord('"'):
        _, end = _json_string(data, position)
        return end
    match = _JSON_SCALAR.match(data, position)
    if match is None:
        raise GeometryLockError(f"invalid JSON scalar at byte {position}")
    end = match.end()
    if end < len(data) and data[end] not in b",}] \t\r\n":
        raise GeometryLockError(f"invalid JSON scalar at byte {position}")
    return end

def _parse_json_object(data: mmap.mmap, start: int, member_handler) -> int:
    position = _skip_json_whitespace(data, start)
    if position >= len(data) or data[position] != ord("{"):
        raise GeometryLockError(f"expected JSON object at byte {position}")
    position = _skip_json_whitespace(data, position + 1)
    if position < len(data) and data[position] == ord("}"):
        return position + 1
    seen: set[str] = set()
    while True:
        key, position = _json_string(data, position)
        if key in seen:
            raise GeometryLockError(f"duplicate JSON object key: {key}")
        seen.add(key)
        position = _skip_json_whitespace(data, position)
        if position >= len(data) or data[position] != ord(":"):
            raise GeometryLockError(f"missing colon after JSON key {key}")
        position = member_handler(key, position + 1)
        position = _skip_json_whitespace(data, position)
        if position >= len(data):
            raise GeometryLockError("unterminated JSON object")
        if data[position] == ord("}"):
            return position + 1
        if data[position] != ord(","):
            raise GeometryLockError(f"expected comma at byte {position}")
        position = _skip_json_whitespace(data, position + 1)
