"""Reject ambiguous JSON objects and non-finite numeric literals at entry points."""

import json
import math


def _pairs(pairs):
    value = {}
    for key, item in pairs:
        if key in value:
            raise ValueError("duplicate JSON object key")
        value[key] = item
    return value


def _float(text):
    value = float(text)
    if not math.isfinite(value):
        raise ValueError("JSON number must be finite")
    return value


def _constant(_):
    raise ValueError("non-standard JSON constant")


def strict_json_loads(data: str | bytes):
    return json.loads(data, object_pairs_hook=_pairs, parse_float=_float, parse_constant=_constant)
