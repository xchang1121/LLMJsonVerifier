"""Metrics against held-out ground truth; no metric is used to train the model."""

from __future__ import annotations

import math
from collections.abc import Sequence

from .schemas import Answer


def classification_metrics(rows: Sequence[tuple[Answer, str]], bins: int = 10) -> dict:
    if not rows or bins < 1:
        raise ValueError("metrics require labeled answers and a positive bin count")
    correct = nll = brier = 0.0
    buckets: list[list[tuple[float, bool]]] = [[] for _ in range(bins)]
    for answer, gold in rows:
        if gold not in answer.probabilities:
            raise ValueError(f"ground truth {gold!r} is not an allowed option for {answer.id!r}")
        hit = answer.selected == gold
        correct += hit
        nll -= math.log(max(answer.probabilities[gold], 1e-300))
        brier += math.fsum((p - (key == gold)) ** 2 for key, p in answer.probabilities.items())
        buckets[min(int(answer.confidence * bins), bins - 1)].append((answer.confidence, hit))
    count = len(rows)
    ece = math.fsum(
        abs(math.fsum(p for p, _ in bucket) - sum(hit for _, hit in bucket)) / count
        for bucket in buckets
        if bucket
    )
    return {
        "questions": count,
        "accuracy": correct / count,
        "nll": nll / count,
        "brier": brier / count,
        "ece": ece,
        "ece_bins": bins,
    }


def percentile(values: Sequence[float], quantile: float) -> float:
    if not values or not 0 <= quantile <= 1:
        raise ValueError("percentile requires data and a quantile in [0, 1]")
    ordered = sorted(values)
    position = (len(ordered) - 1) * quantile
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    return ordered[lower] + (ordered[upper] - ordered[lower]) * (position - lower)
