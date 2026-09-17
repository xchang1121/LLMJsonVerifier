"""Normalize raw vocabulary log-probabilities once over the whole candidate set."""

import math

from .errors import BackendProtocolError
from .schemas import Answer, Question


def make_answer(question: Question, logprobs: list[float], temperature: float) -> Answer:
    if len(logprobs) != len(question.options) or not all(math.isfinite(x) for x in logprobs):
        raise BackendProtocolError("candidate scores are incomplete or non-finite")
    if not math.isfinite(temperature) or temperature <= 0:
        raise ValueError("temperature must be finite and positive")
    # Subtract before scaling to avoid overflow. The vocabulary normalization
    # constant cancels, including when scores arrived in several HTTP chunks.
    peak = max(logprobs)
    weights = [math.exp((value - peak) / temperature) for value in logprobs]
    total = math.fsum(weights)
    probs = [weight / total for weight in weights]
    winner = max(range(len(probs)), key=probs.__getitem__)
    ordered = sorted(probs, reverse=True)
    entropy = -math.fsum(p * math.log(p) for p in probs if p > 0)
    return Answer(
        id=question.id,
        selected=question.options[winner].id,
        probabilities={option.id: p for option, p in zip(question.options, probs, strict=True)},
        confidence=probs[winner],
        margin=ordered[0] - ordered[1],
        entropy=max(0.0, entropy),
    )
