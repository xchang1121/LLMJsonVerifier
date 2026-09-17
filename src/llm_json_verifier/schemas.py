"""Closed request/response schemas. Application code constructs every result."""

from __future__ import annotations

import math
from typing import Annotated, Any, Literal

from pydantic import BaseModel, ConfigDict, Field, JsonValue, StringConstraints, model_validator

Identifier = Annotated[str, StringConstraints(strip_whitespace=True, min_length=1, max_length=128)]
Text = Annotated[str, StringConstraints(strip_whitespace=True, min_length=1, max_length=32_768)]


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True, allow_inf_nan=False)


class Option(StrictModel):
    id: Identifier
    description: Text


class Question(StrictModel):
    id: Identifier
    question: Text
    options: list[Option] = Field(min_length=2, max_length=702)

    @model_validator(mode="after")
    def unique_options(self) -> Question:
        if len({option.id for option in self.options}) != len(self.options):
            raise ValueError("option IDs must be unique within a question")
        if len({option.description for option in self.options}) != len(self.options):
            raise ValueError("identical option descriptions are ambiguous")
        return self


class ClassifyRequest(StrictModel):
    context: str = Field(min_length=1, max_length=2_000_000)
    questions: list[Question] = Field(min_length=1, max_length=128)
    temperature: float = Field(default=1.0, ge=0.05, le=5)
    execution: Literal["auto", "parallel", "serial"] = "auto"
    cache_namespace: str = Field(default="default", min_length=1, max_length=128)

    @model_validator(mode="after")
    def unique_questions(self) -> ClassifyRequest:
        if not self.context.strip():
            raise ValueError("context must contain non-whitespace text")
        if len({q.id for q in self.questions}) != len(self.questions):
            raise ValueError("question IDs must be unique")
        return self


class Answer(StrictModel):
    id: str
    selected: str
    probabilities: dict[str, float]
    confidence: float = Field(ge=0, le=1)
    margin: float = Field(ge=0, le=1)
    entropy: float = Field(ge=0)

    @model_validator(mode="after")
    def distribution_contract(self) -> Answer:
        if self.selected not in self.probabilities:
            raise ValueError("selected option is missing from probabilities")
        if any(not math.isfinite(p) or p < 0 or p > 1 for p in self.probabilities.values()):
            raise ValueError("invalid probability")
        if not math.isclose(math.fsum(self.probabilities.values()), 1.0, abs_tol=1e-9):
            raise ValueError("probabilities must sum to 1")
        if self.probabilities[self.selected] != max(self.probabilities.values()):
            raise ValueError("selected option must maximize probability")
        if not math.isclose(self.confidence, self.probabilities[self.selected], abs_tol=1e-9):
            raise ValueError("confidence must match the selected option probability")
        return self


class Usage(StrictModel):
    logical_prompt_tokens: int
    backend_prompt_tokens: int
    backend_completion_tokens: int
    backend_cached_prompt_tokens: int | None
    scoring_calls: int
    prefix_tokens: int
    prefix_tokenization_cache_hit: bool
    primed: bool


class Timing(StrictModel):
    preparation_ms: float
    scoring_ms: float
    total_ms: float
    queue_ms: float = Field(default=0.0, ge=0)
    backend_queue_ms: float = Field(default=0.0, ge=0)


class ClassifyResponse(StrictModel):
    request_id: str
    model: str
    model_revision: str
    answers: list[Answer]
    usage: Usage
    timing: Timing
    probability_kind: Literal["candidate_conditional_uncalibrated"] = (
        "candidate_conditional_uncalibrated"
    )


Scalar = str | int | float | bool | None


def scalar_key(value: Scalar) -> tuple:
    """JSON equality distinguishes booleans from numbers, but 1 equals 1.0."""
    if type(value) in {int, float}:
        return ("number", value)
    return (type(value).__name__, value)


class SchemaClassifyRequest(StrictModel):
    context: str = Field(min_length=1, max_length=2_000_000)
    output_schema: dict[str, Any] = Field(alias="schema")
    instruction: Text = "Classify each field using the document evidence."
    temperature: float = Field(default=1.0, ge=0.05, le=5)
    execution: Literal["auto", "parallel", "serial"] = "auto"
    cache_namespace: str = Field(default="default", min_length=1, max_length=128)

    @model_validator(mode="after")
    def meaningful_context(self):
        if not self.context.strip():
            raise ValueError("context must contain non-whitespace text")
        return self


class ValueProbability(StrictModel):
    value: Scalar
    probability: float = Field(ge=0, le=1)


class SchemaAnswer(StrictModel):
    path: str = Field(min_length=1)
    selected: Scalar
    probabilities: list[ValueProbability] = Field(min_length=2, max_length=702)
    confidence: float = Field(ge=0, le=1)
    margin: float = Field(ge=0, le=1)
    entropy: float = Field(ge=0)

    @model_validator(mode="after")
    def distribution_contract(self):
        keys = [scalar_key(entry.value) for entry in self.probabilities]
        if len(set(keys)) != len(keys) or scalar_key(self.selected) not in keys:
            raise ValueError("schema distribution must contain distinct values and its selection")
        # Reuse the same numerical contract as the question-based endpoint.
        Answer(
            id=self.path,
            selected=str(keys.index(scalar_key(self.selected))),
            probabilities={str(i): entry.probability for i, entry in enumerate(self.probabilities)},
            confidence=self.confidence,
            margin=self.margin,
            entropy=self.entropy,
        )
        return self


class SchemaClassifyResponse(StrictModel):
    request_id: str
    model: str
    model_revision: str
    result: dict[str, JsonValue]
    fields: list[SchemaAnswer]
    usage: Usage
    timing: Timing
    probability_kind: Literal["candidate_conditional_uncalibrated"] = (
        "candidate_conditional_uncalibrated"
    )
