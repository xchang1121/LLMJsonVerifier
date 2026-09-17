"""Compile a bounded, finite JSON Schema subset into independent choice tasks."""

from __future__ import annotations

import json
import math
from dataclasses import dataclass

from pydantic import ValidationError

from .config import ServiceSettings
from .errors import BackendProtocolError, SchemaError
from .schemas import (
    Answer,
    ClassifyRequest,
    Option,
    Question,
    Scalar,
    SchemaAnswer,
    SchemaClassifyRequest,
    SchemaClassifyResponse,
    ValueProbability,
    scalar_key,
)


def json_text(value) -> str:
    return json.dumps(value, ensure_ascii=False, allow_nan=False, separators=(",", ":"))


def child_path(path: str, key: str) -> str:
    return path + "/" + key.replace("~", "~0").replace("/", "~1")


@dataclass(frozen=True)
class ObjectNode:
    children: tuple[tuple[str, Node], ...]


@dataclass(frozen=True)
class ArrayNode:
    children: tuple[Node, ...]


@dataclass(frozen=True)
class ConstantNode:
    value: Scalar


@dataclass(frozen=True)
class ChoiceNode:
    index: int


Node = ObjectNode | ArrayNode | ConstantNode | ChoiceNode


def materialize(node: Node, selected: list[Scalar]):
    if isinstance(node, ObjectNode):
        return {key: materialize(child, selected) for key, child in node.children}
    if isinstance(node, ArrayNode):
        return [materialize(child, selected) for child in node.children]
    if isinstance(node, ConstantNode):
        return node.value
    return selected[node.index]


@dataclass(frozen=True)
class FieldPlan:
    path: str
    values: tuple[Scalar, ...]
    question: Question


@dataclass(frozen=True)
class SchemaPlan:
    root: ObjectNode
    fields: tuple[FieldPlan, ...]

    def request(self, source: SchemaClassifyRequest) -> ClassifyRequest:
        return ClassifyRequest(
            context=source.context,
            questions=[f.question for f in self.fields],
            temperature=source.temperature,
            execution=source.execution,
            cache_namespace=source.cache_namespace,
        )

    def assemble(self, answers: list[Answer]) -> tuple[dict, list[SchemaAnswer]]:
        if [a.id for a in answers] != [f.question.id for f in self.fields]:
            raise BackendProtocolError("schema response has incomplete field coverage")
        fields = []
        selected = []
        for field, answer in zip(self.fields, answers, strict=True):
            options = field.question.options
            if set(answer.probabilities) != {o.id for o in options}:
                raise BackendProtocolError("schema response has incomplete candidate coverage")
            values = {option.id: value for option, value in zip(options, field.values, strict=True)}
            value = values[answer.selected]
            selected.append(value)
            fields.append(
                SchemaAnswer(
                    path=field.path,
                    selected=value,
                    probabilities=[
                        ValueProbability(value=values[o.id], probability=answer.probabilities[o.id])
                        for o in options
                    ],
                    confidence=answer.confidence,
                    margin=answer.margin,
                    entropy=answer.entropy,
                )
            )
        return materialize(self.root, selected), fields

    def verify_response(self, response: SchemaClassifyResponse) -> None:
        if [f.path for f in response.fields] != [f.path for f in self.fields]:
            raise BackendProtocolError("schema response has incomplete field coverage")
        selected = []
        for plan, answer in zip(self.fields, response.fields, strict=True):
            if [scalar_key(v.value) for v in answer.probabilities] != [
                scalar_key(v) for v in plan.values
            ]:
                raise BackendProtocolError("schema response has different candidates")
            selected.append(
                plan.values[[scalar_key(v) for v in plan.values].index(scalar_key(answer.selected))]
            )
        expected = materialize(self.root, selected)
        # Canonical JSON also distinguishes a boolean from the numeric value 0/1.
        if json.dumps(response.result, sort_keys=True) != json.dumps(expected, sort_keys=True):
            raise BackendProtocolError(
                "schema result differs from its field selections or constants"
            )


def compile_schema(request: SchemaClassifyRequest, limits: ServiceSettings) -> SchemaPlan:
    schema = request.output_schema
    try:
        size = len(json_text(schema).encode("utf-8"))
    except (TypeError, ValueError, RecursionError) as exc:
        raise SchemaError("schema must contain finite JSON data") from exc
    if size > limits.max_schema_bytes:
        raise SchemaError("schema exceeds max_schema_bytes")
    if schema.get("type") != "object":
        raise SchemaError("schema root must have type object")
    fields: list[FieldPlan] = []
    nodes = 0

    def fail(path, message):
        raise SchemaError(f"schema at {path or '/'}: {message}")

    def visit(spec, path, depth, inherited):
        nonlocal nodes
        nodes += 1
        if depth > limits.max_schema_depth or nodes > limits.max_schema_nodes:
            fail(path, "schema depth or node limit exceeded")
        if not isinstance(spec, dict):
            fail(path, "each node must be a schema object")
        if any(not isinstance(key, str) for key in spec):
            fail(path, "schema keyword names must be strings")
        kind = spec.get("type")
        if "type" in spec and kind not in (
            "object",
            "array",
            "string",
            "integer",
            "number",
            "boolean",
            "null",
        ):
            fail(path, "type must be a single supported JSON type")
        allowed = {"type", "title", "description"}
        if not path:
            allowed.add("$schema")
            if (
                "$schema" in spec
                and spec["$schema"] != "https://json-schema.org/draft/2020-12/schema"
            ):
                fail(path, "$schema must be the JSON Schema 2020-12 URI")
        if kind == "object":
            allowed |= {"properties", "required", "additionalProperties"}
        elif kind == "array":
            allowed |= {"prefixItems", "items", "minItems", "maxItems"}
        else:
            allowed |= {"enum", "const"}
            if kind == "integer":
                allowed |= {"minimum", "maximum"}
        if set(spec) - allowed:
            fail(path, "unsupported keywords: " + ", ".join(sorted(set(spec) - allowed)))
        annotations = {key: spec[key] for key in ("title", "description") if key in spec}
        if any(not isinstance(value, str) or len(value) > 32768 for value in annotations.values()):
            fail(path, "title and description must be strings of at most 32768 characters")
        context = inherited + ([{"path": path, **annotations}] if annotations else [])
        if kind == "object":
            properties, required = spec.get("properties"), spec.get("required")
            if not isinstance(properties, dict) or not isinstance(required, list):
                fail(path, "object requires properties and required")
            if (
                any(not isinstance(key, str) for key in required)
                or len(set(required)) != len(required)
                or set(required) != set(properties)
                or spec.get("additionalProperties") is not False
            ):
                fail(path, "every property must be required and additionalProperties must be false")
            if any(not isinstance(key, str) for key in properties):
                fail(path, "property names must be strings")
            return ObjectNode(
                tuple(
                    (key, visit(child, child_path(path, key), depth + 1, context))
                    for key, child in properties.items()
                )
            )
        if kind == "array":
            children = spec.get("prefixItems", [])
            if not isinstance(children, list) or ("prefixItems" in spec and not children):
                fail(path, "prefixItems must be a nonempty array when supplied")
            length = len(children)
            if (
                spec.get("items") is not False
                or type(spec.get("minItems")) is not int
                or spec["minItems"] != length
                or (
                    "maxItems" in spec
                    and (type(spec["maxItems"]) is not int or spec["maxItems"] != length)
                )
            ):
                fail(
                    path,
                    "tuple requires items=false and minItems equal to its length; maxItems must agree",
                )
            return ArrayNode(
                tuple(
                    visit(child, child_path(path, str(i)), depth + 1, context)
                    for i, child in enumerate(children)
                )
            )
        if "const" in spec and "enum" in spec:
            fail(path, "use either const or enum")
        if "const" in spec:
            values = [spec["const"]]
        elif "enum" in spec:
            values = spec["enum"]
            if not isinstance(values, list) or not values:
                fail(path, "enum must be a nonempty array")
        elif kind == "boolean":
            values = [False, True]
        elif kind == "null":
            values = [None]
        elif kind == "integer" and {"minimum", "maximum"} <= set(spec):
            low, high = spec["minimum"], spec["maximum"]
            if type(low) is not int or type(high) is not int or low > high:
                fail(path, "integer bounds must be ordered integers")
            if high - low + 1 > limits.max_options:
                fail(path, "integer range exceeds max_options")
            values = list(range(low, high + 1))
        else:
            fail(path, "scalar requires enum, const, boolean, null, or a bounded integer range")
        if len(values) > limits.max_options:
            fail(path, "enum exceeds max_options")
        for value in values:
            if type(value) not in {str, int, float, bool, type(None)}:
                fail(path, "enum and const values must be JSON scalars")
            if isinstance(value, float) and not math.isfinite(value):
                fail(path, "numbers must be finite")
            matches = {
                None: True,
                "string": type(value) is str,
                "boolean": type(value) is bool,
                "null": value is None,
                "number": type(value) in {int, float},
                "integer": type(value) is int or (type(value) is float and value.is_integer()),
            }
            if not matches[kind]:
                fail(path, "candidate does not match its declared type")
        if len({scalar_key(value) for value in values}) != len(values):
            fail(path, "enum values must be distinct under JSON equality")
        if kind == "integer":
            for bound in ("minimum", "maximum"):
                if bound in spec and type(spec[bound]) is not int:
                    fail(path, "integer bounds must be integers")
            if (
                "minimum" in spec
                and any(v < spec["minimum"] for v in values)
                or "maximum" in spec
                and any(v > spec["maximum"] for v in values)
            ):
                fail(path, "candidates must satisfy the integer bounds")
        if len(values) == 1:
            return ConstantNode(values[0])
        if len(fields) >= limits.max_questions:
            fail(path, "variable fields exceed max_questions")
        task = {
            "instruction": request.instruction,
            "field": path,
            "type": kind,
            "schema_context": context,
        }
        try:
            question = Question(
                id=f"field_{len(fields)}",
                question="Select this field's JSON value from the candidates using the document.\n"
                + json_text(task),
                options=[
                    Option(id=f"v{i}", description=json_text(value))
                    for i, value in enumerate(values)
                ],
            )
        except ValidationError as exc:
            raise SchemaError(
                f"schema at {path}: compiled question or candidate exceeds text limits"
            ) from exc
        field = FieldPlan(path, tuple(values), question)
        index = len(fields)
        fields.append(field)
        return ChoiceNode(index)

    root = visit(schema, "", 1, [])
    return SchemaPlan(root, tuple(fields))
