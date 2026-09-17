import copy
import itertools
import json

import httpx
import pytest
from jsonschema import Draft202012Validator
from pydantic import ValidationError
from test_api import api_client
from test_service import RecordingBackend

from llm_json_verifier.api import create_app
from llm_json_verifier.client import GatewayClient, read_schema_request
from llm_json_verifier.errors import BackendProtocolError, SchemaError
from llm_json_verifier.probabilities import make_answer
from llm_json_verifier.schema_compiler import compile_schema
from llm_json_verifier.schemas import SchemaAnswer, SchemaClassifyRequest, SchemaClassifyResponse
from llm_json_verifier.service import ClassificationService


def object_schema(properties, **annotations):
    return {
        "type": "object",
        "properties": properties,
        "required": list(properties),
        "additionalProperties": False,
        **annotations,
    }


def request_for(schema):
    return SchemaClassifyRequest.model_validate(
        {
            "context": "Document evidence.",
            "schema": schema,
            "instruction": "Use unknown if evidence is insufficient.",
        }
    )


@pytest.fixture
def nested_schema():
    return object_schema(
        {
            "a/b~": object_schema(
                {
                    "status": {"enum": ["ready", "unknown"], "description": "Shipment status."},
                    "flag": {"type": "boolean"},
                    "score": {"type": "integer", "minimum": 0, "maximum": 2},
                },
                description="Shipping information.",
            ),
            "tuple": {
                "type": "array",
                "prefixItems": [
                    {"type": "boolean"},
                    {"enum": [None, "not applicable"]},
                    {"const": 2},
                ],
                "items": False,
                "minItems": 3,
                "maxItems": 3,
            },
            "version": {"const": "v1"},
            "empty": {"type": "array", "items": False, "minItems": 0},
        },
        title="Order review",
        description="Read only the supplied evidence.",
    )


def test_every_choice_combination_satisfies_independent_schema_validator(settings, nested_schema):
    Draft202012Validator.check_schema(nested_schema)
    validator = Draft202012Validator(nested_schema)
    plan = compile_schema(request_for(nested_schema), settings.service)
    assert [field.path for field in plan.fields] == [
        "/a~1b~0/status",
        "/a~1b~0/flag",
        "/a~1b~0/score",
        "/tuple/0",
        "/tuple/1",
    ]
    for choices in itertools.product(*(range(len(field.values)) for field in plan.fields)):
        answers = [
            make_answer(
                field.question,
                [0.0 if i == chosen else -10.0 for i in range(len(field.values))],
                1.0,
            )
            for field, chosen in zip(plan.fields, choices, strict=True)
        ]
        result, fields = plan.assemble(answers)
        validator.validate(result)
        assert result["version"] == "v1" and result["tuple"][2] == 2 and result["empty"] == []
        assert type(result["a/b~"]["flag"]) is bool
        assert type(result["a/b~"]["score"]) is int
        assert all(
            sum(p.probability for p in field.probabilities) == pytest.approx(1) for field in fields
        )


def test_mixed_enum_preserves_boolean_number_string_and_null(settings):
    values = [False, 0, "0", None, True, 1, 1.5]
    schema = object_schema({"value": {"enum": values}})
    Draft202012Validator.check_schema(schema)
    plan = compile_schema(request_for(schema), settings.service)
    for winner, value in enumerate(values):
        answer = make_answer(
            plan.fields[0].question, [0 if i == winner else -2 for i in range(len(values))], 1.0
        )
        result, fields = plan.assemble([answer])
        assert type(result["value"]) is type(value)
        assert result["value"] == value
        assert [type(p.value) for p in fields[0].probabilities] == [type(v) for v in values]
        assert type(fields[0].selected) is type(value)
        Draft202012Validator(schema).validate(result)


def test_descriptions_paths_and_instructions_reach_compiled_task(settings, nested_schema):
    plan = compile_schema(request_for(nested_schema), settings.service)
    task = plan.fields[0].question.question
    for text in (
        "Order review",
        "Shipping information",
        "Shipment status",
        "Use unknown",
        "/a~1b~0/status",
    ):
        assert text in task
    assert [o.description for o in plan.fields[0].question.options] == ['"ready"', '"unknown"']


@pytest.mark.parametrize(
    "leaf",
    [
        {"type": None, "enum": ["a", "b"]},
        {"type": "string"},
        {"type": "number"},
        {"type": ["string", "null"], "enum": ["x", None]},
        {"enum": []},
        {"enum": [1, 1.0]},
        {"enum": ["x", "x"]},
        {"enum": [[1], [2]]},
        {"type": "integer", "enum": [True, 1]},
        {"type": "boolean", "const": 1},
        {"type": "integer", "enum": [1.5, 2]},
        {"const": 1, "enum": [1]},
        {"type": "integer", "minimum": 2, "maximum": 1},
        {"type": "integer", "minimum": 0, "maximum": 1000},
        {"type": "integer", "minimum": 1, "enum": [0, 2]},
        {"type": "string", "enum": ["a", "b"], "pattern": "a"},
        {"$ref": "https://example.com/schema"},
        {"anyOf": [{"const": 1}, {"const": 2}]},
        {"type": "object", "properties": {"x": {"type": "boolean"}}, "required": []},
        {"type": "array", "prefixItems": [{"type": "boolean"}], "items": False},
        {"type": "array", "prefixItems": [], "items": False, "minItems": 0},
        {
            "type": "array",
            "prefixItems": [{"type": "boolean"}],
            "items": False,
            "minItems": 1,
            "maxItems": 2,
        },
    ],
)
def test_unsupported_or_contradictory_schemas_fail_preflight(settings, leaf):
    with pytest.raises(SchemaError):
        compile_schema(request_for(object_schema({"bad": leaf})), settings.service)


@pytest.mark.parametrize(
    "limit,value",
    [
        ("max_questions", 1),
        ("max_schema_depth", 1),
        ("max_schema_nodes", 1),
        ("max_schema_bytes", 256),
        ("max_options", 2),
    ],
)
def test_schema_budgets(settings, nested_schema, limit, value):
    setattr(settings.service, limit, value)
    with pytest.raises(SchemaError):
        compile_schema(request_for(nested_schema), settings.service)


async def test_late_schema_error_spends_no_backend_requests(settings, compiler):
    backend = RecordingBackend()
    service = ClassificationService(settings, compiler, backend)
    schema = object_schema({"valid": {"type": "boolean"}, "invalid": {"type": "string"}})
    with pytest.raises(SchemaError):
        await service.classify_schema(request_for(schema))
    assert not backend.calls
    assert service.admission.active == 0


async def test_constant_only_schema_needs_zero_scoring(settings, compiler):
    backend = RecordingBackend()
    schema = object_schema(
        {"version": {"const": "v1"}, "empty": {"type": "null"}, "level": {"enum": [3]}}
    )
    result = await ClassificationService(settings, compiler, backend).classify_schema(
        request_for(schema)
    )
    assert result.result == {"version": "v1", "empty": None, "level": 3}
    assert result.fields == [] and result.usage.scoring_calls == 0 and not backend.calls
    Draft202012Validator(schema).validate(result.result)


async def test_gateway_schema_client_round_trip_and_large_menu(settings, compiler):
    settings.backend.verify_on_startup = False
    schema = object_schema({"level": {"type": "integer", "minimum": 0, "maximum": 255}})
    request = request_for(schema)
    service = ClassificationService(settings, compiler, RecordingBackend())
    app = create_app(settings, service)
    async with app.router.lifespan_context(app), GatewayClient("http://test") as client:
        await client.http.aclose()
        client.http = httpx.AsyncClient(
            transport=httpx.ASGITransport(app=app), base_url="http://test"
        )
        response = await client.classify_schema(request)
        assert response.usage.scoring_calls == 2
        assert len(response.fields[0].probabilities) == 256
        assert response.result == {"level": 0}
        Draft202012Validator(schema).validate(response.result)


def test_schema_endpoint_uses_existing_vllm_token_score_backend(settings, compiler, nested_schema):
    request = request_for(nested_schema)
    with api_client(settings, compiler) as client:
        response = client.post("/v1/classify-schema", json=request.model_dump(by_alias=True))
        assert response.status_code == 200, response.text
        data = SchemaClassifyResponse.model_validate(response.json())
        assert data.usage.scoring_calls == 5
        assert data.result["a/b~"] == {"status": "ready", "flag": False, "score": 0}
        assert data.result["tuple"] == [False, None, 2]
        compile_schema(request, settings.service).verify_response(data)
        Draft202012Validator(nested_schema).validate(data.result)


@pytest.mark.parametrize("corruption", ["missing", "constant", "boolean", "value", "probabilities"])
async def test_client_rejects_inconsistent_schema_response(
    settings, compiler, nested_schema, corruption
):
    request = request_for(nested_schema)
    result = await ClassificationService(settings, compiler, RecordingBackend()).classify_schema(
        request
    )
    body = result.model_dump()
    if corruption == "missing":
        body["fields"].pop()
    elif corruption == "constant":
        body["result"]["version"] = "v2"
    elif corruption == "boolean":
        body["result"]["a/b~"]["flag"] = 0
    elif corruption == "value":
        body["result"]["a/b~"]["status"] = "unknown"
    else:
        body["fields"][0]["probabilities"][1]["value"] = "invalid"
    async with GatewayClient("http://test") as client:
        await client.http.aclose()
        client.http = httpx.AsyncClient(
            transport=httpx.MockTransport(lambda _: httpx.Response(200, json=body)),
            base_url="http://test",
        )
        with pytest.raises(BackendProtocolError):
            await client.classify_schema(request)


def test_schema_distribution_enforces_normalization():
    with pytest.raises(ValidationError, match="sum to 1"):
        SchemaAnswer(
            path="/a",
            selected=True,
            probabilities=[
                {"value": True, "probability": 0.6},
                {"value": False, "probability": 0.3},
            ],
            confidence=0.6,
            margin=0.3,
            entropy=0.9,
        )


@pytest.mark.parametrize(
    "data",
    [
        '{"context":"SECRET","context":"OTHER"}',
        '{"schema":{"properties":{"a":{},"\\u0061":{}}}}',
        '{"temperature":NaN}',
        '{"temperature":1e999}',
    ],
)
def test_api_rejects_duplicate_keys_and_nonfinite_json(settings, compiler, data):
    with api_client(settings, compiler) as client:
        response = client.post(
            "/v1/classify-schema", content=data, headers={"Content-Type": "application/json"}
        )
        assert response.status_code == 422
        assert response.json()["error"]["code"] == "invalid_json"
        assert "SECRET" not in response.text


def test_schema_file_loader_and_empty_property_name(tmp_path, settings):
    path = tmp_path / "request.json"
    schema = object_schema({"": {"type": "boolean"}})
    path.write_text(json.dumps({"context": "Evidence.", "schema": schema}), encoding="utf-8")
    plan = compile_schema(read_schema_request(path), settings.service)
    assert plan.fields[0].path == "/"
    original = copy.deepcopy(schema)
    compile_schema(request_for(schema), settings.service)
    assert schema == original
