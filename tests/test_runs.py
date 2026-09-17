import asyncio
import json

import pytest
from test_client import FakeGateway

from llm_json_verifier.client import GatewayHTTPError, benchmark, evaluate
from llm_json_verifier.datasets import parse_dataset, regression_cases, write_regression_dataset


class FailingGateway(FakeGateway):
    def __init__(self, fail_on):
        super().__init__()
        self.attempts = 0
        self.fail_on = fail_on

    async def classify(self, request):
        self.attempts += 1
        if self.attempts in self.fail_on:
            raise GatewayHTTPError(503)
        return await super().classify(request)


async def test_benchmark_records_failures_and_separates_warmup(tmp_path, request_body):
    path = tmp_path / "run.jsonl"
    client = FailingGateway({1, 3})
    result = await benchmark(client, request_body, 3, 2, 1, "warm", path)
    assert result["warmup_outcomes"]["failed"] == 1
    assert (result["attempted"], result["succeeded"], result["failed"]) == (3, 2, 1)
    assert result["requests_per_second"] * result["wall_seconds"] == pytest.approx(2)
    events = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
    assert events[0]["kind"] == "run" and len(events[0]["request_sha256"]) == 64
    assert events[-1]["kind"] == "summary"
    attempts = events[1:-1]
    assert len(attempts) == 4
    assert all(event["request"]["context"] == request_body.context for event in attempts)
    assert attempts[0]["error"] == {"type": "GatewayHTTPError", "http_status": 503}
    assert attempts[1]["response"]["answers"][0]["selected"] == "yes"


async def test_all_failed_run_has_no_success_latency(tmp_path, request_body):
    result = await benchmark(FailingGateway({1, 2}), request_body, 2, 2, 0, "cold")
    assert result["requests_per_second"] == 0
    assert result["latency_ms"] == {"p50": None, "p95": None, "p99": None}
    assert result["all_outcomes_latency_ms"]["p50"] >= 0


async def test_record_file_is_exclusive_before_any_http(tmp_path, request_body):
    path = tmp_path / "run.jsonl"
    path.write_text("existing", encoding="utf-8")
    client = FakeGateway()
    with pytest.raises(FileExistsError):
        await benchmark(client, request_body, 1, 1, 0, "warm", path)
    assert not client.requests
    assert path.read_text() == "existing"


async def test_cancellation_drains_workers_and_records_inflight(tmp_path, request_body):
    entered = asyncio.Event()
    active = 0

    class SlowGateway:
        async def classify(self, request):
            nonlocal active
            active += 1
            if active == 2:
                entered.set()
            try:
                await asyncio.Event().wait()
            finally:
                active -= 1

    path = tmp_path / "interrupted.jsonl"
    task = asyncio.create_task(benchmark(SlowGateway(), request_body, 100, 2, 0, "warm", path))
    await asyncio.wait_for(entered.wait(), 1)
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    assert active == 0
    records = [json.loads(line) for line in path.read_text().splitlines()]
    assert len(records) == 3
    assert [record["status"] for record in records[1:]] == ["canceled", "canceled"]


def write_cases(path, request_body, count=2):
    path.write_text(
        "\n".join(
            json.dumps(
                {
                    "id": f"case-{i}",
                    "request": request_body.model_dump(),
                    "expected": {"status": "yes"},
                }
            )
            for i in range(count)
        ),
        encoding="utf-8",
    )


async def test_all_dataset_records_are_validated_before_inference(tmp_path, request_body):
    path = tmp_path / "eval.jsonl"
    write_cases(path, request_body)
    with path.open("a", encoding="utf-8") as handle:
        handle.write('\n{"request": {}}')
    client = FakeGateway()
    with pytest.raises(ValueError, match="line 3"):
        await evaluate(client, path)
    assert not client.requests


async def test_evaluation_partial_failure_denominators(tmp_path, request_body):
    path = tmp_path / "eval.jsonl"
    write_cases(path, request_body)
    result = await evaluate(FailingGateway({2}), path, concurrency=2)
    assert result["samples"] == result["requested_questions"] == 2
    assert result["succeeded"] == result["questions"] == 1
    assert result["accuracy"] == 1
    assert result["question_coverage"] == result["end_to_end_accuracy"] == 0.5
    assert result["exact_match_rate"] == 0.5


async def test_evaluation_all_failed_and_failed_rotation(tmp_path, request_body):
    path = tmp_path / "eval.jsonl"
    write_cases(path, request_body, 1)
    failed = await evaluate(FailingGateway({1}), path, rotate=True)
    assert failed["accuracy"] is None
    assert failed["end_to_end_accuracy"] == 0
    assert failed["rotation_outcomes"]["attempted"] == 0
    rotated = await evaluate(FailingGateway({2}), path, rotate=True)
    assert rotated["rotation_outcomes"]["failed"] == 1
    assert rotated["one_step_rotation_agreement"] is None
    assert rotated["accuracy"] == 1


def test_regression_fixtures_are_deterministic_labeled_and_positioned(tmp_path):
    path = tmp_path / "data.jsonl"
    summary = write_regression_dataset(path, (0, 1000), ("start", "middle", "end"))
    cases = parse_dataset(path.read_bytes())
    assert len(cases) == summary["samples"] == 80
    assert all(len(case.request.questions) == 4 for case in cases)
    assert set(cases[0].expected.values()) == {"unknown"}
    assert cases[15].expected == {
        "delivery": "yes",
        "payment": "yes",
        "return": "no",
        "warranty": "yes",
    }
    contexts = {case.id: case.request.context for case in cases}
    for position, expected_location in (("start", 1), ("middle", 497), ("end", 993)):
        context = contexts[f"facts-01-1000-{position}"]
        assert len(context) == 1000
        assert abs(context.index("物流已签收。") - expected_location) <= 2
    assert regression_cases() == regression_cases()


def test_dataset_rejects_duplicate_ids(tmp_path):
    case = regression_cases()[0]
    with pytest.raises(ValueError, match="duplicate"):
        parse_dataset((json.dumps(case) + "\n" + json.dumps(case)).encode())
