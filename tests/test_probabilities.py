import math

import pytest

from llm_json_verifier.errors import BackendProtocolError
from llm_json_verifier.metrics import classification_metrics, percentile
from llm_json_verifier.probabilities import make_answer


def test_stable_normalization_and_selection(question):
    answer = make_answer(question, [-10001, -10000, -10002], 1)
    assert answer.selected == "no"
    assert math.fsum(answer.probabilities.values()) == pytest.approx(1)
    assert answer.probabilities["no"] == pytest.approx(1 / (1 + math.exp(-1) + math.exp(-2)))
    assert answer.confidence == answer.probabilities["no"]
    assert answer.margin == pytest.approx(answer.probabilities["no"] - answer.probabilities["yes"])


def test_temperature_shifts_confidence_and_ties_are_stable(question):
    cold = make_answer(question, [-2, -1, -3], 0.5)
    warm = make_answer(question, [-2, -1, -3], 2)
    assert cold.selected == warm.selected
    assert cold.confidence > warm.confidence
    tied = make_answer(question, [-2, -2, -2], 1)
    assert tied.selected == question.options[0].id
    assert tied.entropy == pytest.approx(math.log(3))


@pytest.mark.parametrize("scores", [[-1], [-1, float("nan"), -3], [-1, float("inf"), -3]])
def test_invalid_score_sets_fail(question, scores):
    with pytest.raises(BackendProtocolError):
        make_answer(question, scores, 1)


def test_ground_truth_metrics(question):
    answer = make_answer(question, [math.log(0.8), math.log(0.1), math.log(0.1)], 1)
    result = classification_metrics([(answer, "yes"), (answer, "no")])
    assert result["accuracy"] == 0.5
    assert result["nll"] == pytest.approx(-(math.log(0.8) + math.log(0.1)) / 2)
    assert result["ece"] == pytest.approx(0.3)
    assert result["brier"] == pytest.approx((0.06 + 1.46) / 2)
    with pytest.raises(ValueError):
        classification_metrics([(answer, "not-an-option")])
    assert percentile([10, 20, 30], 0.95) == 29
