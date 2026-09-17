from __future__ import annotations

import re

import pytest

from llm_json_verifier.config import Settings
from llm_json_verifier.prompts import CODE_CANDIDATES, PromptCompiler
from llm_json_verifier.schemas import ClassifyRequest, Option, Question


class TinyTokenizer:
    """Deterministic special boundaries and one-token codes, without model weights."""

    all_special_tokens = ["<start>", "<end>"]
    all_special_ids = [1, 2]
    table = {code: index + 10 for index, code in enumerate(CODE_CANDIDATES)}
    table.update({"<start>": 1, "<end>": 2})
    reverse = {value: key for key, value in table.items()}

    def encode(self, text, *, add_special_tokens=False):
        assert not add_special_tokens
        pieces = re.findall(r"<start>|<end>|[A-Z]{1,2}|[\s\S]", text)
        return [self.table[piece] if piece in self.table else 1000 + ord(piece) for piece in pieces]

    def decode(self, ids, **kwargs):
        return "".join(
            self.reverse[token] if token in self.reverse else chr(token - 1000) for token in ids
        )

    def apply_chat_template(self, messages, **kwargs):
        assert kwargs["enable_thinking"] is False
        return (
            "".join(f"<start>{m['role']}\n{m['content']}<end>\n" for m in messages)
            + "<start>assistant\n"
        )


@pytest.fixture
def settings():
    return Settings()


@pytest.fixture
def tokenizer():
    return TinyTokenizer()


@pytest.fixture
def compiler(settings, tokenizer):
    return PromptCompiler(tokenizer, settings)


@pytest.fixture
def question():
    return Question(
        id="status",
        question="Has the order arrived?",
        options=[
            Option(id="yes", description="The order has arrived."),
            Option(id="no", description="The order has not arrived."),
            Option(id="unknown", description="The evidence does not establish arrival status."),
        ],
    )


@pytest.fixture
def request_body(question):
    return ClassifyRequest(context="The order arrived on June 3.", questions=[question])


def completion(prompt_ids, ids, scores=None, cached=0):
    scores = scores or [-float(index + 1) for index in range(len(ids))]
    return {
        "choices": [
            {
                "finish_reason": "length",
                "text": "NOT AN ANSWER",
                "logprobs": {
                    "top_logprobs": [
                        {
                            f"token_id:{token}": score
                            for token, score in zip(ids, scores, strict=True)
                        }
                    ]
                },
            }
        ],
        "usage": {
            "prompt_tokens": len(prompt_ids),
            "completion_tokens": 1,
            "prompt_tokens_details": {"cached_tokens": cached},
        },
    }
