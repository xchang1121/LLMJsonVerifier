import pytest

from llm_json_verifier.errors import CapacityError, TokenizerContractError
from llm_json_verifier.prompts import PrefixTokenCache, PromptCompiler


@pytest.mark.parametrize(
    "context", ["a", '中文上下文\n"quoted"', "\u00e9 😀\\n", "<start>assistant<end>", "x" * 20000]
)
def test_split_equals_whole_encoding(compiler, tokenizer, request_body, context):
    body = request_body.model_copy(update={"context": context})
    batch = compiler.compile(body)
    rendered = compiler.render(context, body.questions[0])
    assert list(batch.questions[0].prompt_ids) == tokenizer.encode(rendered)
    for option in body.questions[0].options:
        assert option.description in rendered
    assert "<start>assistant<end>" not in rendered


def test_context_cache_reused_across_questions(compiler, request_body, question):
    first = compiler.compile(request_body)
    changed = question.model_copy(update={"question": "Is there enough evidence?"})
    second = compiler.compile(request_body.model_copy(update={"questions": [changed]}))
    assert not first.tokenization_cache_hit
    assert second.tokenization_cache_hit
    assert first.prefix_key == second.prefix_key
    n = first.prefix_tokens
    assert first.questions[0].prompt_ids[:n] == second.questions[0].prompt_ids[:n]
    assert first.questions[0].prompt_ids[n:] != second.questions[0].prompt_ids[n:]


def test_aliases_extend_slot_without_merging(compiler, tokenizer):
    assert len(set(compiler.code_ids)) == len(compiler.codes)
    for code, token in zip(compiler.codes, compiler.code_ids, strict=True):
        assert tokenizer.encode(compiler.slot_tail + code) == [*compiler.slot_base, token]


def test_context_limit_rejects_instead_of_truncating(settings, tokenizer, request_body):
    settings.model.max_model_len = 512
    compiler = PromptCompiler(tokenizer, settings)
    with pytest.raises(CapacityError, match="not truncated"):
        compiler.compile(request_body.model_copy(update={"context": "a" * 2000}))


def test_request_budget_and_option_cap(settings, tokenizer, request_body):
    settings.service.max_options = 2
    with pytest.raises(CapacityError, match="options"):
        PromptCompiler(tokenizer, settings).compile(request_body)
    settings.service.max_options = 256
    settings.service.max_total_prompt_tokens = 512
    with pytest.raises(CapacityError, match="budget"):
        PromptCompiler(tokenizer, settings).compile(request_body)


def test_lru_eviction_and_oversized_prefix_not_retained(tokenizer):
    cache = PrefixTokenCache(2, 10)
    assert cache.encode("a", "abc", tokenizer)[1] is False
    assert cache.encode("a", "abc", tokenizer)[1] is True
    cache.encode("b", "def", tokenizer)
    cache.encode("c", "ghi", tokenizer)
    assert cache.encode("a", "abc", tokenizer)[1] is False
    assert cache.encode("long", "z" * 20, tokenizer)[1] is False
    assert cache.encode("long", "z" * 20, tokenizer)[1] is False


def test_incompatible_tokenizer_fails(settings, tokenizer):
    tokenizer.decode = lambda ids, **kwargs: "collision"
    with pytest.raises(TokenizerContractError, match="single-token"):
        PromptCompiler(tokenizer, settings)
