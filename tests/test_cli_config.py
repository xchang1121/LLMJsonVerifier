import json

import pytest
from pydantic import ValidationError

from llm_json_verifier.cli import main
from llm_json_verifier.config import EngineSettings, load_settings
from llm_json_verifier.launch import engine_command


def test_checked_in_config_and_launcher(capsys):
    settings = load_settings("configs/qwen3.8-27b.toml")
    command = engine_command(settings)
    assert "--no-enable-log-requests" in command
    assert "--calculate-kv-scales" not in command  # Removed in the pinned vLLM release.
    assert command[command.index("--logprobs-mode") + 1] == "raw_logprobs"
    assert "--enable-prefix-caching" in command
    assert main(["engine-command", "--json"]) == 0
    assert json.loads(capsys.readouterr().out) == command


def test_cache_mode_consistency():
    with pytest.raises(ValidationError):
        EngineSettings(enable_prefix_caching=False, mamba_cache_mode="align")


def test_environment_override(monkeypatch):
    monkeypatch.setenv("LLMJV_BACKEND_URL", "http://engine:8000")
    assert load_settings().backend.base_url == "http://engine:8000"
    monkeypatch.setenv("LLMJV_BACKEND_URL", "http://engine:8000/v1")
    with pytest.raises(ValidationError):
        load_settings()
