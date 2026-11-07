"""Tests for TRADINGAGENTS_* env-var overlay onto DEFAULT_CONFIG."""

from __future__ import annotations

import os

import pytest

import ta_skills.config as default_config_module


def _config_with_env(monkeypatch, **overrides):
    """The defaults as the environment given here would set them."""
    for key in list(default_config_module._ENV_OVERRIDES):
        monkeypatch.delenv(key, raising=False)
    for key, val in overrides.items():
        monkeypatch.setenv(key, val)
    return default_config_module.build_default_config()


def test_no_env_uses_built_in_defaults(monkeypatch):
    config = _config_with_env(monkeypatch)
    assert config["max_debate_rounds"] == 1
    assert config["max_risk_discuss_rounds"] == 1
    assert config["max_tool_rounds"] == 20
    assert config["output_language"] == "English"
    assert config["sentiment_include_reddit"] is True
    assert config["holding_period_days"] == 5
    assert config["benchmark_ticker"] is None
    assert config["data_vendors"]["fundamental_data"] == "sec_edgar,yfinance"


def test_the_llm_provider_keys_are_not_carried():
    config = default_config_module.build_default_config()
    for key in (
        "llm_provider",
        "deep_think_llm",
        "quick_think_llm",
        "backend_url",
        "checkpoint_enabled",
        "temperature",
        "max_recur_limit",
    ):
        assert key not in config


def test_string_overrides(monkeypatch):
    config = _config_with_env(
        monkeypatch,
        TRADINGAGENTS_OUTPUT_LANGUAGE="Chinese",
        TRADINGAGENTS_BENCHMARK_TICKER="QQQ",
    )
    assert config["output_language"] == "Chinese"
    assert config["benchmark_ticker"] == "QQQ"


def test_int_coercion(monkeypatch):
    config = _config_with_env(
        monkeypatch,
        TRADINGAGENTS_MAX_DEBATE_ROUNDS="3",
        TRADINGAGENTS_MAX_RISK_ROUNDS="2",
        TRADINGAGENTS_MAX_TOOL_ROUNDS="7",
    )
    assert config["max_debate_rounds"] == 3
    assert isinstance(config["max_debate_rounds"], int)
    assert config["max_risk_discuss_rounds"] == 2
    assert isinstance(config["max_risk_discuss_rounds"], int)
    assert config["max_tool_rounds"] == 7


@pytest.mark.parametrize(
    "raw,expected",
    [
        ("true", True),
        ("True", True),
        ("1", True),
        ("yes", True),
        ("on", True),
        ("false", False),
        ("False", False),
        ("0", False),
        ("no", False),
        ("off", False),
    ],
)
def test_bool_coercion(monkeypatch, raw, expected):
    config = _config_with_env(monkeypatch, TRADINGAGENTS_SENTIMENT_INCLUDE_REDDIT=raw)
    assert config["sentiment_include_reddit"] is expected


def test_sentiment_include_reddit_override(monkeypatch):
    config = _config_with_env(
        monkeypatch,
        TRADINGAGENTS_SENTIMENT_INCLUDE_REDDIT="0",
    )
    assert config["sentiment_include_reddit"] is False


def test_empty_env_value_is_passthrough(monkeypatch):
    """Empty TRADINGAGENTS_* values must not clobber the built-in default."""
    config = _config_with_env(
        monkeypatch,
        TRADINGAGENTS_OUTPUT_LANGUAGE="",
        TRADINGAGENTS_MAX_DEBATE_ROUNDS="",
    )
    assert config["output_language"] == "English"
    assert config["max_debate_rounds"] == 1


def test_empty_path_value_keeps_the_default_path(monkeypatch):
    """.env.example lists the path variables blank; uncommenting one made the
    path empty, and the graph failed creating its directories."""
    config = _config_with_env(
        monkeypatch,
        TRADINGAGENTS_RESULTS_DIR="",
        TRADINGAGENTS_CACHE_DIR="",
        TRADINGAGENTS_RUNS_DIR="",
        TRADINGAGENTS_REPORTS_DIR="",
        TRADINGAGENTS_MEMORY_LOG_PATH="",
    )
    home = default_config_module._TRADINGAGENTS_HOME
    assert config["results_dir"] == os.path.join(home, "logs")
    assert config["data_cache_dir"] == os.path.join(home, "cache")
    assert config["runs_dir"] == os.path.join(home, "runs")
    assert config["reports_dir"] == os.path.join(home, "reports")
    assert config["memory_log_path"] == os.path.join(home, "memory", "trading_memory.md")


def test_invalid_int_raises(monkeypatch):
    """Garbage int values should surface a ValueError at import, not silently misconfigure."""
    monkeypatch.setenv("TRADINGAGENTS_MAX_DEBATE_ROUNDS", "not-a-number")
    with pytest.raises(ValueError, match="TRADINGAGENTS_MAX_DEBATE_ROUNDS"):
        default_config_module.build_default_config()


@pytest.mark.parametrize("bad", ["treu", "flase", "maybe", "2", "enabled"])
def test_invalid_bool_raises(monkeypatch, bad):
    """A misspelled boolean must fail loudly (like ints) instead of silently False."""
    monkeypatch.setenv("TRADINGAGENTS_SENTIMENT_INCLUDE_REDDIT", bad)
    with pytest.raises(ValueError, match="TRADINGAGENTS_SENTIMENT_INCLUDE_REDDIT"):
        default_config_module.build_default_config()


def test_unknown_env_var_is_ignored(monkeypatch):
    """Env vars outside _ENV_OVERRIDES must not bleed into DEFAULT_CONFIG."""
    config = _config_with_env(
        monkeypatch,
        TRADINGAGENTS_NONEXISTENT_KEY="oops",
    )
    assert "nonexistent_key" not in config


def test_the_runs_dir_and_its_override(monkeypatch):
    monkeypatch.delenv("TRADINGAGENTS_RUNS_DIR", raising=False)
    config = _config_with_env(monkeypatch)
    assert config["runs_dir"] == os.path.join(default_config_module._TRADINGAGENTS_HOME, "runs")
    monkeypatch.setenv("TRADINGAGENTS_RUNS_DIR", "/tmp/elsewhere")
    assert default_config_module.build_default_config()["runs_dir"] == "/tmp/elsewhere"
