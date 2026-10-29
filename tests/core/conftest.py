"""Core-runtime test isolation: every test gets its own runs, reports, results and memory-log directories."""

from copy import deepcopy

import pytest

from ta_skills import config


@pytest.fixture(autouse=True)
def isolated_config(tmp_path, monkeypatch):
    """A private process-wide config: nothing a test writes reaches the user's home directory."""
    cfg = deepcopy(config.DEFAULT_CONFIG)
    cfg.update(
        runs_dir=str(tmp_path / "runs"),
        reports_dir=str(tmp_path / "reports"),
        results_dir=str(tmp_path / "logs"),
        data_cache_dir=str(tmp_path / "cache"),
        memory_log_path=str(tmp_path / "memory" / "trading_memory.md"),
    )
    monkeypatch.setattr(config, "_config", cfg)
    monkeypatch.delenv("TRADINGAGENTS_MAX_DEBATE_ROUNDS", raising=False)
    monkeypatch.delenv("TRADINGAGENTS_MAX_RISK_ROUNDS", raising=False)
    return cfg


@pytest.fixture(autouse=True)
def offline(monkeypatch):
    """No network: the identity lookup and the sentiment sources are stubbed."""
    from .helpers import IDENTITY, SENTIMENT_INPUTS

    monkeypatch.setattr("ta_skills.identity.resolve_instrument_identity", lambda *a, **k: IDENTITY)
    monkeypatch.setattr(
        "ta_skills.sentiment_sources.build_sentiment_inputs", lambda *a, **k: SENTIMENT_INPUTS
    )
