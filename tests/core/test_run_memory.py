"""The memory flow: ``ta memory due`` -> reflection -> ``ta memory resolve`` -> ``ta run submit memory``."""

import json

import pandas as pd
import pytest

from ta_skills import cli, run
from ta_skills.memory import TradingMemoryLog, settlement

from .helpers import drive, new_run, submit

DECISION = "**Rating**: Buy\n\n**Price Target**: 150\n\nAccumulate."


def prices(monkeypatch, *symbols):
    idx = pd.bdate_range("2025-01-02", periods=60)
    series = pd.Series([100.0 + i for i in range(60)], index=idx)

    def get_closes(symbol, start_date, end_date):
        return series[(series.index >= start_date) & (series.index < end_date)]

    monkeypatch.setattr(settlement, "get_closes", get_closes)


@pytest.fixture
def log(isolated_config):
    return TradingMemoryLog(isolated_config)


def draft(run_dir, text):
    path = run_dir / "drafts" / "reflection.md"
    path.write_text(text, encoding="utf-8")
    return path


@pytest.mark.unit
def test_due_returns_the_rendered_reflection_prompt_for_each_settled_decision(log, monkeypatch):
    log.store_decision("NVDA", "2025-01-06", DECISION)
    log.store_decision("AAPL", "2025-03-24", DECISION)  # window not traded by the series' end
    prices(monkeypatch)
    run_dir = new_run()

    due = run.memory_due(run_dir)

    assert [(d["ticker"], d["date"]) for d in due] == [("NVDA", "2025-01-06")]
    item = due[0]
    assert item["holding_days"] == 5 and item["benchmark"] == "SPY"
    assert item["raw"] == pytest.approx(5 / 103) and item["alpha"] == pytest.approx(0.0)
    system, human = item["prompt"].split("\n---HUMAN---\n")
    assert system == (
        "You are a trading analyst reviewing your own past decision now that the outcome is known.\n"
        "The outcome covers 5 trading days from the first close after the analysis date, "
        "which may be shorter than the horizon the decision was written for.\n"
        "Write exactly 2-4 sentences of plain prose (no bullets, no headers, no markdown).\n\n"
        "Cover in order:\n"
        "1. What the 5-day alpha shows about the directional call (cite the figure), "
        "and say so plainly if the window is too short to judge the thesis.\n"
        "2. Which part of the investment thesis this window supports or undercuts.\n"
        "3. One concrete lesson to apply to the next similar analysis.\n\n"
        "Be specific and terse. Your output will be stored verbatim in a memory log "
        "and re-read by future analysts, so every word must earn its place."
    )
    assert human == (
        f"Raw return over 5 trading days: {item['raw']:+.1%}\n"
        f"Alpha vs SPY: {item['alpha']:+.1%}\n\n"
        f"Final Decision:\n{DECISION}"
    )
    assert len(log.get_pending_entries()) == 2  # nothing is written until a reflection is resolved
    assert run.load_state(run_dir)["run"]["memory_due"][0]["resolution_date"]


@pytest.mark.unit
def test_resolve_writes_the_outcome_and_reflection_once(log, monkeypatch):
    log.store_decision("NVDA", "2025-01-06", DECISION)
    prices(monkeypatch)
    run_dir = new_run()
    run.memory_due(run_dir)

    assert run.memory_resolve(run_dir, "NVDA", "2025-01-06", draft(run_dir, "Held up.\n")) is True
    assert run.memory_resolve(run_dir, "NVDA", "2025-01-06", draft(run_dir, "Held up.\n")) is False

    (entry,) = log.load_entries()
    assert not entry["pending"] and entry["reflection"] == "Held up."
    assert entry["raw"] == "+4.9%" and entry["alpha"] == "+0.0%" and entry["holding"] == "5d"
    assert entry["resolved"] == "2025-01-14"


@pytest.mark.unit
def test_resolve_needs_a_due_entry_and_a_reflection(log, monkeypatch):
    log.store_decision("NVDA", "2025-01-06", DECISION)
    prices(monkeypatch)
    run_dir = new_run()
    run.memory_due(run_dir)
    with pytest.raises(run.RunError, match="not in this run's due list"):
        run.memory_resolve(run_dir, "NVDA", "2024-12-01", draft(run_dir, "x"))
    with pytest.raises(run.RunError) as exc:
        run.memory_resolve(run_dir, "NVDA", "2025-01-06", draft(run_dir, "  "))
    assert exc.value.code == 2


@pytest.mark.unit
def test_submit_memory_computes_past_context_and_marks_the_step_done(log, monkeypatch):
    log.store_decision("NVDA", "2025-01-06", DECISION)
    prices(monkeypatch)
    run_dir = new_run()
    run.memory_due(run_dir)
    run.memory_resolve(run_dir, "NVDA", "2025-01-06", draft(run_dir, "Held up."))

    result = submit(run_dir, "memory")

    state = run.load_state(run_dir)
    assert result["accepted"] and "memory" in state["run"]["done_roles"]
    assert state["past_context"].startswith("Past analyses of NVDA (most recent first):")
    assert "REFLECTION:\nHeld up." in state["past_context"]
    assert state["memory_note"] == ""


@pytest.mark.unit
def test_a_historical_run_only_learns_from_outcomes_known_by_its_date(log, monkeypatch):
    log.store_decision("NVDA", "2025-03-10", DECISION)
    log.update_with_outcome(
        "NVDA", "2025-03-10", 0.05, 0.02, 5, "Too late to know.", resolution_date="2025-03-17"
    )
    run_dir = new_run()  # dated 2025-03-14, in the past

    submit(run_dir, "memory")

    assert run.load_state(run_dir)["past_context"] == ""


@pytest.mark.unit
def test_an_unreadable_price_feed_is_noted_and_the_decision_stays_pending(log, monkeypatch):
    log.store_decision("NVDA", "2025-01-06", DECISION)

    def down(*args, **kwargs):
        raise RuntimeError("vendor outage")

    monkeypatch.setattr(settlement, "get_closes", down)
    run_dir = new_run()

    assert run.memory_due(run_dir) == []
    submit(run_dir, "memory")

    state = run.load_state(run_dir)
    assert (
        state["memory_note"] == "1 past decision(s) could not be settled this run and stay pending."
    )
    assert len(log.get_pending_entries()) == 1


@pytest.mark.unit
def test_a_due_decision_left_without_a_reflection_is_noted(log, monkeypatch):
    log.store_decision("NVDA", "2025-01-06", DECISION)
    prices(monkeypatch)
    run_dir = new_run()
    assert len(run.memory_due(run_dir)) == 1

    submit(run_dir, "memory")  # the reflection was never resolved

    assert "1 past decision(s) could not be settled" in run.load_state(run_dir)["memory_note"]


@pytest.mark.unit
def test_a_failing_settlement_pass_never_ends_the_run(log, monkeypatch):
    def boom(*args, **kwargs):
        raise OSError("disk")

    monkeypatch.setattr(run, "due_outcomes", boom)
    run_dir = new_run()
    assert run.memory_due(run_dir) == []
    submit(run_dir, "memory")
    assert run.load_state(run_dir)["memory_note"] == (
        "Past decisions could not be settled this run (OSError); they stay pending."
    )


@pytest.mark.unit
def test_the_memory_role_has_no_prompt_of_its_own():
    run_dir = new_run()
    with pytest.raises(run.RunError, match="ta memory due"):
        run.prompt_for(run_dir, "memory")


@pytest.mark.unit
def test_memory_cli_round_trip(log, monkeypatch, capsys):
    log.store_decision("NVDA", "2025-01-06", DECISION)
    prices(monkeypatch)
    run_dir = new_run()

    assert cli.main(["memory", "due", "--run", str(run_dir)]) == 0
    due = json.loads(capsys.readouterr().out)
    assert set(due[0]) == {"ticker", "date", "raw", "alpha", "holding_days", "benchmark", "prompt"}

    # `ta run prompt memory` is the same work list
    assert cli.main(["run", "prompt", "memory", "--run", str(run_dir)]) == 0
    assert json.loads(capsys.readouterr().out) == due

    reflection = draft(run_dir, "Lesson learned.")
    args = [
        "memory",
        "resolve",
        "--run",
        str(run_dir),
        "--ticker",
        "NVDA",
        "--date",
        "2025-01-06",
        "--file",
        str(reflection),
    ]
    assert cli.main(args) == 0
    assert json.loads(capsys.readouterr().out)["resolved"] is True
    assert cli.main(args) == 0
    assert json.loads(capsys.readouterr().out)["resolved"] is False  # idempotent

    assert cli.main(["run", "submit", "memory", "--run", str(run_dir)]) == 0
    assert "Lesson learned." in run.load_state(run_dir)["past_context"]
    assert cli.main(["run", "submit", "memory", "--run", str(run_dir)]) == 1  # already done


@pytest.mark.unit
def test_only_the_portfolio_manager_receives_the_lessons(log, monkeypatch):
    log.store_decision("NVDA", "2025-01-06", DECISION)
    prices(monkeypatch)
    run_dir = new_run()
    run.memory_due(run_dir)
    run.memory_resolve(run_dir, "NVDA", "2025-01-06", draft(run_dir, "Lesson learned."))
    drive(run_dir)  # canned outputs, memory included -> past_context set, PM done

    state = run.load_state(run_dir)
    assert "Lesson learned." in state["past_context"]
    for role in ("bull", "research_manager", "trader", "aggressive", "portfolio_manager"):
        values = run.prompt_values(role, state, run_dir)
        carried = [k for k, v in values.items() if "Lesson learned." in str(v)]
        assert carried == (["lessons_line"] if role == "portfolio_manager" else [])
