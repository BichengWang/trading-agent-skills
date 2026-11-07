"""``ta tool``: the analysts' tool budget, tool sets, dates injected from the state, the call log."""

import pytest

from ta_skills import cli, run
from ta_skills.context import WRAP_UP

from .helpers import TRADE_DATE, new_run


@pytest.fixture
def calls(monkeypatch):
    seen = []

    def fake_call_tool(name, ticker, trade_date, **kwargs):
        seen.append((name, ticker, trade_date, kwargs))
        return f"{name} output for {ticker}"

    monkeypatch.setattr("ta_skills.tools.call_tool", fake_call_tool)
    return seen


@pytest.mark.unit
def test_the_ticker_and_date_come_from_the_run_not_the_model(calls):
    run_dir = new_run()
    out = run.run_tool(
        run_dir, "get_stock_data", "market", {"start_date": "2025-01-01", "end_date": "2099-01-01"}
    )
    assert out == "get_stock_data output for NVDA"
    assert calls == [
        (
            "get_stock_data",
            "NVDA",
            TRADE_DATE,
            {"start_date": "2025-01-01", "end_date": "2099-01-01"},
        )
    ]
    assert run.load_state(run_dir)["run"]["tool_calls"] == {"market": 1}
    log = (run_dir / "tools" / "market.log").read_text(encoding="utf-8")
    assert "get_stock_data(start_date=2025-01-01, end_date=2099-01-01)" in log
    assert "get_stock_data output for NVDA" in log


@pytest.mark.unit
def test_once_the_tool_rounds_are_spent_the_wrap_up_text_replaces_the_tool(calls):
    run_dir = new_run()
    state_run = run.load_state(run_dir)["run"]
    for _ in range(state_run["max_tool_rounds"]):
        assert run.run_tool(
            run_dir, "get_news", "news", {"start_date": "a", "end_date": "b"}
        ).startswith("get_news")
    assert len(calls) == 20

    assert (
        run.run_tool(run_dir, "get_news", "news", {"start_date": "a", "end_date": "b"}) == WRAP_UP
    )
    assert len(calls) == 20 and run.load_state(run_dir)["run"]["tool_calls"]["news"] == 20
    # The budget is per analyst, and the next prompt names no tools.
    assert run.run_tool(
        run_dir, "get_stock_data", "market", {"start_date": "a", "end_date": "b"}
    ).startswith("get_stock_data")
    assert "tools: none; your tool rounds are spent." in run.prompt_for(run_dir, "news")


@pytest.mark.unit
def test_an_analyst_may_only_use_its_own_tools(calls):
    run_dir = new_run()
    with pytest.raises(run.RunError, match="not a tool of the market analyst"):
        run.run_tool(run_dir, "get_news", "market", {})
    with pytest.raises(run.RunError, match="not a tool of the sentiment analyst"):
        run.run_tool(run_dir, "get_news", "sentiment", {})
    with pytest.raises(run.RunError, match="not an analyst role"):
        run.run_tool(run_dir, "get_news", "bull", {})
    assert calls == [] and run.load_state(run_dir)["run"]["tool_calls"] == {}


@pytest.mark.unit
def test_cli_tool_call_and_list(calls, capsys):
    run_dir = new_run()
    code = cli.main(
        [
            "tool",
            "get_indicators",
            "--run",
            str(run_dir),
            "--role",
            "market",
            "--indicator",
            "rsi",
            "--curr_date",
            TRADE_DATE,
            "--look_back_days",
            "10",
        ]
    )
    assert code == 0
    assert capsys.readouterr().out == "get_indicators output for NVDA\n"
    assert calls[0][3] == {"indicator": "rsi", "curr_date": TRADE_DATE, "look_back_days": "10"}

    assert (
        cli.main(
            ["tool", "get_indicators", "--run", str(run_dir), "--role", "market", "--bogus", "1"]
        )
        == 1
    )
    assert "no parameter --bogus" in capsys.readouterr().err
    assert cli.main(["tool", "get_stock_data", "--role", "market"]) == 1
    assert "needs --run DIR and --role ROLE" in capsys.readouterr().err

    assert cli.main(["tool", "list", "--role", "news"]) == 0
    listing = capsys.readouterr().out
    for name in run.ANALYST_TOOLS["news"]:
        assert name in listing
    assert "get_stock_data" not in listing
    assert cli.main(["tool", "list", "--role", "sentiment"]) == 0
    assert "no tools" in capsys.readouterr().out
    assert cli.main(["tool", "list"]) == 0
    full = capsys.readouterr().out
    assert all(name in full for names in run.ANALYST_TOOLS.values() for name in names)


@pytest.mark.unit
def test_tool_descriptions_are_the_tools_own_docstrings(capsys):
    from ta_skills.tools import TOOLS

    assert cli.main(["tool", "list", "--role", "market"]) == 0
    listing = capsys.readouterr().out
    first_line = TOOLS["get_stock_data"].description.splitlines()[0]
    assert first_line in listing
