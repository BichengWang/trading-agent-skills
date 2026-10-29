"""``ta`` end to end, offline: init -> next/prompt/submit for every role -> finish -> the report tree."""

import json
import re
from pathlib import Path

import pytest

import ta_skills
from ta_skills import cli
from ta_skills.memory import TradingMemoryLog

from .helpers import PM, TRADE_DATE, canned

# data-spec §4.3: the fixed heading order of complete_report.md (levels 1-3).
EXPECTED_HEADINGS = [
    "# Trading Analysis Report: NVDA",
    "## I. Analyst Team Reports",
    "### Market Analyst",
    "### Sentiment Analyst",
    "### News Analyst",
    "### Fundamentals Analyst",
    "## II. Research Team Decision",
    "### Bull Researcher",
    "### Bear Researcher",
    "### Research Manager",
    "## III. Trading Team Plan",
    "### Trader",
    "## IV. Risk Management Team Decision",
    "### Aggressive Analyst",
    "### Conservative Analyst",
    "### Neutral Analyst",
    "## V. Portfolio Manager Decision",
    "### Portfolio Manager",
]


def ta(capsys, *argv) -> tuple[int, str, str]:
    code = cli.main(list(argv))
    out = capsys.readouterr()
    return code, out.out, out.err


def headings(markdown: str) -> list[str]:
    found, fenced = [], False
    for line in markdown.splitlines():
        if line.startswith("```"):
            fenced = not fenced
        elif not fenced and re.match(r"#{1,3} ", line):
            found.append(line)
    return found


def run_everything(capsys, tmp_path, *extra_init) -> tuple[Path, list[str]]:
    code, out, err = ta(
        capsys, "run", "init", "--ticker", "NVDA", "--date", TRADE_DATE, "--provider", "omp",
        "--deep-model", "gpt-6-sol", "--quick-model", "gpt-6-luna", *extra_init,
    )
    assert code == 0, err
    run_dir = Path(out.strip())
    order = []
    while True:
        code, out, err = ta(capsys, "run", "next", "--run", str(run_dir))
        assert code == 0, err
        step = json.loads(out)
        if step["phase"] == "done":
            return run_dir, order
        for role in step["roles"]:
            turn = order.count(role) + 1
            if role != "memory":
                code, prompt, err = ta(capsys, "run", "prompt", role, "--run", str(run_dir))
                assert code == 0, err
                assert prompt.strip(), role
            ext, content = canned(role, turn)
            draft = tmp_path / f"{role}_{turn}.{ext}"
            draft.write_text(content, encoding="utf-8")
            args = ["run", "submit", role, "--run", str(run_dir)]
            code, out, err = ta(capsys, *args, *([] if role == "memory" else ["--file", str(draft)]))
            assert code == 0, err
            assert json.loads(out)["accepted"] is True
            order.append(role)


@pytest.mark.unit
def test_a_complete_run_writes_the_report_tree(capsys, tmp_path, isolated_config):
    run_dir, order = run_everything(capsys, tmp_path)
    assert order[:5] == ["market", "sentiment", "news", "fundamentals", "memory"]
    assert order[5:] == ["bull", "bear", "research_manager", "trader", "aggressive", "conservative",
                         "neutral", "portfolio_manager"]

    reports = tmp_path / "reports-out"
    code, out, err = ta(capsys, "run", "finish", "--run", str(run_dir), "--reports-dir", str(reports))
    assert code == 0, err
    result = json.loads(out)
    assert result["rating"] == "Overweight"

    report_dir = Path(result["report_dir"])
    assert report_dir.parent == reports / "NVDA"
    assert re.fullmatch(r"20250314_gpt-6-sol_\d{8}_\d{6}", report_dir.name)
    assert Path(result["complete_report"]) == report_dir / "complete_report.md"

    tree = sorted(str(p.relative_to(report_dir)) for p in report_dir.rglob("*") if p.is_file())
    assert tree == sorted([
        "complete_report.html", "complete_report.md",
        "1_analysts/market.md", "1_analysts/sentiment.md", "1_analysts/news.md",
        "1_analysts/fundamentals.md",
        "2_research/bull.md", "2_research/bear.md", "2_research/manager.md",
        "3_trading/trader.md",
        "4_risk/aggressive.md", "4_risk/conservative.md", "4_risk/neutral.md",
        "5_portfolio/decision.md",
    ])

    complete = (report_dir / "complete_report.md").read_text(encoding="utf-8")
    assert headings(complete) == EXPECTED_HEADINGS
    header = complete.split("## I. ")[0]
    assert f"- Analysis date: {TRADE_DATE}" in header
    assert "- Rating: Overweight" in header
    assert f"- trading-agent-skills {ta_skills.__version__}: omp, deep gpt-6-sol, quick gpt-6-luna" in header
    assert "TradingAgents " not in header
    assert "- Analysts: market, sentiment, news, fundamentals; research debate rounds 1, risk debate rounds 1" in header
    assert "- Data vendors: core_stock_apis yfinance" in header
    assert "- Memory log:" not in header
    # Stage files keep the agent's own text; the market analyst's own heading is demoted in the complete report.
    assert (report_dir / "1_analysts" / "market.md").read_text(encoding="utf-8").startswith("**Current Price**")
    assert "#### Market view" in complete
    assert (report_dir / "2_research" / "bull.md").read_text(encoding="utf-8") == "\nBull Analyst: bull argument, turn 1."
    assert (report_dir / "5_portfolio" / "decision.md").read_text(encoding="utf-8").startswith("**Rating**: Overweight")
    assert "NVDA" in (report_dir / "complete_report.html").read_text(encoding="utf-8")

    # The state log, as the source's _log_state wrote it.
    log = json.loads(
        (Path(isolated_config["results_dir"]) / "NVDA" / "TradingAgentsStrategy_logs"
         / f"full_states_log_{TRADE_DATE}.json").read_text(encoding="utf-8"))
    assert log["final_rating"] == "Overweight" and log["trade_date"] == TRADE_DATE
    assert log["run_settings"]["deep_think_llm"] == "gpt-6-sol"
    assert set(log["investment_debate_state"]) == {"bull_history", "bear_history", "history", "current_response"}

    # The decision is recorded for the next run's reflection.
    (entry,) = TradingMemoryLog(isolated_config).load_entries()
    assert (entry["date"], entry["ticker"], entry["rating"], entry["pending"]) == (
        TRADE_DATE, "NVDA", "Overweight", True)


@pytest.mark.unit
def test_finish_can_skip_the_html_page_and_is_repeatable(capsys, tmp_path, isolated_config):
    run_dir, _ = run_everything(capsys, tmp_path, "--analysts", "market,news", "--depth", "2")
    code, out, err = ta(capsys, "run", "finish", "--run", str(run_dir), "--reports-dir", str(tmp_path / "r"),
                        "--no-html")
    assert code == 0, err
    report_dir = Path(json.loads(out)["report_dir"])
    assert not (report_dir / "complete_report.html").exists()
    complete = (report_dir / "complete_report.md").read_text(encoding="utf-8")
    # Analysts that were not selected leave no section; the debate history shows both rounds.
    assert "### Sentiment Analyst" not in complete and "### Fundamentals Analyst" not in complete
    assert "- Analysts: market, news; research debate rounds 2, risk debate rounds 2" in complete
    assert complete.count("Bull Analyst: bull argument") == 2
    assert not (report_dir / "1_analysts" / "sentiment.md").exists()
    # Running finish again writes another report but records the decision once.
    assert ta(capsys, "run", "finish", "--run", str(run_dir), "--reports-dir", str(tmp_path / "r"))[0] == 0
    assert len(TradingMemoryLog(isolated_config).load_entries()) == 1


@pytest.mark.unit
def test_finish_before_the_run_is_done_is_an_error(capsys):
    code, out, err = ta(capsys, "run", "init", "--ticker", "NVDA", "--date", TRADE_DATE)
    run_dir = out.strip()
    code, out, err = ta(capsys, "run", "finish", "--run", run_dir)
    assert code == 1 and "not finished" in err


@pytest.mark.unit
def test_a_portfolio_decision_without_a_price_target_exits_two_and_the_run_does_not_advance(capsys, tmp_path):
    code, out, err = ta(capsys, "run", "init", "--ticker", "NVDA", "--date", TRADE_DATE)
    second = out.strip()
    # fast-forward to the portfolio manager
    while json.loads(ta(capsys, "run", "next", "--run", second)[1])["phase"] != "portfolio_manager":
        step = json.loads(ta(capsys, "run", "next", "--run", second)[1])
        for role in step["roles"]:
            ext, content = canned(role)
            draft = tmp_path / f"x_{role}.{ext}"
            draft.write_text(content, encoding="utf-8")
            args = ["run", "submit", role, "--run", second] + ([] if role == "memory" else ["--file", str(draft)])
            assert ta(capsys, *args)[0] == 0

    bad = tmp_path / "pm.json"
    bad.write_text(json.dumps({**PM, "price_target": None}), encoding="utf-8")
    code, out, err = ta(capsys, "run", "submit", "portfolio_manager", "--run", second, "--file", str(bad))
    assert code == 2 and out == ""
    assert "Price Target" in err
    assert json.loads(ta(capsys, "run", "next", "--run", second)[1]) == {
        "phase": "portfolio_manager", "roles": ["portfolio_manager"]}
    assert ta(capsys, "run", "finish", "--run", second)[0] == 1


@pytest.mark.unit
def test_errors_go_to_stderr_with_a_non_zero_status(capsys, tmp_path):
    assert ta(capsys, "run", "init", "--ticker", "NVDA", "--date", "2999-01-01")[0] == 1
    assert ta(capsys, "run", "next", "--run", str(tmp_path))[0] == 1
    code, out, err = ta(capsys, "run", "next", "--run", str(tmp_path))
    assert out == "" and "not a run directory" in err
    with pytest.raises(SystemExit) as exc:
        cli.main(["run", "prompt", "oracle", "--run", str(tmp_path)])
    assert exc.value.code == 2  # argparse: not one of the roles


@pytest.mark.unit
@pytest.mark.parametrize("argv", [
    [], ["run"], ["run", "init"], ["run", "next"], ["run", "prompt"], ["run", "submit"], ["run", "finish"],
    ["tool"], ["memory"], ["memory", "due"], ["memory", "resolve"],
])
def test_missing_arguments_are_a_usage_error(argv, capsys):
    with pytest.raises(SystemExit) as exc:
        cli.main(argv)
    assert exc.value.code == 2


@pytest.mark.unit
@pytest.mark.parametrize("argv", [
    ["--help"], ["run", "--help"], ["run", "init", "--help"], ["run", "next", "--help"],
    ["run", "prompt", "--help"], ["run", "submit", "--help"], ["run", "finish", "--help"],
    ["tool", "--help"], ["memory", "--help"], ["memory", "due", "--help"], ["memory", "resolve", "--help"],
])
def test_every_command_has_help(argv, capsys):
    with pytest.raises(SystemExit) as exc:
        cli.main(argv)
    assert exc.value.code == 0
    assert "usage: ta" in capsys.readouterr().out


@pytest.mark.unit
def test_version_flag(capsys):
    with pytest.raises(SystemExit):
        cli.main(["--version"])
    assert f"ta {ta_skills.__version__}" in capsys.readouterr().out
