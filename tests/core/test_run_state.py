"""The run state machine: init, routing, prompts, submit validation, state writes, tools, memory."""

import json
import threading
from pathlib import Path

import pytest

from ta_skills import run
from ta_skills.rating import RATING_REVIEW, run_rating

from .helpers import PM, SENTIMENT_INPUTS, TRADE_DATE, canned, drive, new_run, submit

# --- init -------------------------------------------------------------------------------------


@pytest.mark.unit
def test_init_writes_the_source_state_field_for_field(isolated_config):
    run_dir = new_run(model_label="gpt/x:1", provider="omp", deep_model="deep", quick_model="quick")
    state = run.load_state(run_dir)

    assert set(state) == {
        "company_of_interest", "asset_type", "instrument_context", "trade_date", "market_report",
        "sentiment_report", "news_report", "fundamentals_report", "investment_debate_state",
        "investment_plan", "trader_investment_plan", "risk_debate_state", "final_trade_decision",
        "final_rating", "past_context", "memory_note", "portfolio_context", "run",
    }
    assert state["company_of_interest"] == "NVDA" and state["asset_type"] == "stock"
    assert state["trade_date"] == TRADE_DATE
    assert "Company: NVIDIA Corporation" in state["instrument_context"]
    assert state["investment_debate_state"] == {
        "bull_history": "", "bear_history": "", "history": "", "current_response": "", "count": 0,
    }
    assert state["risk_debate_state"] == {
        "aggressive_history": "", "conservative_history": "", "neutral_history": "", "history": "",
        "latest_speaker": "", "current_aggressive_response": "", "current_conservative_response": "",
        "current_neutral_response": "", "count": 0,
    }
    block = state["run"]
    assert block["selected_analysts"] == ["market", "sentiment", "news", "fundamentals"]
    assert (block["max_debate_rounds"], block["max_risk_discuss_rounds"], block["max_tool_rounds"]) == (1, 1, 20)
    assert (block["provider"], block["deep_model"], block["quick_model"], block["model_label"]) == (
        "omp", "deep", "quick", "gpt/x:1")
    assert block["done_roles"] == [] and block["tool_calls"] == {}
    assert block["output_language"] == "English" and block["created_at"]
    for sub in ("prompts", "outputs", "tools"):
        assert (run_dir / sub).is_dir()
    assert run_dir.parent.name == "NVDA" and run_dir.parent.parent == Path(isolated_config["runs_dir"])


@pytest.mark.unit
def test_init_normalizes_the_symbol_and_detects_crypto():
    state = run.load_state(new_run(ticker="btcusd"))
    assert state["company_of_interest"] == "BTC-USD" and state["asset_type"] == "crypto"
    assert state["instrument_context"].startswith("The asset to analyze is `BTC-USD`.")
    explicit = run.load_state(new_run(ticker="NVDA", asset_type="crypto"))
    assert explicit["asset_type"] == "crypto"


@pytest.mark.unit
@pytest.mark.parametrize("date, message", [
    ("2025-3-14", "YYYY-MM-DD"), ("14/03/2025", "YYYY-MM-DD"), ("2999-01-01", "future"),
])
def test_init_rejects_a_date_that_is_not_canonical_or_is_in_the_future(date, message):
    with pytest.raises(run.RunError, match=message):
        new_run(trade_date=date)


@pytest.mark.unit
def test_init_analyst_selection():
    assert run.load_state(new_run(analysts="social,market"))["run"]["selected_analysts"] == [
        "sentiment", "market"]
    assert run.parse_analysts(["news", "news"]) == ["news"]
    with pytest.raises(run.RunError, match="unknown analyst key: macro"):
        new_run(analysts="market,macro")
    with pytest.raises(run.RunError, match="at least one analyst"):
        new_run(analysts=" , ")


@pytest.mark.unit
def test_depth_sets_both_round_counts_unless_the_environment_set_one(monkeypatch, isolated_config):
    state = run.load_state(new_run(depth=3))["run"]
    assert (state["max_debate_rounds"], state["max_risk_discuss_rounds"]) == (3, 3)

    # An exported TRADINGAGENTS_MAX_DEBATE_ROUNDS (already folded into the config) beats --depth (#977).
    monkeypatch.setenv("TRADINGAGENTS_MAX_DEBATE_ROUNDS", "2")
    isolated_config["max_debate_rounds"] = 2
    state = run.load_state(new_run(depth=5))["run"]
    assert (state["max_debate_rounds"], state["max_risk_discuss_rounds"]) == (2, 5)

    # An explicit per-key flag beats the environment.
    state = run.load_state(new_run(depth=5, debate_rounds=4, risk_rounds=1))["run"]
    assert (state["max_debate_rounds"], state["max_risk_discuss_rounds"]) == (4, 1)
    with pytest.raises(run.RunError, match="at least 1"):
        new_run(depth=0)


@pytest.mark.unit
def test_init_renders_the_portfolio_once_and_rejects_a_bad_file(tmp_path):
    book = tmp_path / "book.json"
    book.write_text(json.dumps({"cash": 1000, "currency": "USD",
                                "positions": [{"ticker": "NVDA", "quantity": 12, "average_price": 90}]}))
    state = run.load_state(new_run(portfolio=str(book)))
    assert "Current position in NVDA: 12 units" in state["portfolio_context"]
    assert run.load_state(new_run())["portfolio_context"] == ""

    bad = tmp_path / "bad.json"
    bad.write_text('{"positions": [{"quantity": 5}]}')
    with pytest.raises(run.RunError, match="portfolio"):
        new_run(portfolio=str(bad))


@pytest.mark.unit
def test_two_inits_in_the_same_second_get_distinct_directories():
    assert new_run() != new_run()


# --- routing (conditional_logic.py) -------------------------------------------------------


def role_order(depth=None, **kwargs):
    run_dir = new_run(depth=depth, **kwargs)
    return drive(run_dir), run_dir


@pytest.mark.unit
def test_the_full_sequence_at_depth_one():
    order, run_dir = role_order()
    assert order == [
        "market", "sentiment", "news", "fundamentals", "memory", "bull", "bear", "research_manager",
        "trader", "aggressive", "conservative", "neutral", "portfolio_manager",
    ]
    assert run.next_phase(run.load_state(run_dir)) == ("done", [])


@pytest.mark.unit
def test_the_full_sequence_at_depth_three():
    order, _ = role_order(depth=3)
    debate = order[order.index("bull"):order.index("research_manager")]
    assert debate == ["bull", "bear"] * 3
    risk = order[order.index("aggressive"):order.index("portfolio_manager")]
    assert risk == ["aggressive", "conservative", "neutral"] * 3


@pytest.mark.unit
def test_the_debate_and_risk_round_counts_are_independent():
    order, _ = role_order(debate_rounds=2, risk_rounds=1)
    assert order.count("bull") == 2 and order.count("bear") == 2
    assert order.count("aggressive") == order.count("neutral") == 1


@pytest.mark.unit
def test_phases_are_named_in_order():
    run_dir = new_run()
    seen = []
    while True:
        phase, roles = run.next_phase(run.load_state(run_dir))
        if not seen or seen[-1] != phase:
            seen.append(phase)
        if phase == "done":
            break
        submit(run_dir, roles[0])
    assert seen == [
        "analysts", "research_debate", "research_manager", "trader", "risk_debate", "portfolio_manager",
        "done",
    ]


@pytest.mark.unit
def test_the_debate_needs_every_analyst_and_the_memory_step():
    run_dir = new_run(analysts="market,news")
    phase, roles = run.next_phase(run.load_state(run_dir))
    assert phase == "analysts" and roles == ["market", "news", "memory"]

    submit(run_dir, "market")
    submit(run_dir, "memory")
    phase, roles = run.next_phase(run.load_state(run_dir))
    assert (phase, roles) == ("analysts", ["news"])  # the slowest analyst gates the debate

    submit(run_dir, "news")
    assert run.next_phase(run.load_state(run_dir)) == ("research_debate", ["bull"])


@pytest.mark.unit
def test_a_role_that_is_not_runnable_is_refused():
    run_dir = new_run()
    with pytest.raises(run.RunError, match="not runnable now"):
        run.prompt_for(run_dir, "bull")
    with pytest.raises(run.RunError, match="not runnable now"):
        submit(run_dir, "portfolio_manager")
    with pytest.raises(run.RunError, match="unknown role"):
        run.prompt_for(run_dir, "oracle")
    submit(run_dir, "market")
    with pytest.raises(run.RunError, match="not runnable now"):
        submit(run_dir, "market")  # already submitted


def _debate_state(current_response, count, rounds=1):
    return {"investment_debate_state": {"current_response": current_response, "count": count},
            "run": {"max_debate_rounds": rounds}}


def _risk_state(latest_speaker, count, rounds=1):
    return {"risk_debate_state": {"latest_speaker": latest_speaker, "count": count},
            "run": {"max_risk_discuss_rounds": rounds}}


@pytest.mark.unit
def test_the_bull_speaks_first_and_the_aggressive_analyst_opens_the_risk_debate():
    assert run.next_debate_speaker(_debate_state("", 0)) == "bull"
    assert run.next_risk_speaker(_risk_state("", 0)) == "aggressive"


@pytest.mark.unit
def test_the_count_budget_is_checked_before_the_speaker_prefix():
    assert run.next_debate_speaker(_debate_state("Bull Analyst: x", 2)) == "research_manager"
    assert run.next_debate_speaker(_debate_state("Bear Analyst: x", 2)) == "research_manager"
    assert run.next_debate_speaker(_debate_state("Bull Analyst: x", 1)) == "bear"
    assert run.next_debate_speaker(_debate_state("Bear Analyst: x", 3, rounds=2)) == "bull"
    assert run.next_risk_speaker(_risk_state("Aggressive", 3)) == "portfolio_manager"
    assert run.next_risk_speaker(_risk_state("Aggressive", 1)) == "conservative"
    assert run.next_risk_speaker(_risk_state("Conservative", 2)) == "neutral"
    assert run.next_risk_speaker(_risk_state("Neutral", 5, rounds=2)) == "aggressive"


@pytest.mark.unit
@pytest.mark.parametrize("response", ["Optimista", "", "Bear Researcher", "Bull Researcher"])
def test_speaker_label_drift_always_routes_to_a_known_role(response):
    # The router only reads prefixes; anything else falls through to the opening speaker.
    assert run.next_debate_speaker(_debate_state(response, 1)) in {"bull", "bear"}
    assert run.next_risk_speaker(_risk_state(response, 1)) in {"aggressive", "conservative"}


# --- prompts ----------------------------------------------------------------------------------


@pytest.mark.unit
def test_prompt_is_saved_and_split_into_system_and_human(tmp_path):
    run_dir = new_run(language="Deutsch")
    text = run.prompt_for(run_dir, "market")
    system, human = text.split("\n---HUMAN---\n")
    assert human == "NVDA"
    assert "get_stock_data, get_indicators, get_verified_market_snapshot" in system
    assert f"Today's date is {TRADE_DATE}" in system and "Company: NVIDIA Corporation" in system
    assert "Write your entire response in Deutsch" in system
    assert (run_dir / "prompts" / "market.md").read_text(encoding="utf-8") == text


@pytest.mark.unit
def test_single_prompt_roles_print_no_human_marker():
    run_dir = new_run()
    drive_until(run_dir, "research_manager")
    assert "---HUMAN---" not in run.prompt_for(run_dir, "research_manager")


def drive_until(run_dir, role):
    while True:
        phase, roles = run.next_phase(run.load_state(run_dir))
        if role in roles:
            return
        for r in roles:
            submit(run_dir, r)


@pytest.mark.unit
def test_structured_roles_end_with_the_generated_json_section():
    run_dir = new_run()
    assert run.prompt_for(run_dir, "sentiment").rstrip().splitlines()[-1].startswith("- `narrative`")
    drive_until(run_dir, "trader")
    assert run.prompt_for(run_dir, "trader").rstrip().splitlines()[-1].startswith("- `position_sizing`")
    drive_until(run_dir, "portfolio_manager")
    text = run.prompt_for(run_dir, "portfolio_manager")
    assert "## JSON submission" in text and "- `price_target`" in text


@pytest.mark.unit
def test_the_sentiment_sources_are_fetched_once_per_run(monkeypatch):
    calls = []

    def fetch(*args, **kwargs):
        calls.append(args)
        return SENTIMENT_INPUTS

    monkeypatch.setattr("ta_skills.sentiment_sources.build_sentiment_inputs", fetch)
    run_dir = new_run()
    first = run.prompt_for(run_dir, "sentiment")
    assert run.prompt_for(run_dir, "sentiment") == first
    assert calls == [("NVDA", TRADE_DATE)]
    assert "NEWS-BLOCK" in first and "r/wallstreetbets, r/stocks, r/investing" in first


@pytest.mark.unit
def test_absent_reports_and_opening_markers_reach_the_debate_prompts():
    run_dir = new_run(analysts="market")
    drive_until(run_dir, "bull")
    system, human = run.prompt_for(run_dir, "bull").split("\n---HUMAN---\n")
    assert "(No sentiment report in this run: it is not available, not an empty finding.)" in system
    assert "(No news report in this run" in system and "(No fundamentals report in this run" in system
    assert "Trend is up." in system
    assert "(The bear analyst has not spoken yet — open the debate with your own case.)" in human
    assert "(the debate has not started yet)" in human


@pytest.mark.unit
def test_prompts_and_outputs_are_numbered_per_turn():
    run_dir = new_run(depth=2)
    drive_until(run_dir, "bull")
    run.prompt_for(run_dir, "bull")
    submit(run_dir, "bull", "first bull")
    run.prompt_for(run_dir, "bear")
    submit(run_dir, "bear", "first bear")
    run.prompt_for(run_dir, "bull")
    submit(run_dir, "bull", "second bull")
    names = sorted(p.name for p in (run_dir / "prompts").iterdir())
    assert {"bull_1.md", "bear_1.md", "bull_2.md"} <= set(names)
    outputs = sorted(p.name for p in (run_dir / "outputs").iterdir())
    assert {"bull_1.md", "bear_1.md", "bull_2.md", "market.md"} <= set(outputs)
    # The second bull sees the whole conversation so far, and the bear's last argument.
    text = run.prompt_for(run_dir, "bear")
    assert "Bull Analyst: second bull" in text


# --- submit: validation -----------------------------------------------------------------------


@pytest.mark.unit
@pytest.mark.parametrize("role", ["market", "news", "fundamentals"])
def test_an_empty_free_text_submission_is_rejected_with_exit_code_two(role):
    run_dir = new_run()
    with pytest.raises(run.RunError) as exc:
        submit(run_dir, role, "  \n")
    assert exc.value.code == 2 and "empty" in str(exc.value)
    assert run.load_state(run_dir)["run"]["done_roles"] == []


@pytest.mark.unit
def test_a_malformed_structured_submission_is_rejected_with_a_correction_message():
    run_dir = new_run()
    with pytest.raises(run.RunError) as exc:
        submit(run_dir, "sentiment", "not json", ext="json")
    assert exc.value.code == 2 and "one JSON object" in str(exc.value)

    bad = json.dumps({"overall_band": "Ecstatic", "overall_score": 11, "confidence": "high"})
    with pytest.raises(run.RunError) as exc:
        submit(run_dir, "sentiment", bad, ext="json")
    message = str(exc.value)
    assert exc.value.code == 2
    assert "overall_band" in message and "overall_score" in message and "narrative" in message
    assert "correction attempt" in message  # the second failure is the last one


@pytest.mark.unit
def test_the_agent_gets_one_correction_attempt():
    run_dir = new_run()
    with pytest.raises(run.RunError) as first:
        submit(run_dir, "sentiment", "{}", ext="json")
    assert "resubmit once" in str(first.value)
    assert run.load_state(run_dir)["run"]["submit_failures"] == {"sentiment": 1}
    ext, content = canned("sentiment")
    submit(run_dir, "sentiment", content, ext=ext)  # the corrected submission is accepted
    assert "sentiment" not in run.load_state(run_dir)["run"]["submit_failures"]


@pytest.mark.unit
@pytest.mark.parametrize("target", [None, "not provided", "10%", "100-120", 0, -5])
def test_the_portfolio_manager_needs_a_positive_numeric_price_target(target):
    run_dir = new_run()
    drive_until(run_dir, "portfolio_manager")
    decision = {**PM, "price_target": target}
    with pytest.raises(run.RunError) as exc:
        submit(run_dir, "portfolio_manager", json.dumps(decision), ext="json")
    assert exc.value.code == 2
    assert "positive numeric Price Target" in str(exc.value)
    assert "Correction required: the previous attempt did not produce a valid Price Target." in str(exc.value)
    assert run.next_phase(run.load_state(run_dir)) == ("portfolio_manager", ["portfolio_manager"])


@pytest.mark.unit
def test_a_trader_price_written_as_a_percentage_drops_only_that_field():
    run_dir = new_run()
    drive_until(run_dir, "trader")
    submit(run_dir, "trader", json.dumps({"action": "Sell", "reasoning": "r", "stop_loss": "5%"}), ext="json")
    plan = run.load_state(run_dir)["trader_investment_plan"]
    assert "**Action**: Sell" in plan and "**Stop Loss**: not provided" in plan
    assert plan.endswith("FINAL TRANSACTION PROPOSAL: **SELL**")


@pytest.mark.unit
def test_a_missing_submission_file_is_an_error():
    run_dir = new_run()
    with pytest.raises(run.RunError, match="cannot read submission file"):
        run.submit_role(run_dir, "market", run_dir / "drafts" / "nope.md")
    with pytest.raises(run.RunError, match="needs --file"):
        run.submit_role(run_dir, "market")


# --- submit: state writes (the source nodes' returned dicts) ---------------------------------


@pytest.mark.unit
def test_analysts_write_their_report_key_and_nothing_else():
    run_dir = new_run()
    before = run.load_state(run_dir)
    submit(run_dir, "market")
    submit(run_dir, "sentiment")
    after = run.load_state(run_dir)
    assert after["market_report"] == "**Current Price**: 101.0\n\n# Market view\n\nTrend is up."
    assert after["sentiment_report"].startswith("**Overall Sentiment:** **Bullish** (Score: 7.2/10)")
    assert "**Confidence:** Medium" in after["sentiment_report"]
    for key in ("news_report", "fundamentals_report", "investment_debate_state", "risk_debate_state"):
        assert after[key] == before[key]
    assert after["run"]["done_roles"] == ["market", "sentiment"]


@pytest.mark.unit
def test_the_debate_writes_prefixes_histories_counts_and_the_manager_overwrites_the_response():
    run_dir = new_run()
    drive_until(run_dir, "bull")
    submit(run_dir, "bull", "up\n")
    debate = run.load_state(run_dir)["investment_debate_state"]
    assert debate == {
        "history": "\nBull Analyst: up", "bull_history": "\nBull Analyst: up", "bear_history": "",
        "current_response": "Bull Analyst: up", "count": 1,
    }
    submit(run_dir, "bear", "down")
    debate = run.load_state(run_dir)["investment_debate_state"]
    assert debate == {
        "history": "\nBull Analyst: up\nBear Analyst: down", "bull_history": "\nBull Analyst: up",
        "bear_history": "\nBear Analyst: down", "current_response": "Bear Analyst: down", "count": 2,
    }
    submit(run_dir, "research_manager")
    state = run.load_state(run_dir)
    plan = state["investment_plan"]
    assert plan.startswith("**Recommendation**: Buy\n\n**Rationale**:")
    after = state["investment_debate_state"]
    assert after["current_response"] == plan  # the plan replaces the last speech
    assert after["count"] == 2 and after["history"] == debate["history"]
    assert after["bull_history"] == debate["bull_history"] and after["bear_history"] == debate["bear_history"]


@pytest.mark.unit
def test_the_risk_debate_writes_and_the_portfolio_manager_is_the_judge():
    run_dir = new_run()
    drive_until(run_dir, "aggressive")
    submit(run_dir, "aggressive", "go")
    submit(run_dir, "conservative", "stop")
    submit(run_dir, "neutral", "balance")
    risk = run.load_state(run_dir)["risk_debate_state"]
    assert risk == {
        "history": "\nAggressive Analyst: go\nConservative Analyst: stop\nNeutral Analyst: balance",
        "aggressive_history": "\nAggressive Analyst: go",
        "conservative_history": "\nConservative Analyst: stop",
        "neutral_history": "\nNeutral Analyst: balance",
        "latest_speaker": "Neutral",
        "current_aggressive_response": "Aggressive Analyst: go",
        "current_conservative_response": "Conservative Analyst: stop",
        "current_neutral_response": "Neutral Analyst: balance",
        "count": 3,
    }
    submit(run_dir, "portfolio_manager")
    state = run.load_state(run_dir)
    assert state["risk_debate_state"] == {**risk, "latest_speaker": "Judge"}
    assert state["final_trade_decision"].startswith("**Rating**: Overweight\n\n**Executive Summary**:")
    assert "**Price Target**: 120.0" in state["final_trade_decision"]


@pytest.mark.unit
def test_the_final_rating_is_the_typed_one_even_when_the_thesis_quotes_another():
    run_dir = new_run()
    drive(run_dir)
    state = run.load_state(run_dir)
    assert "rating: Hold" in state["final_trade_decision"]  # the thesis quotes someone else's rating
    assert state["final_rating"] == "Overweight"
    assert run_rating(state) == "Overweight"


@pytest.mark.unit
def test_a_state_without_the_typed_rating_reads_it_from_the_decision_or_says_review():
    assert run_rating({"final_rating": "", "final_trade_decision": "**Rating**: Sell\n\nx"}) == "Sell"
    assert run_rating({"final_rating": "", "final_trade_decision": "Trim the position."}) == RATING_REVIEW
    assert run_rating({}) == RATING_REVIEW


@pytest.mark.unit
def test_trader_and_research_manager_outputs_are_kept_as_files():
    run_dir = new_run()
    drive(run_dir)
    outputs = {p.name for p in (run_dir / "outputs").iterdir()}
    assert {"trader.json", "trader.md", "research_manager.json", "portfolio_manager.json",
            "sentiment.json", "market.md", "aggressive_1.md", "neutral_1.md"} <= outputs
    assert json.loads((run_dir / "outputs" / "trader.json").read_text())["action"] == "Buy"


# --- concurrency ------------------------------------------------------------------------------


@pytest.mark.unit
def test_concurrent_analyst_submissions_keep_every_report():
    run_dir = new_run()
    errors = []

    def work(role):
        try:
            submit(run_dir, role)
        except Exception as exc:  # noqa: BLE001 — collected for the assertion
            errors.append(exc)

    threads = [threading.Thread(target=work, args=(r,)) for r in ("market", "sentiment", "news", "fundamentals", "memory")]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    state = run.load_state(run_dir)
    assert errors == []
    assert sorted(state["run"]["done_roles"]) == sorted(["market", "sentiment", "news", "fundamentals", "memory"])
    for key in ("market_report", "sentiment_report", "news_report", "fundamentals_report"):
        assert state[key]
    assert run.next_phase(state) == ("research_debate", ["bull"])
