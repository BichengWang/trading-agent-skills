"""What the agents are actually told: the shape of each rendered prompt.

Ports the source's prompt-shape guards (debate prompt shape, debate opening,
prompt integrity, structured-agent prompts, news-analyst prompt) to the skill
templates: reports in the system part, per-round content in the human part, the
system part byte-stable across rounds, explicit markers for absent text.
"""

import pytest

from ta_skills import prompts, run
from ta_skills.context import NO_EXTERNAL_TOOLS, WRAP_UP, opponent_argument_or_opening

REPORT_TOKENS = (
    "INSTRUMENT-CONTEXT", "MARKET-REPORT", "SENTIMENT-REPORT", "NEWS-REPORT", "FUNDAMENTALS-REPORT",
)
RISK = [("aggressive", "Aggressive"), ("conservative", "Conservative"), ("neutral", "Neutral")]


def make_state(history="", current_response="", count=0, risk_history="", risk_count=0, **risk):
    risk_state = {
        "history": risk_history, "aggressive_history": "", "conservative_history": "",
        "neutral_history": "", "latest_speaker": "", "count": risk_count,
        "current_aggressive_response": "", "current_conservative_response": "",
        "current_neutral_response": "",
    }
    risk_state.update(risk)
    return {
        "company_of_interest": "NVDA",
        "instrument_context": "INSTRUMENT-CONTEXT",
        "asset_type": "stock",
        "trade_date": "2026-08-14",
        "market_report": "MARKET-REPORT",
        "sentiment_report": "SENTIMENT-REPORT",
        "news_report": "NEWS-REPORT",
        "fundamentals_report": "FUNDAMENTALS-REPORT",
        "investment_plan": "RM-PLAN",
        "trader_investment_plan": "TRADER-PLAN",
        "portfolio_context": "",
        "past_context": "",
        "investment_debate_state": {
            "history": history, "bull_history": "", "bear_history": "",
            "current_response": current_response, "count": count,
        },
        "risk_debate_state": risk_state,
        "run": {"output_language": "English", "max_tool_rounds": 20, "tool_calls": {}},
    }


def parts(role, state):
    system, human = prompts.render_skill(run.ROLE_SKILL[role], run.prompt_values(role, state))
    return system, human


# --- research debate ---------------------------------------------------------------------


@pytest.mark.unit
@pytest.mark.parametrize("role", ["bull", "bear"])
def test_reports_in_system_history_in_human(role):
    system, human = parts(role, make_state(history="ROUND-1-HISTORY", current_response="LAST-ARG"))
    for token in REPORT_TOKENS:
        assert token in system
        assert token not in human
    assert "ROUND-1-HISTORY" in human and "ROUND-1-HISTORY" not in system
    assert "LAST-ARG" in human and "LAST-ARG" not in system


@pytest.mark.unit
@pytest.mark.parametrize("role", ["bull", "bear"])
def test_research_system_is_byte_identical_across_rounds(role):
    first, _ = parts(role, make_state())
    second, _ = parts(role, make_state(history="\nBull: a\nBear: b", current_response="Bear: b", count=2))
    assert first == second


@pytest.mark.unit
@pytest.mark.parametrize("role", ["bull", "bear"])
def test_round_one_human_message_is_not_empty(role):
    _, human = parts(role, make_state())
    assert human.strip()  # a provider may reject empty message content


@pytest.mark.unit
@pytest.mark.parametrize("role, opponent", [("bull", "bear"), ("bear", "bull")])
def test_researcher_opening_has_no_phantom_opponent(role, opponent):
    _, human = parts(role, make_state())
    assert "has not spoken yet" in human and f"The {opponent} analyst" in human


@pytest.mark.unit
def test_researcher_passes_real_opponent_argument():
    system, human = parts("bull", make_state(current_response="Bear Analyst: valuation is stretched"))
    assert "valuation is stretched" in human
    assert "has not spoken yet" not in system + human


# --- risk debate ------------------------------------------------------------------------


@pytest.mark.unit
@pytest.mark.parametrize("role, _", RISK)
def test_plan_and_reports_in_system_history_in_human(role, _):
    system, human = parts(role, make_state(risk_history="RISK-HISTORY"))
    for token in (*REPORT_TOKENS, "TRADER-PLAN"):
        assert token in system
        assert token not in human
    assert "RISK-HISTORY" in human and "RISK-HISTORY" not in system


@pytest.mark.unit
@pytest.mark.parametrize("role, _", RISK)
def test_risk_system_is_byte_identical_across_rounds(role, _):
    first, _h = parts(role, make_state())
    second, _h = parts(
        role,
        make_state(risk_history="\nAggressive Analyst: x", risk_count=1,
                   current_aggressive_response="Aggressive Analyst: x"),
    )
    assert first == second


@pytest.mark.unit
@pytest.mark.parametrize("role, _", RISK)
def test_risk_opening_has_no_phantom_opponent(role, _):
    _system, human = parts(role, make_state())
    # Both opponent slots were empty -> two opening markers, no fabricated arguments.
    assert human.count("has not spoken yet") == 2


@pytest.mark.unit
def test_risk_passes_real_opponent_arguments():
    state = make_state(
        current_conservative_response="Conservative Analyst: trim risk",
        current_neutral_response="Neutral Analyst: hold steady",
    )
    system, human = parts("aggressive", state)
    assert "trim risk" in human and "hold steady" in human
    assert "has not spoken yet" not in system + human


@pytest.mark.unit
def test_helper_marks_empty_and_passes_through():
    assert "has not spoken yet" in opponent_argument_or_opening("", "bear analyst")
    assert opponent_argument_or_opening("  real point ", "bear") == "real point"


# --- reports that were never produced ---------------------------------------------------


@pytest.mark.unit
@pytest.mark.parametrize("role", ["bull", "bear", "aggressive", "conservative", "neutral"])
def test_a_report_that_was_never_produced_says_so(role):
    """`--analysts market` leaves three reports empty; presenting them as blank
    sections invites the model to invent the contents."""
    state = make_state()
    state.update(market_report="RSI 61, price 178.", sentiment_report="", news_report="",
                 fundamentals_report="")
    system, human = parts(role, state)
    text = system + human
    assert text.count("it is not available, not an empty finding") == 3
    assert "(No sentiment report in this run" in text
    assert "RSI 61" in text  # the report that does exist is still passed through


# --- analysts ---------------------------------------------------------------------------

ANALYSTS = ["market", "sentiment", "news", "fundamentals"]


def analyst_prompt(role, tmp_path):
    import json

    (tmp_path / "sentiment_inputs.json").write_text(json.dumps({
        "start_date": "2026-08-07", "end_date": "2026-08-14", "news_block": "n",
        "stocktwits_block": "st", "reddit_block": "rd", "subreddits": ["wallstreetbets"],
        "character": "c",
    }))
    system, human = prompts.render_skill(
        run.ROLE_SKILL[role], run.prompt_values(role, make_state(), tmp_path)
    )
    return system, human


@pytest.mark.unit
@pytest.mark.parametrize("role", ANALYSTS)
def test_an_analyst_is_not_asked_for_a_trade_call_nothing_reads(role, tmp_path):
    system, human = analyst_prompt(role, tmp_path)
    assert "FINAL TRANSACTION PROPOSAL" not in system + human


@pytest.mark.unit
@pytest.mark.parametrize("role", ANALYSTS)
def test_the_analyst_human_message_is_the_company(role, tmp_path):
    """The source's message history opens with the raw ticker string."""
    assert analyst_prompt(role, tmp_path)[1] == "NVDA"


@pytest.mark.unit
def test_news_prompt_matches_get_news_signature():
    """The prompt advertised ``get_news(query, ...)`` once, tricking the model into
    free-text query calls (#1116); the instrument comes from the run, so only the window."""
    template = prompts.load_template("news-analyst")
    assert "get_news(start_date, end_date)" in template
    assert "get_news(query" not in template


@pytest.mark.unit
@pytest.mark.parametrize("skill", ["market-analyst", "news-analyst"])
def test_tool_using_analysts_keep_their_date_guidance(skill):
    assert "tool-call date ranges" in prompts.load_template(skill)


@pytest.mark.unit
def test_the_sentiment_analyst_has_no_tools_and_no_tool_range_wording(tmp_path):
    system, human = analyst_prompt("sentiment", tmp_path)
    assert NO_EXTERNAL_TOOLS in system
    assert "tool-call date ranges" not in system
    assert run.ANALYST_TOOLS["sentiment"] == ()


@pytest.mark.unit
@pytest.mark.parametrize("role", ["market", "news", "fundamentals"])
def test_the_tool_list_is_replaced_once_the_tool_rounds_are_spent(role):
    state = make_state()
    offered, _ = parts(role, state)
    state["run"]["tool_calls"] = {role: 20}
    spent, _ = parts(role, state)
    assert ", ".join(run.ANALYST_TOOLS[role]) in offered
    assert "tools: none; your tool rounds are spent." in spent
    assert ", ".join(run.ANALYST_TOOLS[role]) not in spent


@pytest.mark.unit
def test_the_tool_budget_is_per_role():
    state = make_state()
    state["run"]["tool_calls"] = {"market": 20}
    assert "none; your tool rounds are spent" not in parts("news", state)[0]


@pytest.mark.unit
def test_the_sentiment_prompt_names_the_subreddits_searched(tmp_path):
    import json

    (tmp_path / "sentiment_inputs.json").write_text(json.dumps({
        "start_date": "a", "end_date": "b", "news_block": "n", "stocktwits_block": "s",
        "reddit_block": "r", "subreddits": ["Bitcoin", "CryptoCurrency", "CryptoMarkets"], "character": "c",
    }))
    system, _ = prompts.render_skill(
        "sentiment-analyst", run.prompt_values("sentiment", make_state(), tmp_path)
    )
    assert "r/Bitcoin, r/CryptoCurrency, r/CryptoMarkets" in system
    assert "wallstreetbets" not in system


# --- decision agents --------------------------------------------------------------------


@pytest.mark.unit
@pytest.mark.parametrize("role", ["trader", "research_manager", "portfolio_manager"])
def test_decision_prompts_state_the_no_external_tools_constraint(role):
    system, human = parts(role, make_state())
    assert NO_EXTERNAL_TOOLS in system + (human or "")


@pytest.mark.unit
def test_portfolio_manager_prompt_names_every_decision_field():
    state = make_state()
    state["market_report"] = "Verified 2026-10-01 close: $189.5"
    system, _ = parts("portfolio_manager", state)
    assert "Verified 2026-10-01 close: $189.5" in system
    for field in ("Current Price", "Price Target", "Confidence", "Time Horizon"):
        assert f"**{field}**" in system


@pytest.mark.unit
@pytest.mark.parametrize("role, must_name", [
    ("portfolio_manager", "Rating"), ("research_manager", "Recommendation"), ("trader", "Action"),
])
def test_a_decision_prompt_states_the_shape_of_its_answer(role, must_name):
    system, human = parts(role, make_state())
    text = system + (human or "")
    assert "## Output" in text
    assert f"**{must_name}**" in text.split("## Output", 1)[1]


@pytest.mark.unit
@pytest.mark.parametrize("role", ["research_manager", "portfolio_manager"])
def test_conflict_alone_is_not_a_hold_trigger(role):
    text = " ".join(prompts.load_template(run.ROLE_SKILL[role]).split())
    assert "conflict alone is not a reason to Hold" in text
    assert "materially conflicting" not in text


@pytest.mark.unit
def test_trader_grounding_only_when_there_is_a_market_report():
    with_report, human = parts("trader", make_state())
    assert "Ground concrete price levels" in with_report
    assert "Technical Market Report:\nMARKET-REPORT" in human
    state = make_state()
    state["market_report"] = "   "
    without, human = parts("trader", state)
    assert "Ground concrete price levels" not in without
    assert "Technical Market Report" not in human


@pytest.mark.unit
def test_the_portfolio_manager_gets_lessons_only_when_there_are_some():
    state = make_state()
    assert "Lessons from prior decisions" not in parts("portfolio_manager", state)[0]
    state["past_context"] = "Past analyses of NVDA"
    system, _ = parts("portfolio_manager", state)
    assert "- Lessons from prior decisions and outcomes:\nPast analyses of NVDA\n" in system
    state["market_report"] = ""
    state["fundamentals_report"] = ""
    system, _ = parts("portfolio_manager", state)
    assert "No technical market report supplied." in system
    assert "No fundamentals report supplied." in system


@pytest.mark.unit
def test_the_language_instruction_reaches_every_reporting_role(tmp_path):
    state = make_state()
    state["run"]["output_language"] = "Deutsch"
    import json

    (tmp_path / "sentiment_inputs.json").write_text(json.dumps({
        "start_date": "a", "end_date": "b", "news_block": "n", "stocktwits_block": "s",
        "reddit_block": "r", "subreddits": ["stocks"], "character": "c",
    }))
    for role in (*ANALYSTS, "bull", "bear", "research_manager", "trader", "aggressive",
                 "conservative", "neutral", "portfolio_manager"):
        system, human = prompts.render_skill(
            run.ROLE_SKILL[role], run.prompt_values(role, state, tmp_path)
        )
        assert "Write your entire response in Deutsch" in system + (human or ""), role


@pytest.mark.unit
def test_wrap_up_instruction_is_the_source_text():
    assert WRAP_UP == (
        "You have used every tool round this report allows. Write your final report now "
        "from the tool results above, and say which data you could not retrieve."
    )


@pytest.mark.unit
def test_constraint_text_is_unambiguous():
    assert "do not call external tools" in NO_EXTERNAL_TOOLS.lower()


# --- the rendering rules ----------------------------------------------------------------


@pytest.mark.unit
def test_render_substitutes_in_one_pass_and_never_re_expands_values():
    assert prompts.render("a {{x}} b", {"x": "{{y}}"}) == "a {{y}} b"


@pytest.mark.unit
def test_render_rejects_a_missing_or_an_unused_value():
    with pytest.raises(prompts.PromptError, match="no value for placeholder"):
        prompts.render("{{a}} {{b}}", {"a": "1"})
    with pytest.raises(prompts.PromptError, match="without a placeholder"):
        prompts.render("{{a}}", {"a": "1", "b": "2"})


@pytest.mark.unit
def test_every_role_has_templates_and_only_declared_placeholders():
    for role, skill in run.ROLE_SKILL.items():
        assert prompts.load_template(skill), role
    for skill in ("bull-researcher", "trader", "aggressive-risk-analyst", "decision-reflection"):
        assert prompts.load_template(skill, human=True)
    for skill in ("research-manager", "portfolio-manager"):
        assert prompts.load_template(skill, human=True) is None


@pytest.mark.unit
def test_the_skills_dir_can_be_overridden(tmp_path, monkeypatch):
    (tmp_path / "trader").mkdir()
    (tmp_path / "trader" / "prompt.md").write_text("custom {{x}}", encoding="utf-8")
    monkeypatch.setenv("TA_SKILLS_DIR", str(tmp_path))
    assert prompts.load_template("trader") == "custom {{x}}"
    monkeypatch.delenv("TA_SKILLS_DIR")
    assert prompts.skills_dir().name == "skills"
    assert (prompts.skills_dir() / "trader" / "prompt.md").is_file()


@pytest.mark.unit
def test_a_missing_template_is_an_error(tmp_path, monkeypatch):
    monkeypatch.setenv("TA_SKILLS_DIR", str(tmp_path))
    with pytest.raises(prompts.PromptError, match="not found"):
        prompts.load_template("trader")
