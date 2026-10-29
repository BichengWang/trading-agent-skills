"""Portfolio context: what the caller holds, threaded into the decision agents.

Decisions were made with no knowledge of the current book, so "add to a full
position" and "open a new one" read alike. The context is optional and carries
three distinct states: a position, a flat book, and no context at all. Nothing
may present the third as the second. The research team stays blind so the bull
and bear cases are not anchored by the caller's position.
"""

from __future__ import annotations

import json

import pytest

from ta_skills.context import get_portfolio_context_from_state
from ta_skills.portfolio import PortfolioContext, load_portfolio

HOLDING = {
    "cash": 25000.0,
    "currency": "USD",
    "positions": [
        {"ticker": "AAPL", "quantity": 120, "average_price": 150.0},
        {"ticker": "MSFT", "quantity": 10},
    ],
}


@pytest.mark.unit
def test_position_in_the_analyzed_instrument_leads_the_render():
    text = PortfolioContext.model_validate(HOLDING).render("AAPL")
    assert "120" in text and "150" in text
    assert "MSFT" in text and "25,000" in text and "USD" in text


@pytest.mark.unit
def test_flat_book_says_no_position_rather_than_omitting_it():
    text = PortfolioContext.model_validate({"cash": 1000.0, "positions": []}).render("AAPL")
    assert "No current position in AAPL" in text


@pytest.mark.unit
def test_a_ticker_held_under_another_spelling_is_matched():
    text = PortfolioContext.model_validate({"positions": [{"ticker": "aapl", "quantity": 5}]}).render("AAPL")
    assert "No current position" not in text and "5" in text


@pytest.mark.unit
def test_absent_context_is_reported_as_not_provided():
    notice = get_portfolio_context_from_state({"company_of_interest": "AAPL"})
    assert "not provided" in notice.lower()
    assert "no position" not in notice.lower()  # missing must not read as flat


@pytest.mark.unit
def test_rendered_context_reaches_the_agents_from_state():
    block = get_portfolio_context_from_state({"portfolio_context": "Portfolio: flat", "company_of_interest": "AAPL"})
    assert block == "Portfolio: flat"


@pytest.mark.unit
def test_load_rejects_a_malformed_file_with_a_clear_error(tmp_path):
    bad = tmp_path / "p.json"
    bad.write_text(json.dumps({"positions": [{"quantity": 5}]}))
    with pytest.raises(ValueError, match="portfolio"):
        load_portfolio(bad)


@pytest.mark.unit
def test_load_reads_a_valid_file(tmp_path):
    good = tmp_path / "p.json"
    good.write_text(json.dumps(HOLDING))
    assert load_portfolio(good).positions[0].ticker == "AAPL"
















@pytest.mark.unit
def test_research_layer_sizes_against_a_standard_allocation():
    """The research team is blind to the book, so its plan cannot promise
    position-relative sizing: it sizes against a standard allocation instead."""
    from ta_skills.schemas import ResearchPlan

    description = ResearchPlan.model_fields["strategic_actions"].description
    assert "standard allocation" in description
    assert "does not see the caller's holdings" in description


@pytest.mark.unit
def test_partial_portfolio_states_only_what_it_was_given():
    """Cash omitted is not cash zero; the line is absent rather than invented."""
    text = PortfolioContext.model_validate({"positions": [{"ticker": "AAPL", "quantity": 5}]}).render("AAPL")
    assert "Cash" not in text
    assert "5" in text






@pytest.mark.unit
@pytest.mark.parametrize("role", ["trader", "aggressive", "conservative", "neutral", "portfolio_manager"])
def test_decision_agents_see_the_portfolio(role):
    """The prompt each decision agent gets carries the portfolio block."""
    from ta_skills import prompts, run

    state = {
        "company_of_interest": "AAPL", "trade_date": "2026-08-14", "asset_type": "stock",
        "instrument_context": "", "market_report": "M", "sentiment_report": "S",
        "news_report": "N", "fundamentals_report": "F", "investment_plan": "P",
        "trader_investment_plan": "T", "past_context": "",
        "portfolio_context": "PORTFOLIO_BLOCK_MARKER",
        "investment_debate_state": {"history": "", "count": 0},
        "risk_debate_state": {"history": "", "latest_speaker": "", "count": 0,
                              "aggressive_history": "", "conservative_history": "", "neutral_history": "",
                              "current_aggressive_response": "", "current_conservative_response": "",
                              "current_neutral_response": ""},
        "run": {"output_language": "English", "max_tool_rounds": 20, "tool_calls": {}},
    }
    system, human = prompts.render_skill(run.ROLE_SKILL[role], run.prompt_values(role, state))
    assert "PORTFOLIO_BLOCK_MARKER" in system + (human or ""), f"{role} prompt lacks the portfolio block"


@pytest.mark.unit
@pytest.mark.parametrize("role", ["bull", "bear", "research_manager"])
def test_research_team_stays_blind_to_the_portfolio(role):
    from ta_skills import prompts, run

    skill = run.ROLE_SKILL[role]
    used = prompts.placeholders(prompts.load_template(skill))
    human = prompts.load_template(skill, human=True)
    if human:
        used |= prompts.placeholders(human)
    assert "portfolio_context" not in used
