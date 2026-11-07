"""Rendered prompts are byte-identical to the source nodes' prompts.

``golden_prompts.json`` holds what the source agent nodes (TradingAgents
``agents/**``) sent to the model for two states, captured by running the nodes
against a recording model. The skills' templates plus ``run.prompt_values`` must
reproduce it exactly.
"""

import json
from pathlib import Path

import pytest

from ta_skills import prompts, run

GOLDEN = json.loads((Path(__file__).parent / "golden_prompts.json").read_text(encoding="utf-8"))

ANALYST_ROLES = ("market", "news", "fundamentals")
TEXT_ROLES = (
    "sentiment",
    "bull",
    "bear",
    "aggressive",
    "conservative",
    "neutral",
    "trader",
    "research_manager",
    "portfolio_manager",
)


def build_state(sc: dict, tool_calls: int = 0) -> dict:
    return {
        "company_of_interest": sc["company_of_interest"],
        "asset_type": sc["asset_type"],
        "instrument_context": sc["instrument_context"],
        "trade_date": sc["trade_date"],
        "market_report": sc["market_report"],
        "sentiment_report": sc["sentiment_report"],
        "news_report": sc["news_report"],
        "fundamentals_report": sc["fundamentals_report"],
        "investment_plan": sc["investment_plan"],
        "trader_investment_plan": sc["trader_investment_plan"],
        "portfolio_context": sc["portfolio_context"],
        "past_context": sc["past_context"],
        "investment_debate_state": dict(sc["debate"]),
        "risk_debate_state": dict(sc["risk"]),
        "run": {
            "output_language": sc["language"],
            "max_tool_rounds": 20,
            "tool_calls": dict.fromkeys(ANALYST_ROLES, tool_calls),
        },
    }


@pytest.fixture(params=sorted(GOLDEN))
def scenario(request, tmp_path):
    entry = GOLDEN[request.param]
    inputs = {**entry["state"]["sentiment"], "character": entry["sentiment_character"]}
    (tmp_path / "sentiment_inputs.json").write_text(json.dumps(inputs), encoding="utf-8")
    return entry, tmp_path


def _parts(role, state, run_dir):
    system, human = prompts.render_skill(
        run.ROLE_SKILL[role], run.prompt_values(role, state, run_dir)
    )
    return [system] if human is None else [system, human]


@pytest.mark.parametrize("role", ANALYST_ROLES)
@pytest.mark.parametrize("spent", [False, True])
def test_analyst_prompts_match_source(scenario, role, spent):
    entry, run_dir = scenario
    state = build_state(entry["state"], tool_calls=99 if spent else 0)
    golden = entry["prompts"][role + ("@spent" if spent else "")]
    assert _parts(role, state, run_dir) == golden


@pytest.mark.parametrize("role", TEXT_ROLES)
def test_prompts_match_source(scenario, role):
    entry, run_dir = scenario
    state = build_state(entry["state"])
    assert _parts(role, state, run_dir) == entry["prompts"][role]


@pytest.mark.parametrize("role", sorted(run.STRUCTURED))
def test_structured_roles_end_with_generated_json_section(scenario, role):
    entry, run_dir = scenario
    state = build_state(entry["state"])
    text = run.render_role_prompt(role, state, run_dir)
    schema, part = run.STRUCTURED[role]
    section = prompts.json_output_section(schema)
    golden = entry["prompts"][role]
    assert section in text
    if part == "human":
        assert text == prompts.join_parts(golden[0], golden[1] + "\n\n" + section)
    else:
        assert text == prompts.join_parts(
            golden[0] + "\n\n" + section, golden[1] if len(golden) > 1 else None
        )
