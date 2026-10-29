"""Decision-field parsing: the price target the portfolio manager must give, and how it is read."""

import pytest

from ta_skills.report_fields import (
    MissingPriceTargetError,
    extract_price_target,
    field,
    parse_money,
    require_price_target,
)
from ta_skills.schemas import PortfolioDecision, render_pm_decision


@pytest.mark.unit
def test_decision_parser_handles_table_style_final_plan():
    decision = (
        "| Parameter | Decision |\n"
        "|---|---|\n"
        "| **Rating** | **Overweight** |\n"
        "| **Near-term target** | **$487** (January high) |\n"
        "\n"
        "- 200-SMA reclaim fails on a two-session basis within ~60 days\n"
    )

    assert field(decision, "Rating") == "Overweight"
    assert extract_price_target(decision) == 487.0


@pytest.mark.unit
def test_decision_parser_accepts_target_price_alias():
    assert extract_price_target("**Target Price**: $212.50\n") == 212.5
    assert extract_price_target("| **Fair Value** | **$198** |\n") == 198.0


@pytest.mark.unit
@pytest.mark.parametrize("value", [
    "n/a", "not provided", "2026-10-01", "15%", "150-160", "150 to 160", "nan", "inf", "6 months",
    "20 basis points", "15 percent", "6-month target", "6 trading days",
])
def test_price_parser_does_not_turn_non_prices_into_quotes(value):
    assert parse_money(value) is None


@pytest.mark.unit
def test_next_line_target_does_not_consume_another_field():
    assert extract_price_target("**Price Target**:\n$120") == 120
    assert extract_price_target("**Price Target**:\n**Time Horizon**: 6 months") is None


@pytest.mark.unit
def test_small_structured_target_survives_completion_guard():
    rendered = render_pm_decision(PortfolioDecision(
        rating="Hold", executive_summary="Retain current position.",
        investment_thesis="Supported micro-price objective.", price_target=1e-5))
    require_price_target(rendered)
    assert extract_price_target(rendered) == 1e-5
    assert parse_money(f"{1e-5:.8f}".rstrip("0")) == 1e-5


@pytest.mark.unit
@pytest.mark.parametrize("decision", ["", "**Rating**: Buy", "**Price Target**: not provided",
                                      "**Price Target**: 100-120", "**Price Target**: 0"])
def test_a_decision_without_a_positive_numeric_target_is_incomplete(decision):
    with pytest.raises(MissingPriceTargetError, match="positive numeric Price Target"):
        require_price_target(decision)
    assert issubclass(MissingPriceTargetError, ValueError)


@pytest.mark.unit
def test_the_missing_target_message_is_the_sources():
    with pytest.raises(MissingPriceTargetError) as exc:
        require_price_target("nothing")
    assert str(exc.value) == (
        "Portfolio decision requires a positive numeric Price Target supported by the "
        "supplied evidence; report remains incomplete."
    )
