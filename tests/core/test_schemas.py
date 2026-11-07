"""Tests for the structured-output schemas (Research Manager, Trader, Portfolio Manager, Sentiment).

The schemas, their render functions and the field normalisation are what a
structured submission is validated and rendered with (``ta run submit``).
"""

import pytest
from pydantic import ValidationError

from ta_skills import prompts
from ta_skills.schemas import (
    PortfolioDecision,
    PortfolioRating,
    ResearchPlan,
    SentimentBand,
    SentimentReport,
    TraderAction,
    TraderProposal,
    render_pm_decision,
    render_research_plan,
    render_sentiment_report,
    render_trader_proposal,
)

# ---------------------------------------------------------------------------
# Render functions
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestRenderTraderProposal:
    def test_minimal_required_fields(self):
        p = TraderProposal(action=TraderAction.HOLD, reasoning="Balanced setup; no edge.")
        md = render_trader_proposal(p)
        assert "**Action**: Hold" in md
        assert "**Reasoning**: Balanced setup; no edge." in md
        # The trailing FINAL TRANSACTION PROPOSAL line is preserved for the
        # analyst stop-signal text and any external code that greps for it.
        assert "FINAL TRANSACTION PROPOSAL: **HOLD**" in md

    def test_optional_fields_included_when_present(self):
        p = TraderProposal(
            action=TraderAction.BUY,
            reasoning="Strong technicals + fundamentals.",
            entry_price=189.5,
            stop_loss=178.0,
            position_sizing="6% of portfolio",
        )
        md = render_trader_proposal(p)
        assert "**Action**: Buy" in md
        assert "**Entry Price**: 189.5" in md
        assert "**Stop Loss**: 178.0" in md
        assert "**Position Sizing**: 6% of portfolio" in md
        assert "FINAL TRANSACTION PROPOSAL: **BUY**" in md

    def test_optional_fields_are_named_as_not_provided(self):
        """An omitted line reads as a field nobody asked for; the reader cannot
        tell it from a level the trader declined to set."""
        p = TraderProposal(action=TraderAction.SELL, reasoning="Guidance cut.")
        md = render_trader_proposal(p)
        for field in ("Entry Price", "Stop Loss", "Position Sizing"):
            assert f"**{field}**: not provided" in md
        assert "FINAL TRANSACTION PROPOSAL: **SELL**" in md


@pytest.mark.unit
class TestNullishFloatCoercion:
    """A weak LLM may write "None"/"N/A" into an optional float field (#1058);
    coerce those to None so the structured call validates instead of erroring."""

    def test_trader_nullish_strings_coerce_to_none(self):
        for sentinel in ("None", "N/A", "null", "-", "", "TBD"):
            p = TraderProposal(
                action=TraderAction.HOLD,
                reasoning="x",
                entry_price=sentinel,
                stop_loss=sentinel,
            )
            assert p.entry_price is None
            assert p.stop_loss is None

    def test_trader_real_numeric_string_still_parses(self):
        p = TraderProposal(action=TraderAction.BUY, reasoning="x", entry_price="189.5")
        assert p.entry_price == 189.5

    def test_pm_nullish_price_target_coerces_to_none(self):
        d = PortfolioDecision(
            rating=PortfolioRating.OVERWEIGHT,
            executive_summary="s",
            investment_thesis="t",
            price_target="N/A",
        )
        assert d.price_target is None

    def test_percentage_answer_to_a_price_field_becomes_none(self):
        # The Trader is asked for concrete levels and may answer a price field
        # with a distance ("15%"), which failed the whole proposal (#1288).
        # A percentage cannot be salvaged: 15% must not become a $15 stop.
        for pct in ("15%", " 7.5% ", "-10%"):
            p = TraderProposal(
                action=TraderAction.BUY,
                reasoning="x",
                entry_price=pct,
                stop_loss=pct,
            )
            assert p.entry_price is None
            assert p.stop_loss is None

    def test_human_formatted_price_is_reduced_to_its_number(self):
        p = TraderProposal(
            action=TraderAction.BUY,
            reasoning="x",
            entry_price="$1,234.50",
            stop_loss="1,180",
        )
        assert p.entry_price == 1234.50
        assert p.stop_loss == 1180.0

    def test_one_bad_field_no_longer_fails_the_whole_proposal(self):
        # Previously a single '15%' raised, forcing a free-text retry that lost
        # the action and reasoning; now the rest of the proposal survives.
        p = TraderProposal(
            action=TraderAction.SELL,
            reasoning="downgrade on margin compression",
            entry_price="612.40",
            stop_loss="15%",
        )
        assert p.action is TraderAction.SELL
        assert p.entry_price == 612.40
        assert p.stop_loss is None


@pytest.mark.unit
class TestRenderResearchPlan:
    def test_required_fields(self):
        p = ResearchPlan(
            recommendation=PortfolioRating.OVERWEIGHT,
            rationale="Bull case carried; tailwinds intact.",
            strategic_actions="Build position over two weeks; cap at 5%.",
        )
        md = render_research_plan(p)
        assert "**Recommendation**: Overweight" in md
        assert "**Rationale**: Bull case carried" in md
        assert "**Strategic Actions**: Build position" in md

    def test_all_5_tier_ratings_render(self):
        for rating in PortfolioRating:
            p = ResearchPlan(
                recommendation=rating,
                rationale="r",
                strategic_actions="s",
            )
            md = render_research_plan(p)
            assert f"**Recommendation**: {rating.value}" in md


# ---------------------------------------------------------------------------
# Trader agent: structured happy path + fallback
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# Research Manager agent: structured happy path + fallback
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# Sentiment Analyst: schema, render, structured happy path + fallback
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestRenderSentimentReport:
    def test_header_contains_band_and_score(self):
        report = SentimentReport(
            overall_band=SentimentBand.BULLISH,
            overall_score=7.2,
            confidence="high",
            narrative="Source breakdown here.",
        )
        md = render_sentiment_report(report)
        assert "**Overall Sentiment:** **Bullish**" in md
        assert "(Score: 7.2/10)" in md

    def test_header_contains_confidence(self):
        report = SentimentReport(
            overall_band=SentimentBand.NEUTRAL,
            overall_score=5.0,
            confidence="low",
            narrative="Limited data.",
        )
        assert "**Confidence:** Low" in render_sentiment_report(report)

    def test_narrative_preserved_in_output(self):
        narrative = "## Breakdown\n\nStockTwits: 70% bullish.\n\n| Signal | Direction |\n|---|---|\n| News | Neutral |"
        report = SentimentReport(
            overall_band=SentimentBand.MILDLY_BULLISH,
            overall_score=6.0,
            confidence="medium",
            narrative=narrative,
        )
        assert narrative in render_sentiment_report(report)

    def test_all_six_bands_render(self):
        for band in SentimentBand:
            report = SentimentReport(
                overall_band=band,
                overall_score=5.0,
                confidence="medium",
                narrative="n",
            )
            assert band.value in render_sentiment_report(report)

    def test_score_out_of_range_rejected(self):
        with pytest.raises(ValidationError):
            SentimentReport(
                overall_band=SentimentBand.BULLISH,
                overall_score=11.0,
                confidence="high",
                narrative="n",
            )


@pytest.mark.unit
@pytest.mark.parametrize(
    "source",
    [
        pytest.param(
            lambda: ResearchPlan.model_fields["recommendation"].description,
            id="ResearchPlan.recommendation",
        ),
        pytest.param(
            lambda: PortfolioDecision.model_fields["rating"].description,
            id="PortfolioDecision.rating",
        ),
        pytest.param(
            lambda: prompts.load_template("research-manager"), id="research_manager prompt"
        ),
        pytest.param(
            lambda: prompts.load_template("portfolio-manager"), id="portfolio_manager prompt"
        ),
    ],
)
def test_conflict_alone_is_not_a_hold_trigger(source):
    # The debate always contains conflicting arguments, so a Hold condition that
    # conflict satisfies fires on every run and swallows directional calls
    # (#1321). All four decision sites must state the same rule.
    text = " ".join(source().split())
    assert (
        "conflict alone is not a reason to Hold" in text
        or "Conflicting arguments alone are not a reason to Hold" in text
    )
    assert "materially conflicting" not in text


@pytest.mark.unit
@pytest.mark.parametrize("written", ["150-160", "150 to 160", "around 150", "150/160", "~150"])
def test_a_price_written_as_a_range_drops_only_that_field(written):
    """Anything that is not a single number becomes None. Letting it through
    fails the whole decision's validation, and the run falls back to free text,
    losing every other field the model got right."""
    from ta_skills.schemas import PortfolioDecision, PortfolioRating

    decision = PortfolioDecision(
        rating=PortfolioRating.BUY,
        executive_summary="s",
        investment_thesis="t",
        price_target=written,
    )
    assert decision.price_target is None


@pytest.mark.unit
def test_a_price_that_is_a_number_survives():
    from ta_skills.schemas import PortfolioDecision, PortfolioRating

    decision = PortfolioDecision(
        rating=PortfolioRating.BUY,
        executive_summary="s",
        investment_thesis="t",
        price_target="$1,150.25",
    )
    assert decision.price_target == 1150.25


@pytest.mark.unit
def test_a_field_the_model_did_not_give_says_so():
    """An omitted line and a line never asked for read the same to an analyst."""
    from ta_skills.schemas import PortfolioDecision, PortfolioRating

    rendered = render_pm_decision(
        PortfolioDecision(rating=PortfolioRating.HOLD, executive_summary="s", investment_thesis="t")
    )
    assert "Price Target" in rendered and "not provided" in rendered.lower()


@pytest.mark.unit
def test_the_trader_names_the_levels_it_did_not_give():
    from ta_skills.schemas import TraderAction, TraderProposal, render_trader_proposal

    rendered = render_trader_proposal(TraderProposal(action=TraderAction.HOLD, reasoning="r"))
    for field in ("Entry Price", "Stop Loss", "Position Sizing"):
        assert field in rendered
    assert rendered.lower().count("not provided") == 3


_NARRATIVE_CASES = [
    (
        ResearchPlan,
        {
            "recommendation": "Hold",
            "rationale": "Balanced evidence.",
            "strategic_actions": "Retain current position.",
        },
        "rationale",
    ),
    (
        ResearchPlan,
        {
            "recommendation": "Hold",
            "rationale": "Balanced evidence.",
            "strategic_actions": "Retain current position.",
        },
        "strategic_actions",
    ),
    (TraderProposal, {"action": "Hold", "reasoning": "Balanced evidence."}, "reasoning"),
    (
        PortfolioDecision,
        {
            "rating": "Hold",
            "executive_summary": "Retain current position.",
            "investment_thesis": "Balanced evidence.",
        },
        "executive_summary",
    ),
    (
        PortfolioDecision,
        {
            "rating": "Hold",
            "executive_summary": "Retain current position.",
            "investment_thesis": "Balanced evidence.",
        },
        "investment_thesis",
    ),
    (
        SentimentReport,
        {
            "overall_band": "Neutral",
            "overall_score": 5.0,
            "confidence": "low",
            "narrative": "No social posts are available for this window.",
        },
        "narrative",
    ),
]


@pytest.mark.unit
@pytest.mark.parametrize("schema, fields, field", _NARRATIVE_CASES)
@pytest.mark.parametrize("empty", ["", " \n\t", "N/A", "None", "not provided"])
def test_structured_rating_cannot_hide_an_empty_narrative(schema, fields, field, empty):
    """Generated headings make the rendered report nonempty even without analysis."""
    with pytest.raises(ValidationError) as error:
        schema.model_validate({**fields, field: empty})
    assert error.value.errors()[0]["loc"] == (field,)


@pytest.mark.unit
@pytest.mark.parametrize("schema, fields, field", _NARRATIVE_CASES)
def test_required_narrative_preserves_an_explanation_of_missing_data(schema, fields, field):
    narrative = "\n  The source returned no data, so confidence is low.\n"
    parsed = schema.model_validate({**fields, field: narrative})
    assert getattr(parsed, field) == narrative
    assert schema.model_json_schema()["properties"][field]["minLength"] == 1


@pytest.mark.unit
@pytest.mark.parametrize(
    "schema, fields, price_fields",
    [
        (
            TraderProposal,
            {"action": "Buy", "reasoning": "Supported by supplied evidence."},
            ("entry_price", "stop_loss"),
        ),
        (
            PortfolioDecision,
            {
                "rating": "Buy",
                "executive_summary": "Build a small position.",
                "investment_thesis": "Supported by supplied evidence.",
            },
            ("price_target", "current_price"),
        ),
    ],
)
@pytest.mark.parametrize(
    "invalid", [float("nan"), float("inf"), float("-inf"), "NaN", "Infinity", 0, -1, True, {}, []]
)
def test_invalid_absolute_prices_drop_only_the_price_fields(schema, fields, price_fields, invalid):
    parsed = schema.model_validate({**fields, **dict.fromkeys(price_fields, invalid)})
    assert all(getattr(parsed, field) is None for field in price_fields)
    assert all(getattr(parsed, field) == value for field, value in fields.items())
    assert "NaN" not in parsed.model_dump_json()
    assert "Infinity" not in parsed.model_dump_json()


@pytest.mark.unit
@pytest.mark.parametrize(
    "written, expected",
    [
        (" HIGH ", "high"),
        ("Medium", "medium"),
        ("low", "low"),
        ("N/A", None),
        ("", None),
        ("very confident", None),
        (None, None),
    ],
)
def test_optional_pm_confidence_does_not_discard_the_decision(written, expected):
    decision = PortfolioDecision(
        rating="Hold",
        executive_summary="Retain the position.",
        investment_thesis="Balanced evidence.",
        confidence=written,
    )
    assert decision.confidence == expected


@pytest.mark.unit
@pytest.mark.parametrize(
    "written, expected",
    [
        (" 3-6 months ", "3-6 months"),
        ("N/A", None),
        ("unknown", None),
        ("not provided", None),
        (" \n", None),
        (None, None),
    ],
)
def test_optional_pm_horizon_distinguishes_missing_data(written, expected):
    decision = PortfolioDecision(
        rating="Hold",
        executive_summary="Retain the position.",
        investment_thesis="Balanced evidence.",
        time_horizon=written,
    )
    assert decision.time_horizon == expected
