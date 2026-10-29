"""Instrument context (#814, #888), the output-language instruction and the absent-text markers."""

import pytest

from ta_skills.context import (
    build_instrument_context,
    get_instrument_context_from_state,
    get_language_instruction,
    get_portfolio_context_from_state,
    report_or_absent,
)


@pytest.mark.unit
class TestBuildInstrumentContext:
    def test_mentions_exact_symbol_without_identity(self):
        context = build_instrument_context("7203.T")
        assert "7203.T" in context
        assert "exchange suffix" in context
        assert "Resolved identity" not in context

    def test_injects_resolved_identity(self):
        context = build_instrument_context(
            "TOTDY", "stock",
            {
                "company_name": "TOTO LTD.",
                "sector": "Industrials",
                "industry": "Building Products & Equipment",
                "exchange": "PNK",
            },
        )
        assert "Company: TOTO LTD." in context
        assert "Industrials / Building Products & Equipment" in context
        assert "Exchange: PNK" in context
        assert "Do not substitute a different company" in context

    def test_crypto_uses_name_label_and_keeps_hint(self):
        context = build_instrument_context("BTC-USD", "crypto", {"company_name": "Bitcoin USD"})
        assert "Name: Bitcoin USD" in context
        assert "crypto asset rather than a company" in context
        assert context.startswith("The asset to analyze is `BTC-USD`.")

    def test_a_historical_run_gets_only_the_current_name(self):
        context = build_instrument_context(
            "TOTDY", "stock",
            {"company_name": "TOTO LTD.", "sector": "Industrials", "exchange": "PNK"},
            trade_date="2020-01-02",
        )
        assert "Company: TOTO LTD. (its current name, given only to identify it; on 2020-01-02" in context
        assert "Industrials" not in context and "PNK" not in context

    def test_a_sector_or_an_industry_alone_is_labelled(self):
        assert "Sector: Tech" in build_instrument_context("X", "stock", {"sector": "Tech"})
        assert "Industry: Chips" in build_instrument_context("X", "stock", {"industry": "Chips"})


@pytest.mark.unit
class TestGetInstrumentContextFromState:
    def test_prefers_precomputed_context(self):
        state = {"company_of_interest": "TOTDY", "instrument_context": "PRECOMPUTED"}
        assert get_instrument_context_from_state(state) == "PRECOMPUTED"

    def test_fallback_is_network_free_ticker_only(self):
        context = get_instrument_context_from_state({"company_of_interest": "NVDA", "asset_type": "stock"})
        assert "NVDA" in context

    def test_fallback_respects_asset_type(self):
        context = get_instrument_context_from_state(
            {"company_of_interest": "BTC-USD", "asset_type": "crypto"}
        )
        assert "crypto asset" in context


@pytest.mark.unit
class TestLanguageInstruction:
    def test_english_adds_no_tokens(self):
        assert get_language_instruction("English") == ""
        assert get_language_instruction("  english ") == ""

    def test_non_english_emits_directive(self):
        out = get_language_instruction("中文")
        assert "中文" in out
        assert "entire response" in out
        assert out.startswith(" ")  # the leading space is part of the string

    def test_the_labelled_lines_stay_as_the_format_gives_them(self):
        # The rating is read from its "**Rating**:" line; a translated label leaves
        # the reader a translated negation to find instead, and a Buy reads as Sell (#1435).
        out = get_language_instruction("中文")
        assert "**Rating**:" in out and "FINAL TRANSACTION PROPOSAL" in out

    def test_the_configured_language_is_the_default(self):
        from ta_skills.config import set_config

        set_config({"output_language": "Deutsch"})
        assert "Deutsch" in get_language_instruction()


@pytest.mark.unit
class TestMarkers:
    def test_an_absent_report_says_so(self):
        text = report_or_absent("  ", "news")
        assert text == "(No news report in this run: it is not available, not an empty finding.)"
        assert report_or_absent(" body ", "news") == "body"

    def test_absent_portfolio_is_not_a_flat_book(self):
        text = get_portfolio_context_from_state({})
        assert "not provided" in text and "do not assume a flat book" in text
        assert get_portfolio_context_from_state({"portfolio_context": "HELD"}) == "HELD"
