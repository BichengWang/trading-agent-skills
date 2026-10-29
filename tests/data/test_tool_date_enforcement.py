"""Dated tools take the analysis date from the run, not from the model.

Every point-in-time guard behind a tool trusts the date it is given. A model that
omits the date, or passes today's instead of the analysis date, would otherwise
walk past them. The run's trade_date is injected by the caller and is not a
parameter of any tool.
"""

from __future__ import annotations

import inspect
from unittest import mock

import pytest

from ta_skills import tools
from ta_skills.data.date_window import as_of, as_of_window

TRADE_DATE = "2026-08-14"


@pytest.mark.unit
@pytest.mark.parametrize("requested, expected", [
    ("2026-09-14", TRADE_DATE),     # later than the run: clamped
    ("2026-08-01", "2026-08-01"),   # earlier: narrows, allowed
    (None, TRADE_DATE),             # omitted
    ("Sept 1", TRADE_DATE),         # unparseable
    ("", TRADE_DATE),
])
def test_as_of_takes_the_earlier_date(requested, expected):
    assert as_of(requested, TRADE_DATE) == expected


@pytest.mark.unit
def test_as_of_without_a_trade_date_passes_the_request_through():
    assert as_of("2026-09-14", "") == "2026-09-14"


@pytest.mark.unit
@pytest.mark.parametrize("start, end, expected", [
    ("2026-08-01", "2026-09-14", ("2026-08-01", TRADE_DATE)),  # end clamped
    ("2026-08-01", "2026-08-10", ("2026-08-01", "2026-08-10")),  # inside: unchanged
    ("2026-09-01", "2026-09-08", ("2026-08-07", TRADE_DATE)),  # wholly later: span kept, moved back
])
def test_as_of_window(start, end, expected):
    assert as_of_window(start, end, TRADE_DATE) == expected


@pytest.mark.unit
def test_there_are_twelve_tools_with_verbatim_descriptions():
    assert list(tools.TOOLS) == [
        "get_stock_data", "get_indicators", "get_verified_market_snapshot", "get_fundamentals",
        "get_balance_sheet", "get_cashflow", "get_income_statement", "get_news",
        "get_global_news", "get_insider_transactions", "get_macro_indicators",
        "get_prediction_markets",
    ]
    for name, spec in tools.TOOLS.items():
        assert spec.name == name
        assert spec.description == inspect.getdoc(spec.func)
    assert tools.TOOLS["get_stock_data"].description.startswith(
        "Retrieve stock price data (OHLCV) for the instrument under analysis.\n"
        "Uses the configured core_stock_apis vendor.\n"
        "Args:\n"
    )
    assert tools.TOOLS["get_verified_market_snapshot"].description.startswith(
        "Deterministic verification snapshot for exact market-data claims.\n\n"
        "Returns the latest OHLCV row on or before curr_date"
    )


@pytest.mark.unit
@pytest.mark.parametrize("name", list(tools.TOOLS))
def test_trade_date_and_the_instrument_are_hidden_from_the_model(name):
    """The model once passed an indicator name as the symbol ("rsi", "atr" are
    real tickers), and another company's data reached the report."""
    names = {p.name for p in tools.TOOLS[name].params}
    assert not names & {"trade_date", "symbol", "ticker"}


@pytest.mark.unit
def test_a_tool_serves_the_run_instrument_whatever_the_model_passes():
    with mock.patch.object(tools, "route_to_vendor", return_value="ok") as routed:
        tools.call_tool("get_indicators", "NVDA", TRADE_DATE, indicator="rsi", curr_date=TRADE_DATE)
        with pytest.raises(ValueError, match="symbol"):
            tools.call_tool(
                "get_indicators", "NVDA", TRADE_DATE,
                symbol="RSI", indicator="rsi", curr_date=TRADE_DATE,
            )
    assert routed.call_args.args[1] == "NVDA"
    assert routed.call_count == 1


def _routed(name, ticker="NVDA", trade_date=TRADE_DATE, **kwargs):
    """The arguments the tool handed the router."""
    with mock.patch.object(tools, "route_to_vendor", return_value="ok") as routed:
        tools.call_tool(name, ticker, trade_date, **kwargs)
    return routed.call_args.args


@pytest.mark.unit
def test_statement_tool_with_omitted_date_uses_the_run_date():
    assert _routed("get_balance_sheet")[-1] == TRADE_DATE  # #1331: an omitted date no longer means unfiltered


@pytest.mark.unit
def test_future_curr_date_from_the_model_is_clamped():
    assert _routed("get_fundamentals", curr_date="2026-09-14") == ("get_fundamentals", "NVDA", TRADE_DATE)


@pytest.mark.unit
def test_future_window_from_the_model_is_clamped():
    args = _routed("get_stock_data", start_date="2026-08-01", end_date="2026-09-14")
    assert args == ("get_stock_data", "NVDA", "2026-08-01", TRADE_DATE)


@pytest.mark.unit
def test_every_dated_tool_clamps_a_future_date():
    future = "2026-09-14"
    assert _routed("get_news", start_date="2026-08-01", end_date=future)[-1] == TRADE_DATE
    assert _routed("get_global_news", curr_date=future)[1] == TRADE_DATE
    assert _routed("get_cashflow", curr_date=future)[-1] == TRADE_DATE
    assert _routed("get_income_statement", curr_date=future)[-1] == TRADE_DATE
    assert _routed("get_macro_indicators", indicator="cpi", curr_date=future)[2] == TRADE_DATE
    assert _routed("get_indicators", indicator="rsi", curr_date=future)[3] == TRADE_DATE


@pytest.mark.unit
def test_direct_call_without_a_trade_date_is_unchanged():
    args = _routed("get_news", "AAPL", "", start_date="2026-09-01", end_date="2026-09-08")
    assert args == ("get_news", "AAPL", "2026-09-01", "2026-09-08")


@pytest.mark.unit
@pytest.mark.parametrize("name", ["get_news", "get_stock_data"])
def test_a_start_date_that_is_not_a_date_is_sent_back_to_the_model(name):
    # A model can append a stray character to a date (#1476). No window can be
    # guessed from it, so the vendor is not called and the model is told the form.
    with mock.patch.object(tools, "route_to_vendor", return_value="ok") as routed:
        out = tools.call_tool(
            name, "NVDA", "2026-10-03", start_date="2026-09-26\u0629", end_date="2026-10-03"
        )
    assert not routed.called
    assert "YYYY-MM-DD" in out and "start_date" in out


@pytest.mark.unit
def test_a_window_whose_start_is_not_a_date_is_refused():
    with pytest.raises(ValueError, match="start_date"):
        as_of_window("2026-09-26\u0629", "2026-10-03", "2026-10-03")


@pytest.mark.unit
def test_comma_separated_indicators_are_split_and_joined():
    with mock.patch.object(tools, "route_to_vendor", side_effect=["RSI TEXT", "MFI TEXT"]) as routed:
        out = tools.call_tool(
            "get_indicators", "NVDA", TRADE_DATE, indicator=" RSI, mfi ,", curr_date=TRADE_DATE
        )
    assert out == "RSI TEXT\n\nMFI TEXT"
    assert [c.args[2] for c in routed.call_args_list] == ["rsi", "mfi"]


@pytest.mark.unit
def test_an_unsupported_indicator_is_reported_not_raised():
    with mock.patch.object(tools, "route_to_vendor", side_effect=ValueError("Indicator foo is not supported")):
        out = tools.call_tool("get_indicators", "NVDA", TRADE_DATE, indicator="foo", curr_date=TRADE_DATE)
    assert out == "Indicator foo is not supported"


@pytest.mark.unit
@pytest.mark.parametrize("asked, sent", [(3650, 30), (0, 1), (-5, 1), (7, 7), ("12", 12)])
def test_the_indicator_look_back_is_bounded(asked, sent):
    args = _routed("get_indicators", indicator="rsi", curr_date=TRADE_DATE, look_back_days=asked)
    assert args[4] == sent


@pytest.mark.unit
def test_the_snapshot_appends_its_no_prices_sentence_to_both_sentinels():
    from ta_skills.data.errors import NoMarketDataError, VendorUnavailableError

    sentence = (
        " Do not claim exact prices, price levels, or indicator values; "
        "report that verified market data is unavailable."
    )
    for error, lead in (
        (VendorUnavailableError("Yahoo Finance is unreachable"), "DATA_UNAVAILABLE:"),
        (NoMarketDataError("NVDA", "NVDA", "no price rows"), "NO_DATA_AVAILABLE:"),
    ):
        with mock.patch.object(tools, "build_verified_market_snapshot", side_effect=error):
            out = tools.call_tool(
                "get_verified_market_snapshot", "NVDA", TRADE_DATE, curr_date=TRADE_DATE
            )
        assert out.startswith(lead) and out.endswith(sentence)


@pytest.mark.unit
def test_the_snapshot_is_built_as_of_the_clamped_date():
    with mock.patch.object(tools, "build_verified_market_snapshot", return_value="ok") as built:
        tools.call_tool(
            "get_verified_market_snapshot", "NVDA", TRADE_DATE,
            curr_date="2026-09-14", look_back_days=10,
        )
    assert built.call_args.args == ("NVDA", TRADE_DATE, 10)


@pytest.mark.unit
def test_a_vendor_error_comes_back_as_text_for_the_agent():
    with mock.patch.object(tools, "route_to_vendor", side_effect=RuntimeError("boom")):
        out = tools.call_tool("get_fundamentals", "NVDA", TRADE_DATE, curr_date=TRADE_DATE)
    assert out.startswith("Error: RuntimeError('boom')")


@pytest.mark.unit
def test_caller_mistakes_raise():
    with pytest.raises(ValueError, match="Unknown tool"):
        tools.call_tool("get_nothing", "NVDA", TRADE_DATE)
    with pytest.raises(ValueError, match="requires parameter 'indicator'"):
        tools.call_tool("get_indicators", "NVDA", TRADE_DATE, curr_date=TRADE_DATE)
    with pytest.raises(ValueError, match="look_back_days must be int"):
        tools.call_tool(
            "get_indicators", "NVDA", TRADE_DATE,
            indicator="rsi", curr_date=TRADE_DATE, look_back_days="lots",
        )


@pytest.mark.unit
def test_string_arguments_from_a_cli_are_coerced():
    args = _routed("get_global_news", curr_date=TRADE_DATE, look_back_days="3", limit="4")
    assert args == ("get_global_news", TRADE_DATE, 3, 4)
