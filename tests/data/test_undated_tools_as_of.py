"""Insider filings and prediction-market odds are bounded by the run's trade date.

Neither tool takes a date from the model, so the run's trade_date is injected from
graph state. Insider filings carry dates and are filtered to it; Polymarket serves
only live odds, so a historical run withholds them.
"""

from __future__ import annotations

from unittest import mock

import pandas as pd
import pytest

from ta_skills import tools
from ta_skills.data import indicators
from ta_skills.data.date_window import get_current_date
from ta_skills.data.vendors import polymarket
from ta_skills.data.vendors.yahoo import (
    fundamentals as yahoo_fundamentals,
    market as yahoo_market,
)


def _insider_frame(*dates):
    return pd.DataFrame({
        "Shares": [100] * len(dates),
        "Text": [f"Sale at price {100 + i} per share." for i in range(len(dates))],
        "Start Date": pd.to_datetime(list(dates)),
    })


def _yf_insider(frame, curr_date):
    ticker = mock.Mock(insider_transactions=frame)
    with mock.patch.object(yahoo_market.yf, "Ticker", return_value=ticker):
        return yahoo_fundamentals.get_insider_transactions("AAPL", curr_date)


@pytest.mark.unit
def test_yfinance_insider_without_a_date_is_unfiltered():
    out = _yf_insider(_insider_frame("2026-09-08", "2025-01-10"), None)
    assert "2026-09-08" in out and "2025-01-10" in out


@pytest.mark.unit
def test_polymarket_withholds_live_odds_from_a_historical_run():
    with mock.patch.object(polymarket, "_request", side_effect=AssertionError("must not fetch")):
        out = polymarket.get_prediction_markets("Fed rate cut", as_of_date="2025-06-01")
    assert "withheld" in out


@pytest.mark.unit
def test_polymarket_serves_a_current_run():
    with mock.patch.object(polymarket, "_request", return_value={"events": []}) as req:
        polymarket.get_prediction_markets("Fed rate cut", as_of_date=get_current_date())
    req.assert_called_once()


@pytest.mark.unit
@pytest.mark.parametrize("name", ["get_insider_transactions", "get_prediction_markets"])
def test_trade_date_is_injected_not_model_visible(name):
    params = {p.name for p in tools.TOOLS[name].params}
    assert "trade_date" not in params and "curr_date" not in params
    assert not {"symbol", "ticker"} & params


@pytest.mark.unit
def test_insider_transactions_are_bounded_by_the_run_trade_date():
    with mock.patch.object(tools, "route_to_vendor", return_value="ok") as routed:
        tools.call_tool("get_insider_transactions", "AAPL", "2026-09-08")
    assert routed.call_args.args == ("get_insider_transactions", "AAPL", "2026-09-08")


@pytest.mark.unit
def test_prediction_markets_are_bounded_by_the_run_trade_date():
    with mock.patch.object(tools, "route_to_vendor", return_value="ok") as routed:
        tools.call_tool("get_prediction_markets", "AAPL", "2025-06-01", topic="Fed rate cut")
    assert routed.call_args.args == ("get_prediction_markets", "Fed rate cut", None, "2025-06-01")


@pytest.mark.unit
def test_an_indicator_that_could_not_be_read_is_not_shown_as_a_blank_value():
    """The per-day fallback returned an empty string for a failed read, so the
    table rendered a row per day with nothing after the colon: an analyst reads
    that as "no value on that day" rather than "could not be obtained"."""
    from ta_skills.data.errors import VendorError

    with mock.patch.object(indicators, "get_stock_stats",
                           side_effect=RuntimeError("cache parse failed")), \
            pytest.raises(VendorError):
        indicators.get_stockstats_indicator("AAPL", "rsi", "2026-05-08")


@pytest.mark.unit
@pytest.mark.parametrize("func, args", [
    # A past date withholds the live profile before any request, so the
    # fundamentals case is exercised on the date it does fetch.
    ("get_fundamentals", ("AAPL", None)),
    # Statements are requested only for a run dated today; a past one withholds them.
    ("get_balance_sheet", ("AAPL", "annual", get_current_date())),
    ("get_cashflow", ("AAPL", "annual", get_current_date())),
    ("get_income_statement", ("AAPL", "annual", get_current_date())),
    ("get_insider_transactions", ("AAPL", get_current_date())),
])
def test_a_yfinance_failure_is_a_vendor_error_not_a_report(func, args):
    """Returning the failure as text makes the router count it as an answer, so
    the chain stops and the analyst reads the error message as if it were data.
    yfinance serves the default path, so this is the one that matters most."""
    from ta_skills.data.errors import VendorError

    with mock.patch.object(yahoo_market.yf, "Ticker", side_effect=RuntimeError("yahoo hiccup")), \
            pytest.raises(VendorError):
        getattr(yahoo_fundamentals, func)(*args)


@pytest.mark.unit
@pytest.mark.parametrize("func, args", [
    ("get_news_yfinance", ("AAPL", "2026-08-25", "2026-09-01")),
    ("get_global_news_yfinance", ("2026-09-01", 7, 5)),
])
def test_a_yfinance_news_failure_is_a_vendor_error_not_a_report(func, args):
    from ta_skills.data.errors import VendorError
    from ta_skills.data.vendors.yahoo import news as yahoo_news

    target = "Ticker" if "global" not in func else "Search"
    with mock.patch.object(yahoo_news.yf, target, side_effect=RuntimeError("yahoo hiccup")), \
            pytest.raises(VendorError):
        getattr(yahoo_news, func)(*args)


@pytest.mark.unit
def test_an_unreachable_vendor_is_not_reported_as_a_missing_symbol(monkeypatch):
    """yfinance returns an empty frame when it cannot reach Yahoo, with no
    exception. Reporting that as "no data for AAPL" tells the analyst the
    company has no balance sheet, when the truth is we could not ask."""
    import pandas as pd

    from ta_skills.data.errors import NoMarketDataError, VendorUnavailableError
    from ta_skills.data.vendors.yahoo import common

    empty = mock.Mock(quarterly_balance_sheet=pd.DataFrame(), balance_sheet=pd.DataFrame())
    monkeypatch.setattr(yahoo_market.yf, "Ticker", lambda s: empty)

    monkeypatch.setattr(common, "vendor_reachable", lambda url: False)
    with pytest.raises(VendorUnavailableError, match="unreachable"):
        yahoo_fundamentals.get_balance_sheet("AAPL", "annual", get_current_date())

    monkeypatch.setattr(common, "vendor_reachable", lambda url: True)
    with pytest.raises(NoMarketDataError):
        yahoo_fundamentals.get_balance_sheet("AAPL", "annual", get_current_date())


@pytest.mark.unit
def test_every_vendor_unavailable_says_so_rather_than_crashing(monkeypatch):
    """A throttled or unreachable chain used to raise RuntimeError('No available
    vendor'), which ends the run, and never said the vendor was the problem."""
    from ta_skills.config import set_config
    from ta_skills.data import router
    from ta_skills.data.errors import VendorUnavailableError

    def _down(*a, **k):
        raise VendorUnavailableError("Yahoo Finance is unreachable")

    set_config({"data_vendors": {"fundamental_data": "yfinance"}})
    monkeypatch.setitem(router.VENDOR_METHODS["get_balance_sheet"], "yfinance", _down)

    out = router.route_to_vendor("get_balance_sheet", "AAPL", "annual", "2026-09-01")

    assert "unavailable" in out.lower() and "unreachable" in out.lower()
    assert "delisted" not in out.lower()  # not a claim about the symbol


@pytest.mark.unit
def test_the_price_path_also_tells_an_outage_from_an_unknown_symbol(monkeypatch):
    """Prices are the most-used path, so an outage there must not read as a
    delisted symbol either."""
    import pandas as pd

    from ta_skills.data.errors import NoMarketDataError, VendorUnavailableError
    from ta_skills.data.vendors.yahoo import common

    monkeypatch.setattr(yahoo_market.yf, "Ticker", lambda s: mock.Mock(history=lambda **k: pd.DataFrame()))

    monkeypatch.setattr(common, "vendor_reachable", lambda url: False)
    with pytest.raises(VendorUnavailableError, match="unreachable"):
        yahoo_market.get_YFin_data_online("AAPL", "2026-09-01", "2026-09-10")

    monkeypatch.setattr(common, "vendor_reachable", lambda url: True)
    with pytest.raises(NoMarketDataError):
        yahoo_market.get_YFin_data_online("AAPL", "2026-09-01", "2026-09-10")


@pytest.mark.unit
@pytest.mark.parametrize("func, args", [
    pytest.param(f, a, id=f.__name__) for f, a in (
        (yahoo_market.get_YFin_data_online, ("AAPL", "2025-06-02", "2025-06-06")),
        (yahoo_fundamentals.get_balance_sheet, ("AAPL", "quarterly", "2025-06-06")),
        (yahoo_fundamentals.get_cashflow, ("AAPL", "quarterly", "2025-06-06")),
        (yahoo_fundamentals.get_income_statement, ("AAPL", "quarterly", "2025-06-06")),
        (yahoo_fundamentals.get_insider_transactions, ("AAPL", "2025-06-06")),
    )
])
def test_a_historical_run_is_not_told_todays_date(func, args):
    """A header stamped with the wall clock tells a backtest when it is really running."""
    from datetime import date

    statement = pd.DataFrame({pd.Timestamp("2025-03-31"): [1.0]}, index=["Total Assets"])
    prices = pd.DataFrame({"Open": [1.0], "High": [1.0], "Low": [1.0], "Close": [1.0], "Volume": [1]},
                          index=pd.DatetimeIndex(["2025-06-02"], name="Date"))
    ticker = mock.Mock(quarterly_balance_sheet=statement, quarterly_cashflow=statement,
                       quarterly_income_stmt=statement,
                       insider_transactions=_insider_frame("2025-05-30"),
                       history=lambda **k: prices)
    with mock.patch.object(yahoo_market.yf, "Ticker", return_value=ticker):
        out = func(*args)

    assert date.today().isoformat() not in out


def _dates_after(text: str, cutoff: str) -> list[str]:
    import re
    return [d for d in re.findall(r"\d{4}-\d{2}-\d{2}", text) if d > cutoff]


@pytest.mark.unit
def test_an_unavailable_notice_names_no_date_after_the_run():
    """A notice explaining why data is missing named where the vendor's coverage
    starts or today's date, both after a historical run's date."""
    from ta_skills.data.date_window import (
        coverage_gap,
        get_current_date,
        withhold_live_profile,
    )

    today = get_current_date()
    notices = [
        coverage_gap([pd.Timestamp(today, tz="UTC")], "2025-01-01", "2025-01-07", "Feed", "news"),
        withhold_live_profile("2025-01-07", "AAPL"),
        _yf_insider(_insider_frame(today), "2025-01-07"),
    ]
    for notice in notices:
        assert _dates_after(notice, "2025-01-07") == [], notice
