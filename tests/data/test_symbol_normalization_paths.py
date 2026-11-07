"""Symbol normalization must apply on every yfinance path, not just price fetch.

Regression tests for #983 (instrument identity), #984 (reflection returns), and
the news path: a broker symbol like XAUUSD must resolve to the same Yahoo symbol
(GC=F) that the price path uses, so identity, realized-return, and news lookups
hit the right instrument instead of failing/mismatching.
"""

import ta_skills.data.vendors.yahoo.market as yahoo_market
import ta_skills.data.vendors.yahoo.news as ynews
import ta_skills.identity as au


def test_identity_lookup_normalizes_symbol(monkeypatch):
    seen = {}

    class FakeTicker:
        def __init__(self, symbol):
            seen["symbol"] = symbol

        @property
        def info(self):
            return {"longName": "Gold Futures", "quoteType": "FUTURE"}

    monkeypatch.setattr(yahoo_market.yf, "Ticker", FakeTicker)
    au._identity.cache_clear()

    identity = au.resolve_instrument_identity("XAUUSD")

    assert seen["symbol"] == "GC=F"  # normalized, not the raw broker symbol
    assert identity["company_name"] == "Gold Futures"


def test_news_lookup_normalizes_symbol(monkeypatch):
    seen = {}

    class FakeTicker:
        def __init__(self, symbol):
            seen["query"] = symbol

        def get_news(self, **kwargs):
            return []

    class FakeSearch:
        def __init__(self, query, **k):
            seen["searched"] = query
            self.news = [
                {
                    "title": "Gold",
                    "publisher": "P",
                    "link": "l",
                    "providerPublishTime": 1736150400,
                    "relatedTickers": ["GC=F"],
                }
            ]
            self.quotes = [{"symbol": "GC=F"}]

    monkeypatch.setattr(ynews.yf, "Ticker", FakeTicker)
    monkeypatch.setattr(ynews.yf, "Search", FakeSearch)
    monkeypatch.setattr(ynews, "yf_retry", lambda fn: fn())

    out = ynews.get_news_yfinance("XAUUSD", "2025-01-01", "2025-01-10")

    assert seen["query"] == "GC=F"  # news queried with the canonical symbol
    assert seen["searched"] == "GC=F"
    assert "XAUUSD" in out  # the user's ticker stays in the report
    assert "GC=F" in out  # provenance noted
