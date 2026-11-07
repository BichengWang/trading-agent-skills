"""Tests for deterministic instrument-identity resolution (#814) and asset-type detection
on the canonical symbol (#981/#982)."""

import unittest
from unittest.mock import patch

import pytest

from ta_skills.data.symbols import normalize_symbol
from ta_skills.identity import _identity, detect_asset_type, resolve_instrument_identity


@pytest.mark.unit
class ResolveInstrumentIdentityTests(unittest.TestCase):
    def setUp(self):
        _identity.cache_clear()

    def test_resolves_company_metadata_from_yfinance(self):
        with patch("ta_skills.data.vendors.yahoo.market.yf.Ticker") as mock:
            mock.return_value.info = {
                "longName": "TOTO LTD.",
                "shortName": "TOTO",
                "sector": "Industrials",
                "industry": "Building Products & Equipment",
                "exchange": "PNK",
                "quoteType": "EQUITY",
            }
            identity = resolve_instrument_identity("totdy")
        mock.assert_called_once_with("TOTDY")
        self.assertEqual(identity["company_name"], "TOTO LTD.")
        self.assertEqual(identity["sector"], "Industrials")
        self.assertEqual(identity["industry"], "Building Products & Equipment")
        self.assertEqual(identity["exchange"], "PNK")

    def test_falls_back_to_short_name(self):
        with patch("ta_skills.data.vendors.yahoo.market.yf.Ticker") as mock:
            mock.return_value.info = {"shortName": "TOTO", "sector": "Industrials"}
            identity = resolve_instrument_identity("TOTDY")
        self.assertEqual(identity["company_name"], "TOTO")

    def test_skips_placeholder_values(self):
        with patch("ta_skills.data.vendors.yahoo.market.yf.Ticker") as mock:
            mock.return_value.info = {"longName": "  ", "sector": "None", "industry": "n/a"}
            identity = resolve_instrument_identity("TOTDY")
        self.assertIsNone(identity)

    def test_fails_open_on_exception(self):
        with patch(
            "ta_skills.data.vendors.yahoo.market.yf.Ticker",
            side_effect=RuntimeError("rate limited"),
        ):
            self.assertIsNone(resolve_instrument_identity("TOTDY"))

    def test_a_failed_lookup_is_asked_again(self):
        # A long process (a backtest) must not run every later cell without an
        # identity because one request failed.
        with patch("ta_skills.data.vendors.yahoo.market.yf.Ticker") as mock:
            type(mock.return_value).info = property(
                lambda self: (_ for _ in ()).throw(TimeoutError())
            )
            self.assertIsNone(resolve_instrument_identity("TOTDY"))
            type(mock.return_value).info = {"longName": "TOTO LTD."}
            self.assertEqual(resolve_instrument_identity("TOTDY")["company_name"], "TOTO LTD.")

    def test_result_is_cached(self):
        with patch("ta_skills.data.vendors.yahoo.market.yf.Ticker") as mock:
            mock.return_value.info = {"longName": "TOTO LTD."}
            first = resolve_instrument_identity("TOTDY")
            second = resolve_instrument_identity("TOTDY")
        mock.assert_called_once()  # second call served from cache
        self.assertEqual(first, second)


@pytest.mark.unit
def test_run_context_arguments_do_not_change_the_lookup():
    _identity.cache_clear()
    with patch("ta_skills.data.vendors.yahoo.market.yf.Ticker") as mock:
        mock.return_value.info = {"longName": "Bitcoin USD", "quoteType": "CRYPTOCURRENCY"}
        identity = resolve_instrument_identity("BTC-USD", "crypto", "2026-01-02")
    mock.assert_called_once_with("BTC-USD")
    assert identity == {"company_name": "Bitcoin USD", "quote_type": "CRYPTOCURRENCY"}


# --- #982: stablecoin-quoted crypto normalizes to Yahoo's -USD pair ---
@pytest.mark.parametrize(
    "raw,expected",
    [
        ("BTCUSD", "BTC-USD"),
        ("BTCUSDT", "BTC-USD"),
        ("BTC-USDT", "BTC-USD"),
        ("BTC-USDC", "BTC-USD"),
        ("ethusdt", "ETH-USD"),
        # non-crypto must be untouched
        ("AAPL", "AAPL"),
        ("GC=F", "GC=F"),
        ("600519.SS", "600519.SS"),
        ("EURUSD", "EURUSD=X"),
    ],
)
def test_normalize_symbol_crypto_and_passthrough(raw, expected):
    assert normalize_symbol(raw) == expected


# --- #981/#982: asset-type classified on the canonical symbol ---
@pytest.mark.parametrize(
    "raw,expected",
    [
        ("BTCUSD", "crypto"),
        ("BTC-USDT", "crypto"),
        ("BTC-USD", "crypto"),
        ("ETHUSD", "crypto"),
        ("AAPL", "stock"),
        ("GC=F", "stock"),
        ("XAUUSD", "stock"),
        ("600519.SS", "stock"),
        ("SPY", "stock"),
        ("^GSPC", "stock"),
    ],
)
def test_detect_asset_type(raw, expected):
    assert detect_asset_type(raw) == expected


def test_hong_kong_codes_are_padded_before_the_asset_check():
    # The data path pads HK codes; the CLI used to classify the unpadded symbol.
    assert normalize_symbol("700.HK") == "0700.HK"
    assert detect_asset_type("700.HK") == "stock"


def test_normalize_symbol_preserves_exchange_suffix():
    assert normalize_symbol(" cnc.to ") == "CNC.TO"
