"""Every Yahoo call has a hard deadline: a stalled connection becomes an outage, not a hang."""

import threading
import time

import pytest

from ta_skills.data import router
from ta_skills.data.errors import VendorUnavailableError
from ta_skills.data.vendors.yahoo import common


@pytest.fixture
def short_deadline(monkeypatch):
    monkeypatch.setenv("TRADINGAGENTS_YFINANCE_TIMEOUT", "0.1")
    monkeypatch.setattr(common.time, "sleep", lambda seconds: None)


@pytest.mark.unit
def test_the_default_deadline_is_twenty_seconds(monkeypatch):
    monkeypatch.delenv("TRADINGAGENTS_YFINANCE_TIMEOUT", raising=False)
    assert common.yfinance_timeout() == 20.0
    monkeypatch.setenv("TRADINGAGENTS_YFINANCE_TIMEOUT", "5")
    assert common.yfinance_timeout() == 5.0


@pytest.mark.unit
def test_a_call_that_outlasts_the_deadline_is_a_vendor_outage(short_deadline):
    release = threading.Event()
    try:
        started = time.monotonic()
        with pytest.raises(VendorUnavailableError, match="exceeded"):
            common.yf_retry(lambda: release.wait(5))
        assert time.monotonic() - started < 2
    finally:
        release.set()


@pytest.mark.unit
def test_a_fast_call_returns_its_value_and_a_raised_error_keeps_its_type(short_deadline):
    assert common.yf_retry(lambda: 42) == 42

    def boom():
        raise ConnectionError("down")

    with pytest.raises(VendorUnavailableError, match="ConnectionError"):
        common.yf_retry(boom)


@pytest.mark.unit
def test_a_stalled_price_request_is_reported_as_unavailable_not_as_missing_data(
    short_deadline, monkeypatch
):
    from ta_skills.config import set_config
    from ta_skills.data.vendors.yahoo import market

    release = threading.Event()

    class Stalled:
        def __init__(self, symbol):
            pass

        def history(self, **kwargs):
            release.wait(5)

    monkeypatch.setattr(market.yf, "Ticker", Stalled)
    set_config({"data_vendors": {"core_stock_apis": "yfinance"}})
    try:
        out = router.route_to_vendor("get_stock_data", "AAPL", "2026-09-01", "2026-09-10")
    finally:
        release.set()
    assert out.startswith("DATA_UNAVAILABLE:") and "exceeded" in out
