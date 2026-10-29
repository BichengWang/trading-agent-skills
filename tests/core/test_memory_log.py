"""Tests for TradingMemoryLog — storage, deferred reflection, settlement, past-context retrieval."""

from unittest.mock import MagicMock, patch

import pandas as pd
import pytest

from ta_skills.memory import TradingMemoryLog, settlement

_SEP = TradingMemoryLog._SEPARATOR

DECISION_BUY = "Rating: Buy\nEnter at $189-192, 6% portfolio cap."
DECISION_OVERWEIGHT = (
    "Rating: Overweight\n"
    "Executive Summary: Moderate position, await confirmation.\n"
    "Investment Thesis: Strong fundamentals but near-term headwinds."
)
DECISION_SELL = "Rating: Sell\nExit position immediately."
DECISION_NO_RATING = (
    "Executive Summary: Complex situation with multiple competing factors.\n"
    "Investment Thesis: No clear directional signal at this time."
)


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

def make_log(tmp_path, filename="trading_memory.md"):
    config = {"memory_log_path": str(tmp_path / filename)}
    return TradingMemoryLog(config)


def _seed_completed(tmp_path, ticker, date, decision_text, reflection_text, filename="trading_memory.md"):
    """Write a completed entry directly to file, bypassing the API."""
    entry = (
        f"[{date} | {ticker} | Buy | +1.0% | +0.5% | 5d]\n\n"
        f"DECISION:\n{decision_text}\n\n"
        f"REFLECTION:\n{reflection_text}"
        + _SEP
    )
    with open(tmp_path / filename, "a", encoding="utf-8") as f:
        f.write(entry)


def _resolve_entry(log, ticker, date, decision, reflection="Good call."):
    """Store a decision then immediately resolve it via the API."""
    log.store_decision(ticker, date, decision)
    log.update_with_outcome(ticker, date, 0.05, 0.02, 5, reflection)


def _price_df(prices, start="2026-01-05"):
    """Minimal DataFrame matching yfinance .history() output shape.

    Uses a DatetimeIndex like real yfinance output, so resolution-date
    extraction (stock.index[holding_days]) works (#1251).
    """
    idx = pd.date_range(start=start, periods=len(prices), freq="D")
    return pd.DataFrame({"Close": prices}, index=idx)






# ---------------------------------------------------------------------------
# Core: storage and read path
# ---------------------------------------------------------------------------

class TestTradingMemoryLogCore:

    def test_store_creates_file(self, tmp_path):
        log = make_log(tmp_path)
        assert not (tmp_path / "trading_memory.md").exists()
        log.store_decision("NVDA", "2026-01-10", DECISION_BUY)
        assert (tmp_path / "trading_memory.md").exists()

    def test_store_appends_not_overwrites(self, tmp_path):
        log = make_log(tmp_path)
        log.store_decision("NVDA", "2026-01-10", DECISION_BUY)
        log.store_decision("AAPL", "2026-01-11", DECISION_OVERWEIGHT)
        entries = log.load_entries()
        assert len(entries) == 2
        assert entries[0]["ticker"] == "NVDA"
        assert entries[1]["ticker"] == "AAPL"

    def test_store_decision_idempotent(self, tmp_path):
        """Calling store_decision twice with same (ticker, date) stores only one entry."""
        log = make_log(tmp_path)
        log.store_decision("NVDA", "2026-01-10", DECISION_BUY)
        log.store_decision("NVDA", "2026-01-10", DECISION_BUY)
        assert len(log.load_entries()) == 1

    def test_store_decision_idempotent_after_the_entry_resolves(self, tmp_path):
        """A settled entry still blocks a duplicate.

        The guard matched only pending entries, so re-running a ticker and date
        whose outcome had already been settled appended a second entry: the same
        decision counted twice in past context and in any aggregate over the log.
        """
        log = make_log(tmp_path)
        log.store_decision("NVDA", "2026-01-10", DECISION_BUY)
        log.update_with_outcome("NVDA", "2026-01-10", 0.05, 0.02, 5, "worked", "2026-01-17")

        log.store_decision("NVDA", "2026-01-10", DECISION_BUY)

        entries = log.load_entries()
        assert len(entries) == 1
        assert entries[0]["pending"] is False  # the settled record is kept, not replaced


    def test_pending_tag_format(self, tmp_path):
        log = make_log(tmp_path)
        log.store_decision("NVDA", "2026-01-10", DECISION_BUY)
        text = (tmp_path / "trading_memory.md").read_text(encoding="utf-8")
        assert "[2026-01-10 | NVDA | Buy | pending]" in text

    # Rating parsing

    def test_rating_parsed_buy(self, tmp_path):
        log = make_log(tmp_path)
        log.store_decision("NVDA", "2026-01-10", DECISION_BUY)
        assert log.load_entries()[0]["rating"] == "Buy"

    def test_rating_parsed_overweight(self, tmp_path):
        log = make_log(tmp_path)
        log.store_decision("AAPL", "2026-01-11", DECISION_OVERWEIGHT)
        assert log.load_entries()[0]["rating"] == "Overweight"

    def test_an_unreadable_decision_is_tagged_for_review(self, tmp_path):
        """Not a Hold: a fabricated rating is quoted back to the next run as a
        call that was never made, and counted in the backtest figures."""
        from ta_skills.rating import RATING_REVIEW

        log = make_log(tmp_path)
        log.store_decision("MSFT", "2026-01-12", DECISION_NO_RATING)
        assert log.load_entries()[0]["rating"] == RATING_REVIEW

    def test_rating_priority_over_prose(self, tmp_path):
        """'Rating: X' label wins even when an opposing rating word appears earlier in prose."""
        decision = (
            "The sell thesis is weak. The hold case is marginal.\n\n"
            "Rating: Buy\n\n"
            "Executive Summary: Strong fundamentals support the position."
        )
        log = make_log(tmp_path)
        log.store_decision("NVDA", "2026-01-10", decision)
        assert log.load_entries()[0]["rating"] == "Buy"

    # Delimiter robustness

    def test_decision_with_markdown_separator(self, tmp_path):
        """LLM decision containing '---' must not corrupt the entry."""
        decision = "Rating: Buy\n\n---\n\nRisk: elevated volatility."
        log = make_log(tmp_path)
        log.store_decision("NVDA", "2026-01-10", decision)
        entries = log.load_entries()
        assert len(entries) == 1
        assert "Risk: elevated volatility" in entries[0]["decision"]

    # load_entries

    def test_load_entries_empty_file(self, tmp_path):
        log = make_log(tmp_path)
        assert log.load_entries() == []

    def test_load_entries_single(self, tmp_path):
        log = make_log(tmp_path)
        log.store_decision("NVDA", "2026-01-10", DECISION_BUY)
        entries = log.load_entries()
        assert len(entries) == 1
        e = entries[0]
        assert e["date"] == "2026-01-10"
        assert e["ticker"] == "NVDA"
        assert e["rating"] == "Buy"
        assert e["pending"] is True
        assert e["raw"] is None

    def test_load_entries_multiple(self, tmp_path):
        log = make_log(tmp_path)
        log.store_decision("NVDA", "2026-01-10", DECISION_BUY)
        log.store_decision("AAPL", "2026-01-11", DECISION_OVERWEIGHT)
        log.store_decision("MSFT", "2026-01-12", DECISION_NO_RATING)
        entries = log.load_entries()
        assert len(entries) == 3
        assert [e["ticker"] for e in entries] == ["NVDA", "AAPL", "MSFT"]

    def test_decision_content_preserved(self, tmp_path):
        log = make_log(tmp_path)
        log.store_decision("NVDA", "2026-01-10", DECISION_BUY)
        assert log.load_entries()[0]["decision"] == DECISION_BUY.strip()

    # get_pending_entries

    def test_get_pending_returns_pending_only(self, tmp_path):
        log = make_log(tmp_path)
        _seed_completed(tmp_path, "NVDA", "2026-01-05", "Buy NVDA.", "Correct.")
        log.store_decision("NVDA", "2026-01-10", DECISION_BUY)
        pending = log.get_pending_entries()
        assert len(pending) == 1
        assert pending[0]["ticker"] == "NVDA"
        assert pending[0]["date"] == "2026-01-10"

    # get_past_context

    def test_get_past_context_empty(self, tmp_path):
        log = make_log(tmp_path)
        assert log.get_past_context("NVDA") == ""

    def test_get_past_context_pending_excluded(self, tmp_path):
        log = make_log(tmp_path)
        log.store_decision("NVDA", "2026-01-10", DECISION_BUY)
        assert log.get_past_context("NVDA") == ""

    def test_get_past_context_same_ticker(self, tmp_path):
        log = make_log(tmp_path)
        _seed_completed(tmp_path, "NVDA", "2026-01-05", "Buy NVDA — AI capex thesis intact.", "Directionally correct.")
        ctx = log.get_past_context("NVDA")
        assert "Past analyses of NVDA" in ctx
        assert "Buy NVDA" in ctx

    def test_get_past_context_cross_ticker(self, tmp_path):
        log = make_log(tmp_path)
        _seed_completed(tmp_path, "AAPL", "2026-01-05", "Buy AAPL — Services growth.", "Correct.")
        ctx = log.get_past_context("NVDA")
        assert "Recent cross-ticker lessons" in ctx
        assert "Past analyses of NVDA" not in ctx

    def test_n_same_limit_respected(self, tmp_path):
        """Only the n_same most recent same-ticker entries are included."""
        log = make_log(tmp_path)
        for i in range(6):
            _seed_completed(tmp_path, "NVDA", f"2026-01-{i+1:02d}", f"Buy entry {i}.", "Correct.")
        ctx = log.get_past_context("NVDA", n_same=5)
        assert "Buy entry 0" not in ctx
        assert "Buy entry 5" in ctx

    def test_n_cross_limit_respected(self, tmp_path):
        """Only the n_cross most recent cross-ticker entries are included."""
        log = make_log(tmp_path)
        for i, ticker in enumerate(["AAPL", "MSFT", "GOOG", "META"]):
            _seed_completed(tmp_path, ticker, f"2026-01-{i+1:02d}", f"Buy {ticker}.", "Correct.")
        ctx = log.get_past_context("NVDA", n_cross=3)
        assert "AAPL" not in ctx
        assert "META" in ctx

    # No-op when config is None

    def test_no_log_path_is_noop(self):
        log = TradingMemoryLog(config=None)
        log.store_decision("NVDA", "2026-01-10", DECISION_BUY)
        assert log.load_entries() == []
        assert log.get_past_context("NVDA") == ""

    # Rotation: opt-in cap on resolved entries

    def test_rotation_disabled_by_default(self, tmp_path):
        """Without max_entries, all resolved entries are kept."""
        log = make_log(tmp_path)
        for i in range(7):
            _resolve_entry(log, "NVDA", f"2026-01-{i+1:02d}", DECISION_BUY, f"Lesson {i}.")
        assert len(log.load_entries()) == 7

    def test_rotation_prunes_oldest_resolved(self, tmp_path):
        """When max_entries is set and exceeded, oldest resolved entries are pruned."""
        log = TradingMemoryLog({
            "memory_log_path": str(tmp_path / "trading_memory.md"),
            "memory_log_max_entries": 3,
        })
        # Resolve 5 entries; rotation should keep only the 3 most recent.
        for i in range(5):
            _resolve_entry(log, "NVDA", f"2026-01-{i+1:02d}", DECISION_BUY, f"Lesson {i}.")
        entries = log.load_entries()
        assert len(entries) == 3
        # Confirm the OLDEST were dropped, not the newest.
        dates = [e["date"] for e in entries]
        assert dates == ["2026-01-03", "2026-01-04", "2026-01-05"]

    def test_rotation_never_prunes_pending(self, tmp_path):
        """Pending entries (unresolved) are kept regardless of the cap."""
        log = TradingMemoryLog({
            "memory_log_path": str(tmp_path / "trading_memory.md"),
            "memory_log_max_entries": 2,
        })
        # 3 resolved + 2 pending. With cap=2, only 2 resolved survive; both pending stay.
        for i in range(3):
            _resolve_entry(log, "NVDA", f"2026-01-{i+1:02d}", DECISION_BUY, f"Resolved {i}.")
        log.store_decision("NVDA", "2026-02-01", DECISION_BUY)
        log.store_decision("NVDA", "2026-02-02", DECISION_OVERWEIGHT)
        # Trigger rotation by resolving one more entry — pending entries must stay.
        _resolve_entry(log, "NVDA", "2026-01-04", DECISION_BUY, "Resolved 3.")
        entries = log.load_entries()
        pending = [e for e in entries if e["pending"]]
        resolved = [e for e in entries if not e["pending"]]
        assert len(pending) == 2, "pending entries must never be pruned"
        assert len(resolved) == 2, f"expected 2 resolved after rotation, got {len(resolved)}"

    def test_rotation_under_cap_is_noop(self, tmp_path):
        """No rotation when resolved count <= max_entries."""
        log = TradingMemoryLog({
            "memory_log_path": str(tmp_path / "trading_memory.md"),
            "memory_log_max_entries": 10,
        })
        for i in range(3):
            _resolve_entry(log, "NVDA", f"2026-01-{i+1:02d}", DECISION_BUY, f"Lesson {i}.")
        assert len(log.load_entries()) == 3

    # Rating parsing: markdown bold and numbered list formats

    def test_rating_parsed_from_bold_markdown(self, tmp_path):
        """**Rating**: Buy — markdown bold around the label must not prevent parsing."""
        decision = "**Rating**: Buy\nEnter at $190."
        log = make_log(tmp_path)
        log.store_decision("NVDA", "2026-01-10", decision)
        assert log.load_entries()[0]["rating"] == "Buy"

    def test_rating_parsed_from_bold_value(self, tmp_path):
        """Rating: **Sell** — markdown bold around the value must not prevent parsing."""
        decision = "Rating: **Sell**\nExit immediately."
        log = make_log(tmp_path)
        log.store_decision("NVDA", "2026-01-10", decision)
        assert log.load_entries()[0]["rating"] == "Sell"

    def test_rating_label_wins_over_prose_with_markdown(self, tmp_path):
        """Rating: **Sell** must win even when prose contains a conflicting rating word."""
        decision = (
            "The buy thesis is weakened by guidance.\n"
            "Rating: **Sell**\n"
            "Exit before earnings."
        )
        log = make_log(tmp_path)
        log.store_decision("NVDA", "2026-01-10", decision)
        assert log.load_entries()[0]["rating"] == "Sell"

    def test_rating_parsed_from_numbered_list(self, tmp_path):
        """1. Rating: Buy — numbered list prefix must not prevent parsing."""
        decision = "1. Rating: Buy\nEnter at $190."
        log = make_log(tmp_path)
        log.store_decision("NVDA", "2026-01-10", decision)
        assert log.load_entries()[0]["rating"] == "Buy"


# ---------------------------------------------------------------------------
# Deferred reflection: update_with_outcome, fetch_returns
# ---------------------------------------------------------------------------

class TestDeferredReflection:

    # update_with_outcome

    def test_update_replaces_pending_tag(self, tmp_path):
        log = make_log(tmp_path)
        log.store_decision("NVDA", "2026-01-10", DECISION_BUY)
        log.update_with_outcome("NVDA", "2026-01-10", 0.042, 0.021, 5, "Momentum confirmed.")
        text = (tmp_path / "trading_memory.md").read_text(encoding="utf-8")
        assert "[2026-01-10 | NVDA | Buy | pending]" not in text
        assert "+4.2%" in text
        assert "+2.1%" in text
        assert "5d" in text

    def test_update_appends_reflection(self, tmp_path):
        log = make_log(tmp_path)
        log.store_decision("NVDA", "2026-01-10", DECISION_BUY)
        log.update_with_outcome("NVDA", "2026-01-10", 0.042, 0.021, 5, "Momentum confirmed.")
        entries = log.load_entries()
        assert len(entries) == 1
        e = entries[0]
        assert e["pending"] is False
        assert e["reflection"] == "Momentum confirmed."
        assert e["decision"] == DECISION_BUY.strip()

    def test_update_preserves_other_entries(self, tmp_path):
        """Only the matching entry is modified; all other entries remain unchanged."""
        log = make_log(tmp_path)
        log.store_decision("NVDA", "2026-01-10", DECISION_BUY)
        log.store_decision("AAPL", "2026-01-11", "Rating: Hold\nHold AAPL.")
        log.store_decision("MSFT", "2026-01-12", DECISION_SELL)
        log.update_with_outcome("AAPL", "2026-01-11", 0.01, -0.01, 5, "Neutral result.")
        entries = log.load_entries()
        assert len(entries) == 3
        nvda, aapl, msft = entries
        assert nvda["ticker"] == "NVDA" and nvda["pending"] is True
        assert aapl["ticker"] == "AAPL" and aapl["pending"] is False
        assert aapl["reflection"] == "Neutral result."
        assert msft["ticker"] == "MSFT" and msft["pending"] is True

    def test_update_atomic_write(self, tmp_path):
        """A pre-existing .tmp file is overwritten; the log is correctly updated."""
        log = make_log(tmp_path)
        log.store_decision("NVDA", "2026-01-10", DECISION_BUY)
        stale_tmp = tmp_path / "trading_memory.tmp"
        stale_tmp.write_text("GARBAGE CONTENT — should be overwritten", encoding="utf-8")
        log.update_with_outcome("NVDA", "2026-01-10", 0.042, 0.021, 5, "Correct.")
        assert not stale_tmp.exists()
        entries = log.load_entries()
        assert len(entries) == 1
        assert entries[0]["reflection"] == "Correct."
        assert entries[0]["pending"] is False

    def test_update_noop_when_no_log_path(self):
        log = TradingMemoryLog(config=None)
        log.update_with_outcome("NVDA", "2026-01-10", 0.05, 0.02, 5, "Reflection")

    def test_formatting_roundtrip_after_update(self, tmp_path):
        """All fields intact and blank line between tag and DECISION preserved after update."""
        log = make_log(tmp_path)
        log.store_decision("NVDA", "2026-01-10", DECISION_BUY)
        log.update_with_outcome("NVDA", "2026-01-10", 0.042, 0.021, 5, "Momentum confirmed.")
        entries = log.load_entries()
        assert len(entries) == 1
        e = entries[0]
        assert e["pending"] is False
        assert e["decision"] == DECISION_BUY.strip()
        assert e["reflection"] == "Momentum confirmed."
        assert e["raw"] == "+4.2%"
        assert e["alpha"] == "+2.1%"
        assert e["holding"] == "5d"
        raw_text = (tmp_path / "trading_memory.md").read_text(encoding="utf-8")
        assert "[2026-01-10 | NVDA | Buy | +4.2% | +2.1% | 5d]\n\nDECISION:" in raw_text



    # settlement.fetch_returns

    def test_fetch_returns_valid_ticker(self):
        stock_prices = [100.0, 102.0, 104.0, 103.0, 105.0, 106.0, 107.0]
        spy_prices   = [400.0, 402.0, 404.0, 403.0, 405.0, 406.0, 407.0]
        with patch("yfinance.Ticker") as mock_ticker_cls:
            def _make_ticker(sym):
                m = MagicMock()
                m.history.return_value = _price_df(spy_prices if sym == "SPY" else stock_prices)
                return m
            mock_ticker_cls.side_effect = _make_ticker
            raw, alpha, days, resolved = settlement.fetch_returns("NVDA", "2026-01-05")
        assert raw is not None and alpha is not None and days is not None
        assert isinstance(raw, float) and isinstance(alpha, float) and isinstance(days, int)
        assert days == 5
        # resolution date = the bar `days` sessions after entry, the first close after the decision (#1251)
        assert resolved == "2026-01-11"

    def test_fetch_returns_too_recent(self):
        """Only 1 data point available → returns all-None, no crash."""
        with patch("yfinance.Ticker") as mock_ticker_cls:
            m = MagicMock()
            m.history.return_value = _price_df([100.0])
            mock_ticker_cls.return_value = m
            raw, alpha, days, resolved = settlement.fetch_returns("NVDA", "2026-04-19")
        assert (raw, alpha, days, resolved) == (None, None, None, None)

    def test_fetch_returns_delisted(self):
        """Empty DataFrame → returns all-None, no crash."""
        with patch("yfinance.Ticker") as mock_ticker_cls:
            m = MagicMock()
            m.history.return_value = pd.DataFrame({"Close": []})
            mock_ticker_cls.return_value = m
            raw, alpha, days, resolved = settlement.fetch_returns("XXXXXFAKE", "2026-01-10")
        assert (raw, alpha, days, resolved) == (None, None, None, None)

    def test_fetch_returns_spy_shorter_than_stock(self):
        """SPY having fewer rows than the stock (but still a full window) must
        not raise IndexError."""
        stock_prices = [100.0, 102.0, 104.0, 103.0, 105.0, 106.0, 107.0, 108.0]  # 8 rows
        spy_prices   = [400.0, 402.0, 403.0, 405.0, 406.0, 407.0, 408.0]         # 7 rows
        with patch("yfinance.Ticker") as mock_ticker_cls:
            def _make_ticker(sym):
                m = MagicMock()
                m.history.return_value = _price_df(spy_prices if sym == "SPY" else stock_prices)
                return m
            mock_ticker_cls.side_effect = _make_ticker
            raw, alpha, days, resolved = settlement.fetch_returns("NVDA", "2026-01-05")
        assert raw is not None and alpha is not None
        assert days == 5  # full holding window used for both series
        assert resolved == "2026-01-11"

    def test_a_decision_is_scored_from_the_first_close_after_its_date(self):
        """#1518: news published after the close on the decision date reaches the
        analysts, so the close of that date is not a price they could have traded at."""
        stock_prices = [100.0, 150.0, 150.0, 150.0, 150.0, 150.0, 165.0]   # jumps on the next close
        spy_prices   = [400.0] * 7
        with patch("yfinance.Ticker") as mock_ticker_cls:
            def _make_ticker(sym):
                m = MagicMock()
                m.history.return_value = _price_df(spy_prices if sym == "SPY" else stock_prices)
                return m
            mock_ticker_cls.side_effect = _make_ticker
            raw, alpha, days, resolved = settlement.fetch_returns("NVDA", "2026-01-05")
        assert raw == pytest.approx(0.10) and alpha == pytest.approx(0.10)   # 150 -> 165, not 100 -> 150
        assert resolved == "2026-01-11"

    def test_fetch_returns_incomplete_window_stays_pending(self):
        """#1169: a rerun before the full holding window has traded returns
        unavailable (all-None) so the entry stays pending, rather than settling
        on a premature partial return."""
        stock_prices = [100.0, 102.0, 104.0]  # only 3 rows; holding window is 5
        spy_prices   = [400.0, 402.0, 404.0]
        with patch("yfinance.Ticker") as mock_ticker_cls:
            def _make_ticker(sym):
                m = MagicMock()
                m.history.return_value = _price_df(spy_prices if sym == "SPY" else stock_prices)
                return m
            mock_ticker_cls.side_effect = _make_ticker
            result = settlement.fetch_returns("NVDA", "2026-01-05")
        assert result == (None, None, None, None)

    # settlement.resolve_benchmark picks the index for the alpha calculation

    def test_resolve_benchmark_explicit_override(self):
        """config['benchmark_ticker'] wins for every ticker."""
        config = {
            "benchmark_ticker": "QQQ",
            "benchmark_map": {"": "SPY", ".T": "^N225"},
        }
        assert settlement.resolve_benchmark("7203.T", config) == "QQQ"
        assert settlement.resolve_benchmark("NVDA", config) == "QQQ"

    def test_resolve_benchmark_suffix_map(self):
        """Known suffixes route to their regional index."""
        config = {
            "benchmark_ticker": None,
            "benchmark_map": {
                ".T": "^N225", ".HK": "^HSI", ".NS": "^NSEI",
                ".L": "^FTSE", ".TO": "^GSPTSE", ".AX": "^AXJO",
                ".BO": "^BSESN", "": "SPY",
            },
        }
        assert settlement.resolve_benchmark("7203.T", config) == "^N225"
        assert settlement.resolve_benchmark("0700.HK", config) == "^HSI"
        assert settlement.resolve_benchmark("RELIANCE.NS", config) == "^NSEI"
        assert settlement.resolve_benchmark("AZN.L", config) == "^FTSE"

    def test_explicit_benchmark_is_resolved_like_any_other_symbol(self):
        """A configured benchmark takes the same alias mapping as the ticker, or
        the return lookup finds nothing and the decision never settles."""
        config = {"benchmark_ticker": "SPX500", "benchmark_map": {"": "SPY"}}
        assert settlement.resolve_benchmark("NVDA", config) == "^GSPC"

    def test_resolve_benchmark_china_a_shares(self):
        """A-share tickers route to their exchange composite (uses the real
        default benchmark_map, since A-share support relies on it)."""
        from ta_skills.config import DEFAULT_CONFIG
        config = {"benchmark_ticker": None,
                             "benchmark_map": DEFAULT_CONFIG["benchmark_map"]}
        assert settlement.resolve_benchmark("600519.SS", config) == "000001.SS"
        assert settlement.resolve_benchmark("000001.SZ", config) == "399001.SZ"
        # .SH is the exchange's own suffix; Yahoo spells Shanghai .SS (#1260)
        assert settlement.resolve_benchmark("600519.SH", config) == "000001.SS"

    def test_resolve_benchmark_brazil(self):
        """B3 tickers were measured against SPY."""
        from ta_skills.config import DEFAULT_CONFIG
        config = {"benchmark_ticker": None,
                             "benchmark_map": DEFAULT_CONFIG["benchmark_map"]}
        assert settlement.resolve_benchmark("PETR4.SA", config) == "^BVSP"

    @pytest.mark.parametrize(("ticker", "index"), [
        ("2330.TW", "^TWII"), ("6488.TWO", "^TWII"), ("005930.KS", "^KS11"), ("247540.KQ", "^KQ11"), ("D05.SI", "^STI"), ("SAP.DE", "^GDAXI"),
        ("MC.PA", "^FCHI"), ("ASML.AS", "^AEX"), ("NESN.SW", "^SSMI"), ("ENI.MI", "FTSEMIB.MI"),
    ])
    def test_resolve_benchmark_regional_indexes(self, ticker, index):
        """Taiwan, Korea, Singapore and the main European exchanges were measured against SPY."""
        from ta_skills.config import DEFAULT_CONFIG
        config = {"benchmark_ticker": None, "benchmark_map": DEFAULT_CONFIG["benchmark_map"]}
        assert settlement.resolve_benchmark(ticker, config) == index

    def test_resolve_benchmark_us_ticker_defaults_to_spy(self):
        """US tickers (no dotted suffix) take the empty-suffix entry."""
        config = {
            "benchmark_ticker": None,
            "benchmark_map": {"": "SPY", ".T": "^N225"},
        }
        assert settlement.resolve_benchmark("NVDA", config) == "SPY"
        assert settlement.resolve_benchmark("AAPL", config) == "SPY"

    def test_resolve_benchmark_unknown_suffix_falls_back(self):
        """Unrecognised suffix (BRK.B, FAKE.XX) falls back to SPY."""
        config = {
            "benchmark_ticker": None,
            "benchmark_map": {"": "SPY", ".T": "^N225"},
        }
        assert settlement.resolve_benchmark("FAKE.XX", config) == "SPY"
        assert settlement.resolve_benchmark("BRK.B", config) == "SPY"

    def test_resolve_benchmark_case_insensitive(self):
        """Suffix matching is case-insensitive so 7203.t resolves like 7203.T."""
        config = {
            "benchmark_ticker": None,
            "benchmark_map": {".T": "^N225", "": "SPY"},
        }
        assert settlement.resolve_benchmark("7203.t", config) == "^N225"



    # settlement.settle_pending





# ---------------------------------------------------------------------------
# Past context: what the Portfolio Manager is shown from the log
# ---------------------------------------------------------------------------

class TestPastContext:

    # PM prompt





    # get_past_context ordering and limits

    def test_same_ticker_prioritised(self, tmp_path):
        """Same-ticker entries in same-ticker section; cross-ticker entries in cross-ticker section."""
        log = make_log(tmp_path)
        _resolve_entry(log, "NVDA", "2026-01-05", DECISION_BUY, "Momentum confirmed.")
        _resolve_entry(log, "AAPL", "2026-01-06", DECISION_SELL, "Overvalued.")
        result = log.get_past_context("NVDA")
        assert "Past analyses of NVDA" in result
        assert "Recent cross-ticker lessons" in result
        same_block, cross_block = result.split("Recent cross-ticker lessons")
        assert "NVDA" in same_block
        assert "AAPL" in cross_block

    def test_cross_ticker_reflection_only(self, tmp_path):
        """Cross-ticker entries show only the REFLECTION text, not the full DECISION."""
        log = make_log(tmp_path)
        _resolve_entry(log, "AAPL", "2026-01-06", DECISION_SELL, "Overvalued correction.")
        result = log.get_past_context("NVDA")
        assert "Overvalued correction." in result
        assert "Exit position immediately." not in result

    def test_n_same_limit_respected(self, tmp_path):
        """More than 5 same-ticker completed entries → only 5 injected."""
        log = make_log(tmp_path)
        for i in range(7):
            _resolve_entry(log, "NVDA", f"2026-01-{i+1:02d}", DECISION_BUY, f"Lesson {i}.")
        result = log.get_past_context("NVDA", n_same=5)
        lessons_present = sum(1 for i in range(7) if f"Lesson {i}." in result)
        assert lessons_present == 5

    def test_n_cross_limit_respected(self, tmp_path):
        """More than 3 cross-ticker completed entries → only 3 injected."""
        log = make_log(tmp_path)
        tickers = ["AAPL", "MSFT", "TSLA", "AMZN", "GOOG"]
        for i, ticker in enumerate(tickers):
            _resolve_entry(log, ticker, f"2026-01-{i+1:02d}", DECISION_BUY, f"{ticker} lesson.")
        result = log.get_past_context("NVDA", n_cross=3)
        cross_count = sum(result.count(f"{t} lesson.") for t in tickers)
        assert cross_count == 3

    # Full A→B→C integration cycle

    def test_full_cycle_store_resolve_inject(self, tmp_path):
        """store pending → resolve with outcome → past_context non-empty for PM."""
        log = make_log(tmp_path)
        log.store_decision("NVDA", "2026-01-05", DECISION_BUY)
        assert len(log.get_pending_entries()) == 1
        assert log.get_past_context("NVDA") == ""
        log.update_with_outcome("NVDA", "2026-01-05", 0.05, 0.02, 5, "Correct call.")
        assert log.get_pending_entries() == []
        past_ctx = log.get_past_context("NVDA")
        assert past_ctx != ""
        assert "NVDA" in past_ctx
        assert "Correct call." in past_ctx
        assert "DECISION:" in past_ctx
        assert "REFLECTION:" in past_ctx


# ---------------------------------------------------------------------------
# Legacy removal: BM25 / FinancialSituationMemory fully gone
# ---------------------------------------------------------------------------









@pytest.mark.unit
def test_a_longer_window_asks_for_enough_price_history(monkeypatch):
    """Trading days are not calendar days: a 21-day window needs about a month
    of bars, and asking for 28 days left every outcome unsettled."""
    asked = {}

    class _Ticker:
        def __init__(self, symbol):
            self.symbol = symbol

        def history(self, start, end):
            asked["start"], asked["end"] = start, end
            import pandas as pd
            days = pd.bdate_range(start, end)
            return pd.DataFrame({"Close": range(len(days))}, index=days)

    monkeypatch.setattr("ta_skills.data.vendors.yahoo.market.yf.Ticker", _Ticker)

    raw, alpha, days, resolved = settlement.fetch_returns("NVDA", "2026-06-01", 21, benchmark="SPY")

    assert days == 21 and resolved is not None, (raw, alpha, days, resolved)


@pytest.mark.unit
def test_concurrent_writers_keep_every_decision_and_outcome(tmp_path):
    """Graphs sharing one log record and settle at the same time; none may lose another's write."""
    import threading

    path = tmp_path / "memory.md"
    tickers = [f"T{i}" for i in range(8)]
    dates = [f"2026-01-{day:02d}" for day in range(1, 13)]
    errors = []

    def work(ticker):
        log = TradingMemoryLog({"memory_log_path": str(path)})
        try:
            for date in dates:
                log.store_decision(ticker, date, "**Rating**: Buy")
                log.update_with_outcome(ticker, date, 0.01, 0.005, 5, "held up")
        except Exception as exc:  # noqa: BLE001 — collected for the assertion
            errors.append(exc)

    threads = [threading.Thread(target=work, args=(t,)) for t in tickers]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()

    entries = TradingMemoryLog({"memory_log_path": str(path)}).load_entries()
    assert errors == []
    assert len(entries) == len(tickers) * len(dates)
    assert all(not e["pending"] for e in entries)


def _closes(by_symbol):
    """A get_closes stand-in serving each symbol's series between the dates asked."""
    def get_closes(symbol, start_date, end_date):
        series = by_symbol[symbol]
        return series[(series.index >= start_date) & (series.index < end_date)]
    return get_closes


@pytest.mark.unit
class TestSettlementWindow:
    """Stock and benchmark are measured over the same dates, whatever their calendars."""

    def test_a_seven_day_market_is_scored_against_the_benchmark_over_the_same_dates(self, monkeypatch):
        coin = pd.Series([100.0 + i for i in range(10)], index=pd.date_range("2026-01-03", periods=10, freq="D"))
        weekdays = pd.bdate_range("2025-12-29", "2026-01-16")
        spy = pd.Series([400.0 + 2 * i for i in range(len(weekdays))], index=weekdays)
        monkeypatch.setattr(settlement, "get_closes", _closes({"BTC-USD": coin, "SPY": spy}))

        raw, alpha, days, resolved = settlement.fetch_returns("BTC-USD", "2026-01-03", 5, "SPY")

        # Decided Sat 3, the coin is held Sun 4 -> Fri 9; SPY over the same span is Fri 2 -> Fri 9.
        assert resolved == "2026-01-09"
        assert raw == pytest.approx(106.0 / 101.0 - 1)
        assert alpha == pytest.approx(raw - (spy["2026-01-09"] / spy["2026-01-02"] - 1))

    def test_bars_stamped_in_different_time_zones_meet_on_their_dates(self, monkeypatch):
        # Yahoo stamps a coin's daily bar at midnight UTC and SPY's at midnight
        # New York, five hours later: the same day, not the day after.
        coin = pd.Series([100.0 + i for i in range(10)],
                         index=pd.date_range("2026-01-03", periods=10, freq="D", tz="UTC"))
        weekdays = pd.bdate_range("2025-12-29", "2026-01-16", tz="America/New_York")
        spy = pd.Series([400.0 + 2 * i for i in range(len(weekdays))], index=weekdays)
        monkeypatch.setattr(settlement, "get_closes", _closes({"BTC-USD": coin, "SPY": spy}))

        raw, alpha, _, resolved = settlement.fetch_returns("BTC-USD", "2026-01-03", 5, "SPY")

        assert resolved == "2026-01-09"
        spy_by_day = spy.tz_localize(None)
        assert alpha == pytest.approx(raw - (spy_by_day["2026-01-09"] / spy_by_day["2026-01-02"] - 1))

    def test_the_outcome_waits_for_the_benchmark_to_trade_through_the_window(self, monkeypatch):
        coin = pd.Series([100.0 + i for i in range(10)], index=pd.date_range("2026-01-03", periods=10, freq="D"))
        spy = pd.Series([400.0, 401.0, 402.0], index=pd.to_datetime(["2026-01-02", "2026-01-05", "2026-01-06"]))
        monkeypatch.setattr(settlement, "get_closes", _closes({"BTC-USD": coin, "SPY": spy}))

        assert settlement.fetch_returns("BTC-USD", "2026-01-03", 5, "SPY") == (None, None, None, None)

    def test_a_missing_close_is_skipped_not_scored(self, monkeypatch):
        idx = pd.bdate_range("2026-01-05", periods=8)
        stock = pd.Series([100.0, float("nan"), 101.0, 102.0, 103.0, 104.0, 105.0, 106.0], index=idx)
        spy = pd.Series([400.0 + i for i in range(8)], index=idx)
        monkeypatch.setattr(settlement, "get_closes", _closes({"NVDA": stock, "SPY": spy}))

        raw, alpha, _, resolved = settlement.fetch_returns("NVDA", "2026-01-05", 5, "SPY")

        # Entered at the first real close after the decision: Wed 7, past the missing Tue 6.
        assert raw == pytest.approx(106.0 / 101.0 - 1)
        assert resolved == "2026-01-14"
        assert alpha == pytest.approx(raw - (spy["2026-01-14"] / spy["2026-01-07"] - 1))


@pytest.mark.unit
def test_a_close_that_is_not_a_price_is_skipped_not_scored(monkeypatch):
    """A zero close would score as an infinite return and be written to the memory log."""
    idx = pd.bdate_range("2026-01-05", periods=8)
    stock = pd.Series([0.0, 100.0, 101.0, 102.0, 103.0, 104.0, 105.0, 106.0], index=idx)
    spy = pd.Series([400.0 + i for i in range(8)], index=idx)
    monkeypatch.setattr(settlement, "get_closes", _closes({"NVDA": stock, "SPY": spy}))

    raw, alpha, _, _ = settlement.fetch_returns("NVDA", "2026-01-02", 5, "SPY")

    assert raw == pytest.approx(105.0 / 100.0 - 1)   # the window opens on the first real close












@pytest.mark.unit
class TestDueOutcomes:
    """``ta memory due`` scores every ticker's decisions whose window has traded (#1445);
    the reflection itself is written later by the memory role."""

    def _setup(self, tmp_path, monkeypatch, tickers=("AAPL", "MSFT"), dates=("2026-01-05",)):
        log = make_log(tmp_path)
        for ticker in tickers:
            for date in dates:
                log.store_decision(ticker, date, DECISION_BUY)
        idx = pd.bdate_range("2026-01-02", periods=40)
        series = pd.Series([100.0 + i for i in range(40)], index=idx)
        monkeypatch.setattr(
            settlement, "get_closes", _closes(dict.fromkeys((*tickers, "SPY"), series))
        )
        return log

    def test_every_ticker_with_a_due_decision_is_scored(self, tmp_path, monkeypatch):
        log = self._setup(tmp_path, monkeypatch)

        result = settlement.due_outcomes(log, {"holding_period_days": 5})

        assert [(d.ticker, d.date) for d in result.due] == [("AAPL", "2026-01-05"), ("MSFT", "2026-01-05")]
        assert result.failed == []
        due = result.due[0]
        assert due.holding_days == 5 and due.benchmark == "SPY"
        assert due.decision == DECISION_BUY and due.resolution_date
        # Scoring never writes the log: the entry stays pending until the reflection is resolved.
        assert len(log.get_pending_entries()) == 2

    def test_a_decision_whose_window_has_not_traded_is_left_out(self, tmp_path, monkeypatch):
        log = self._setup(tmp_path, monkeypatch, tickers=("AAPL",), dates=("2026-01-05", "2026-03-20"))

        result = settlement.due_outcomes(log, {"holding_period_days": 5})

        assert [d.date for d in result.due] == ["2026-01-05"]
        assert result.failed == []

    def test_a_price_request_that_fails_is_reported_not_taken_for_not_due(self, tmp_path, monkeypatch):
        from ta_skills.data.errors import VendorUnavailableError

        log = self._setup(tmp_path, monkeypatch, tickers=("AAPL",))

        def down(*args, **kwargs):
            raise VendorUnavailableError("Yahoo Finance request failed: TimeoutError")

        monkeypatch.setattr(settlement, "get_closes", down)

        result = settlement.due_outcomes(log, {"holding_period_days": 5})

        assert result.due == [] and "prices unavailable" in result.failed[0][2]
        assert len(log.get_pending_entries()) == 1

    def test_any_failure_to_price_a_decision_is_reported(self, tmp_path, monkeypatch):
        log = self._setup(tmp_path, monkeypatch, tickers=("AAPL",))

        def broken(*args, **kwargs):
            raise RuntimeError("vendor outage")

        monkeypatch.setattr(settlement, "get_closes", broken)

        result = settlement.due_outcomes(log, {"holding_period_days": 5})

        assert result.due == [] and "vendor outage" in result.failed[0][2]

    def test_resolving_a_scored_decision_settles_it_once(self, tmp_path, monkeypatch):
        log = self._setup(tmp_path, monkeypatch, tickers=("AAPL",))
        due = settlement.due_outcomes(log, {"holding_period_days": 5}).due[0]

        args = ("AAPL", due.date, due.raw, due.alpha, due.holding_days, "Held up.")
        assert log.update_with_outcome(*args, resolution_date=due.resolution_date) is True
        assert log.update_with_outcome(*args, resolution_date=due.resolution_date) is False

        entry = log.load_entries()[0]
        assert not entry["pending"] and entry["reflection"] == "Held up."
        assert entry["resolved"] == due.resolution_date
        assert settlement.due_outcomes(log, {"holding_period_days": 5}).due == []
