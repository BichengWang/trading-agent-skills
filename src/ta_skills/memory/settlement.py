"""Settling past decisions: once a decision's holding window has traded, score
it against its benchmark. The reflection on the outcome is written by the
memory role (``ta memory due`` / ``ta memory resolve``), not by this module."""

import logging
from dataclasses import dataclass, field
from datetime import datetime, timedelta

from ta_skills.data.symbols import normalize_symbol
from ta_skills.data.vendors.yahoo.market import get_closes

logger = logging.getLogger(__name__)


def resolve_benchmark(ticker: str, config: dict) -> str:
    """Pick the benchmark ticker for alpha calculation against ``ticker``.

    ``config["benchmark_ticker"]`` overrides everything when set; otherwise
    the suffix map matches the ticker's exchange suffix (e.g. ``.T`` for
    Tokyo). US-listed tickers without a dotted suffix fall through to the
    empty-suffix entry (SPY by default). Unrecognised suffixes, including
    US tickers with dots like ``BRK.B``, also take the empty-suffix entry.
    Returns are compared as percentages, each in its own currency.
    """

    explicit = config.get("benchmark_ticker")
    if explicit:
        # Same alias mapping as the analyzed ticker; an unmapped alias finds
        # no prices, and the decision would stay pending for good.
        return normalize_symbol(explicit)
    benchmark_map = config.get("benchmark_map", {})
    ticker_upper = normalize_symbol(ticker)
    for suffix, benchmark in benchmark_map.items():
        if suffix and ticker_upper.endswith(suffix.upper()):
            return benchmark
    return benchmark_map.get("", "SPY")


def _by_day(closes):
    """Closes keyed by the calendar day of their bar, missing closes dropped.

    Yahoo stamps a daily bar at midnight in its market's time zone (UTC for a
    coin, New York for SPY), so two series meet on their dates, not their
    instants. A missing or non-positive close is no price at all, not a bar to
    score on.
    """
    closes = closes[closes > 0]
    if getattr(closes.index, "tz", None) is not None:
        closes = closes.tz_localize(None)
    return closes


def fetch_returns(
    ticker: str, trade_date: str, holding_days: int = 5,
    benchmark: str = "SPY",
) -> tuple[float | None, float | None, int | None, str | None]:
    """Fetch raw and alpha return for ticker over holding_days after trade_date.

    The position is entered at the first close after ``trade_date``: a run dated
    ``trade_date`` reads news published after that day's close (#1518).

    ``benchmark`` is the index used as the alpha baseline (resolved by the
    caller via ``resolve_benchmark``). Returns ``(raw_return, alpha_return,
    holding_days, resolution_date)`` — where ``resolution_date`` is the date
    of the last price bar used, i.e. when the outcome became known (#1251) —
    or ``(None, None, None, None)`` when the full holding window has not traded
    yet (#1169). A failure to get or read the prices is raised, for the caller
    to report.
    """
    start = datetime.strptime(trade_date, "%Y-%m-%d")
    # holding_days counts trading days after the entry session, so ask for the
    # calendar span they occupy (about 7 for every 5) plus a week for holidays.
    end = start + timedelta(days=round((holding_days + 1) * 7 / 5) + 7)
    end_str = end.strftime("%Y-%m-%d")

    # Closes for the instrument the analysis priced (XAUUSD -> GC=F, #984),
    # from the first session after the decision date.
    stock = _by_day(get_closes(ticker, trade_date, end_str))
    stock = stock[stock.index > start]
    # From a week earlier, so the benchmark has a close on or before entry.
    bench_start = (start - timedelta(days=7)).strftime("%Y-%m-%d")
    bench = _by_day(get_closes(benchmark, bench_start, end_str))

    # Require the full holding window to have traded. A rerun before it has
    # leaves the entry pending to retry next run, rather than settling on a
    # premature partial return (#1169).
    if len(stock) <= holding_days:
        return None, None, None, None
    entry, exit_ = stock.index[0], stock.index[holding_days]
    # The window is the stock's own sessions; the benchmark is valued at its
    # last close by each end, so both returns span the same dates even when
    # the calendars differ (a coin trades at weekends, an index does not).
    # It must have traded through the exit, or its close there may yet move.
    if bench.empty or bench.index[0] > entry or bench.index[-1] < exit_:
        return None, None, None, None

    raw = float((stock.iloc[holding_days] - stock.iloc[0]) / stock.iloc[0])
    bench_entry, bench_exit = bench.asof(entry), bench.asof(exit_)
    bench_ret = float((bench_exit - bench_entry) / bench_entry)
    alpha = raw - bench_ret
    # Every close used is known by the exit: the point-in-time cutoff for
    # injecting the lesson (#1251).
    resolution_date = exit_.strftime("%Y-%m-%d")
    return raw, alpha, holding_days, resolution_date


@dataclass
class DueOutcome:
    """A pending decision whose holding window has traded, scored and ready to reflect on."""

    ticker: str
    date: str
    raw: float
    alpha: float
    holding_days: int
    benchmark: str
    resolution_date: str
    decision: str


@dataclass
class Settlement:
    """What a settlement pass found: decisions ready for reflection and those it could not score."""

    due: list[DueOutcome] = field(default_factory=list)
    failed: list[tuple[str, str, str]] = field(default_factory=list)    # (ticker, date, reason)


def due_outcomes(memory_log, config: dict) -> Settlement:
    """Score every ticker's pending decisions whose holding window has traded (#1445).

    Tickers are visited in first-seen order of the pending entries. Entries whose
    window has not traded yet are skipped (still pending, retried next run); an
    entry whose prices could not be had is reported as failed and stays pending.
    """
    result = Settlement()
    pending = memory_log.get_pending_entries()
    for ticker in dict.fromkeys(e["ticker"] for e in pending):
        benchmark = resolve_benchmark(ticker, config)
        for entry in (e for e in pending if e["ticker"] == ticker):
            try:
                raw, alpha, days, resolution_date = fetch_returns(
                    ticker, entry["date"], config.get("holding_period_days", 5), benchmark=benchmark,
                )
            except Exception as exc:
                logger.warning("Prices for settling %s on %s were unavailable: %s", ticker, entry["date"], exc)
                result.failed.append((ticker, entry["date"], f"prices unavailable: {exc}"))
                continue
            if raw is None:
                continue  # the holding window has not traded yet: try again next run
            result.due.append(DueOutcome(
                ticker=ticker, date=entry["date"], raw=raw, alpha=alpha, holding_days=days,
                benchmark=benchmark, resolution_date=resolution_date,
                decision=entry.get("decision", ""),
            ))
    return result
