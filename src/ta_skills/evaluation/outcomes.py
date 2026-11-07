"""Price each forecast over what followed it, using only bars on or before an as-of date.

Conventions, shared with memory settlement (``memory/settlement.py``):

- Entry is the first close after the analysis date: a run dated D reads news
  published after D's close.
- A horizon of h trading days is scored only once the instrument has traded h
  sessions past entry and the benchmark has traded through the exit; until then
  it stays pending, never settled on a partial window.
- Returns use dividend- and split-adjusted closes (total return); alpha is the
  return minus the benchmark's over the same dates. An instrument that is its own
  benchmark has no alpha.
- Price targets are nominal prices, so they are compared with closes adjusted for
  splits only, and a split after the report is applied to the target as well.
"""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from datetime import timedelta

import pandas as pd

from ta_skills.data.symbols import normalize_symbol
from ta_skills.data.vendors.yahoo.market import PRICE_HISTORY_COLUMNS, get_price_history
from ta_skills.evaluation.ledger import RATING_SCORE
from ta_skills.memory.settlement import resolve_benchmark

HORIZONS = (5, 21, 63, 126, 252)
MOMENTUM = (21, 63)


def _empty() -> pd.DataFrame:
    return pd.DataFrame(columns=PRICE_HISTORY_COLUMNS, index=pd.DatetimeIndex([]), dtype=float)


def load_histories(symbols: set[str], start: str, end: str, workers: int = 8) -> tuple[dict, dict]:
    """Histories by symbol, and the symbols whose prices could not be had with the reason."""
    histories, failed = {}, {}

    def one(symbol):
        try:
            return symbol, get_price_history(symbol, start, end), None
        except Exception as exc:  # one unreachable symbol must not end the evaluation
            return symbol, None, str(exc)

    with ThreadPoolExecutor(workers) as pool:
        for symbol, history, error in pool.map(one, sorted(symbols)):
            if error is not None:
                failed[symbol] = error
            elif history.empty:
                failed[symbol] = "no prices"
            else:
                histories[symbol] = history
    return histories, failed


def _trailing(series: pd.Series, n: int) -> float | None:
    return float(series.iloc[-1] / series.iloc[-1 - n] - 1) if len(series) > n else None


def score_forecast(
    forecast: dict,
    stock: pd.DataFrame,
    bench: pd.DataFrame | None,
    as_of: pd.Timestamp,
    horizons=HORIZONS,
) -> dict:
    """One forecast's outcome row. ``bench`` None: the instrument is its own benchmark."""
    date = pd.Timestamp(forecast["date"])
    row = {
        k: forecast[k]
        for k in (
            "path",
            "ticker",
            "date",
            "model",
            "generated_at",
            "rating",
            "confidence",
            "time_horizon",
            "horizon_min_td",
            "horizon_max_td",
        )
    }
    row["score"] = RATING_SCORE.get(forecast["rating"])
    # Yahoo rescales all history for every split up to today, as-of or not: a target
    # written before a split is rescaled by the same factor.
    split = float(stock.loc[stock.index > date, "splits"].replace(0, 1).prod())
    stock = stock[stock.index <= as_of]
    before, after = stock[stock.index <= date], stock[stock.index > date]
    if before.empty or after.empty:
        row["status"] = "no entry yet" if not before.empty else "no prices before the analysis date"
        return row
    row["status"] = "ok"
    entry = after.index[0]
    row["entry_date"] = entry.strftime("%Y-%m-%d")
    # Generated after the entry session: it could have seen the entry close and beyond.
    row["retro"] = pd.Timestamp(forecast["generated_at"]).normalize() > entry
    row["p0"] = float(before["close"].iloc[-1])
    for key in ("reported_price", "price_target"):
        row[key] = forecast[key] / split if forecast[key] else None

    bench = None if bench is None else bench[bench.index <= as_of]
    for n in MOMENTUM:
        own = _trailing(before["adj_close"], n)
        base = (
            None
            if bench is None or own is None
            else _trailing(bench.loc[bench.index <= date, "adj_close"], n)
        )
        row[f"mom{n}"] = own - base if base is not None else None

    adj = after["adj_close"]
    for h in horizons:
        if len(after) <= h:
            continue
        exit_ = after.index[h]
        row[f"raw{h}"] = float(adj.iloc[h] / adj.iloc[0] - 1)
        if (
            bench is not None
            and not bench.empty
            and bench.index[0] <= entry
            and exit_ <= bench.index[-1]
        ):
            b = bench["adj_close"]
            row[f"alpha{h}"] = row[f"raw{h}"] - float(b.asof(exit_) / b.asof(entry) - 1)

    target = row["price_target"]
    for end in ("min", "max"):
        td = forecast[f"horizon_{end}_td"]
        if not target or not td or len(after) <= td:
            continue
        window = after.iloc[: td + 1]
        row[f"pt_close_{end}"] = float(window["close"].iloc[-1])
        row[f"pt_exit_{end}"] = window.index[-1].strftime("%Y-%m-%d")
        if target > row["p0"]:
            row[f"pt_touch_{end}"] = bool(window["high"].max() >= target)
        elif target < row["p0"]:
            row[f"pt_touch_{end}"] = bool(window["low"].min() <= target)
    return row


def score_ledger(forecasts: list[dict], config: dict, as_of: str) -> tuple[pd.DataFrame, dict]:
    """Outcome rows for every rated forecast, and the symbols whose prices failed."""
    rated = [f for f in forecasts if f["rating"]]
    if not rated:
        return pd.DataFrame(), {}
    benchmarks = {f["ticker"]: resolve_benchmark(f["ticker"], config) for f in rated}
    # Enough history before the first decision for 63-session momentum.
    start = (pd.Timestamp(min(f["date"] for f in rated)) - timedelta(days=130)).strftime("%Y-%m-%d")
    # Through today, not the as-of date: Yahoo rescales every close for each split up to
    # today, so a target needs the split rows after as-of too. score_forecast drops the
    # bars after as-of itself.
    end = (pd.Timestamp.today().normalize() + timedelta(days=1)).strftime("%Y-%m-%d")
    histories, failed = load_histories(set(benchmarks) | set(benchmarks.values()), start, end)
    cutoff = pd.Timestamp(as_of)
    rows = []
    for f in rated:
        stock, bench_symbol = histories.get(f["ticker"]), benchmarks[f["ticker"]]
        if stock is None:
            rows.append(
                {
                    "path": f["path"],
                    "ticker": f["ticker"],
                    "date": f["date"],
                    "model": f["model"],
                    "rating": f["rating"],
                    "status": f"prices unavailable: {failed.get(f['ticker'])}",
                }
            )
            continue
        own = normalize_symbol(f["ticker"]) == normalize_symbol(bench_symbol)
        bench = None if own else histories.get(bench_symbol, _empty())
        row = score_forecast(f, stock, bench, cutoff)
        row["benchmark"] = None if own else bench_symbol
        rows.append(row)
    return pd.DataFrame(rows), failed
