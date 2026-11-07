"""Technical indicators: stockstats columns plus the custom multi-timeframe maths.

``get_stock_stats_indicators_window`` is the Yahoo ``get_indicators`` vendor. It is
registry-driven: the column comes from ``indicator_registry.resolve_column`` (so
``config["indicator_params"]`` is honoured), values are rescaled by the spec's
``scale`` (MFI is reported 0-100; stockstats computes 0-1), and the description
shown is the registry's. TD-9, Z-Score, SuperTrend and OBV are computed here from
OHLCV; every other name is a stockstats column.
"""

import logging
from datetime import datetime
from typing import Annotated

import pandas as pd
from dateutil.relativedelta import relativedelta
from stockstats import wrap

from ta_skills.data.errors import NoMarketDataError, VendorError
from ta_skills.data.indicator_registry import effective_params, get_spec, resolve_column
from ta_skills.data.vendors.yahoo.ohlcv import _ensure_date_column, load_ohlcv

logger = logging.getLogger(__name__)


def compute_td_setup(close: "pd.Series") -> "pd.Series":
    """TD Sequential *Setup* running count, signed by direction.

    For each bar, compare Close to the Close 4 bars earlier:
      * Close < Close[-4]  -> buy setup  (count climbs +1, +2, ...)
      * Close > Close[-4]  -> sell setup (count falls  -1, -2, ...)
      * equal, a flip, or no 4-bar lookback resets the run.
    The magnitude is clamped at 9 (a completed setup); the Countdown phase and
    TDST levels are intentionally out of scope (see plan P1). The *last* value
    is the current running count — the live signal, not just a completed 9.
    """
    values = pd.to_numeric(pd.Series(close).reset_index(drop=True), errors="coerce")
    counts = [0] * len(values)
    run = 0
    for i in range(len(values)):
        if i < 4 or pd.isna(values.iloc[i]) or pd.isna(values.iloc[i - 4]):
            run = 0
        elif values.iloc[i] < values.iloc[i - 4]:
            run = run + 1 if run > 0 else 1
        elif values.iloc[i] > values.iloc[i - 4]:
            run = run - 1 if run < 0 else -1
        else:
            run = 0
        run = max(-9, min(9, run))
        counts[i] = run
    return pd.Series(counts, index=pd.Series(close).index)


def td_setup_by_timeframe(data: "pd.DataFrame", curr_date: str = None) -> dict:
    """Current TD Setup count on weekly / monthly / daily bars.

    Weekly and monthly are resampled (last close per period) from the same daily
    frame, so no extra fetch is needed. When ``curr_date`` is given, rows after
    it are dropped first to preserve the no-look-ahead guarantee. Returns signed
    ints keyed ``weekly``/``monthly``/``daily`` (the plan's tier 1/2/3).
    """
    df = _ensure_date_column(pd.DataFrame(data)).copy()
    df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
    df = df.dropna(subset=["Date"]).sort_values("Date")
    if curr_date:
        df = df[df["Date"] <= pd.Timestamp(curr_date)]

    close = pd.to_numeric(df["Close"], errors="coerce")
    close.index = pd.DatetimeIndex(df["Date"])
    close = close.dropna()

    def _last(series: "pd.Series") -> int:
        return int(compute_td_setup(series).iloc[-1]) if len(series) else 0

    return {
        "weekly": _last(close.resample("W").last().dropna()),
        "monthly": _last(close.resample("ME").last().dropna()),
        "daily": _last(close),
    }


def _td_phrase(n: int) -> str:
    if n == 0:
        return "(no active setup)"
    side = "buy" if n > 0 else "sell"
    if abs(n) == 9:
        return f"({side}-setup COMPLETE 9 of 9 — reversal watch)"
    return f"({side}-setup, {abs(n)} of 9)"


def format_td_setup_block(counts: dict) -> str:
    """Render the tiered TD-9 report block (weekly > monthly > daily)."""
    tiers = [
        ("Tier 1", "Weekly", counts.get("weekly", 0)),
        ("Tier 2", "Monthly", counts.get("monthly", 0)),
        ("Tier 3", "Daily", counts.get("daily", 0)),
    ]
    lines = [
        "## TD-9 (TD Sequential Setup) — current running count by timeframe (higher tier wins):"
    ]
    for tier, name, n in tiers:
        lines.append(f"- {tier} {name}: {n:+d}  {_td_phrase(n)}")
    return "\n".join(lines)


def compute_zscore(close: "pd.Series", window: int = 20) -> "pd.Series":
    """Rolling z-score of close: (close - rolling mean) / rolling std.

    Measures how many standard deviations the latest close sits from its
    ``window``-period mean — a mean-reversion / stretch gauge. Positive means
    extended above the mean (overbought), negative below (oversold); magnitude
    >= 2 flags a statistically stretched price. Bars without a full window or a
    zero/NaN std yield 0 (no signal). The *last* value is the current reading.
    """
    values = pd.to_numeric(pd.Series(close).reset_index(drop=True), errors="coerce")
    mean = values.rolling(window).mean()
    std = values.rolling(window).std()
    z = (values - mean) / std
    z = z.where(std > 0)  # guard divide-by-zero (a flat window)
    return pd.Series(z.fillna(0.0).to_numpy(), index=pd.Series(close).index)


def zscore_by_timeframe(data: "pd.DataFrame", curr_date: str = None, window: int = 20) -> dict:
    """Current close z-score on weekly / monthly / daily bars.

    Weekly and monthly are resampled (last close per period) from the same daily
    frame, so no extra fetch is needed. When ``curr_date`` is given, rows after
    it are dropped first to preserve the no-look-ahead guarantee. Returns floats
    keyed ``weekly``/``monthly``/``daily`` (tier 1/2/3), mirroring TD-9.
    """
    df = _ensure_date_column(pd.DataFrame(data)).copy()
    df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
    df = df.dropna(subset=["Date"]).sort_values("Date")
    if curr_date:
        df = df[df["Date"] <= pd.Timestamp(curr_date)]

    close = pd.to_numeric(df["Close"], errors="coerce")
    close.index = pd.DatetimeIndex(df["Date"])
    close = close.dropna()

    def _last(series: "pd.Series") -> float:
        return float(compute_zscore(series, window).iloc[-1]) if len(series) else 0.0

    return {
        "weekly": _last(close.resample("W").last().dropna()),
        "monthly": _last(close.resample("ME").last().dropna()),
        "daily": _last(close),
    }


def _zscore_phrase(z: float) -> str:
    if abs(z) < 1:
        return "(near the mean)"
    side = "above" if z > 0 else "below"
    if abs(z) >= 2:
        stretch = "overbought" if z > 0 else "oversold"
        return f"(stretched {side} mean — {stretch})"
    return f"({side} the mean)"


def format_zscore_block(zscores: dict, window: int = 20) -> str:
    """Render the tiered z-score report block (weekly > monthly > daily)."""
    tiers = [
        ("Tier 1", "Weekly", zscores.get("weekly", 0.0)),
        ("Tier 2", "Monthly", zscores.get("monthly", 0.0)),
        ("Tier 3", "Daily", zscores.get("daily", 0.0)),
    ]
    lines = [
        f"## Z-Score ({window}-period close z-score) — current reading by timeframe "
        "(higher tier wins):"
    ]
    for tier, name, z in tiers:
        lines.append(f"- {tier} {name}: {z:+.2f}  {_zscore_phrase(z)}")
    return "\n".join(lines)


def compute_obv(data: "pd.DataFrame") -> "pd.Series":
    """On-Balance Volume: cumulative volume signed by the close-to-close move.

    Volume is added on up days, subtracted on down days and ignored on
    unchanged days. The absolute level is meaningless — the signal is the
    slope and divergences against price over the lookback window. Returned as
    a date-indexed daily series (OBV has no tiered timeframe payload).
    """
    df = _ensure_date_column(pd.DataFrame(data)).copy()
    df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
    df = df.dropna(subset=["Date"]).sort_values("Date")

    close = pd.to_numeric(df["Close"], errors="coerce")
    volume = pd.to_numeric(df["Volume"], errors="coerce").fillna(0.0)
    move = close.diff()
    direction = move.mask(move > 0, 1.0).mask(move < 0, -1.0).fillna(0.0)
    obv = (volume * direction).cumsum()
    obv.index = pd.DatetimeIndex(df["Date"])
    return obv


def _resample_ohlcv(df: "pd.DataFrame", rule: str) -> "pd.DataFrame":
    """Resample a date-indexed daily OHLCV frame to weekly/monthly bars."""
    agg = df.resample(rule).agg(
        {"Open": "first", "High": "max", "Low": "min", "Close": "last", "Volume": "sum"}
    )
    return agg.dropna(subset=["Close"])


def supertrend_by_timeframe(data: "pd.DataFrame", curr_date: str = None, window: int = 14) -> dict:
    """Current SuperTrend state on weekly / monthly / daily bars.

    stockstats computes the SuperTrend line itself (``window``-period ATR,
    3x multiplier); the custom part is only the weekly/monthly OHLC resample —
    mirroring ``td_setup_by_timeframe``. When ``curr_date`` is given, rows
    after it are dropped first to preserve the no-look-ahead guarantee.
    Returns per tier (``weekly``/``monthly``/``daily``) either None
    (insufficient history) or a dict with:

    * ``direction`` — "up" (close at/above the line) or "down"
    * ``level`` — the SuperTrend line, i.e. the current trailing stop
    * ``distance_pct`` — signed % distance from the line to the close
    """
    df = _ensure_date_column(pd.DataFrame(data)).copy()
    df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
    df = df.dropna(subset=["Date"]).sort_values("Date")
    if curr_date:
        df = df[df["Date"] <= pd.Timestamp(curr_date)]

    column = f"supertrend_{window}"

    def _last(frame: "pd.DataFrame"):
        if len(frame) < 2:
            return None
        stock_df = wrap(frame.copy())
        level = stock_df[column].iloc[-1]
        close = pd.to_numeric(frame["Close"], errors="coerce").iloc[-1]
        if pd.isna(level) or level <= 0 or pd.isna(close):
            return None
        level = float(level)
        close = float(close)
        return {
            "direction": "up" if close >= level else "down",
            "level": level,
            "distance_pct": (close - level) / level * 100.0,
        }

    if df.empty:
        return {"weekly": None, "monthly": None, "daily": None}

    daily = df.set_index(pd.DatetimeIndex(df["Date"]))[["Open", "High", "Low", "Close", "Volume"]]
    return {
        "weekly": _last(_resample_ohlcv(daily, "W")),
        "monthly": _last(_resample_ohlcv(daily, "ME")),
        "daily": _last(daily),
    }


def format_supertrend_block(snapshots: dict, window: int = 14) -> str:
    """Render the tiered SuperTrend report block (weekly > monthly > daily)."""
    tiers = [
        ("Tier 1", "Weekly", snapshots.get("weekly")),
        ("Tier 2", "Monthly", snapshots.get("monthly")),
        ("Tier 3", "Daily", snapshots.get("daily")),
    ]
    lines = [
        f"## SuperTrend ({window}-period, 3x ATR) — trend direction and "
        "trailing stop by timeframe (higher tier wins):"
    ]
    for tier, name, snap in tiers:
        if snap is None:
            lines.append(f"- {tier} {name}: n/a (insufficient history)")
        else:
            direction = snap["direction"].upper()
            lines.append(
                f"- {tier} {name}: {direction} — trailing stop {snap['level']:.2f} "
                f"(close {snap['distance_pct']:+.2f}% from stop)"
            )
    return "\n".join(lines)


def get_stock_stats_indicators_window(
    symbol: Annotated[str, "ticker symbol of the company"],
    indicator: Annotated[str, "technical indicator to get the analysis and report of"],
    as_of_date: Annotated[str, "The current trading date you are trading on, YYYY-mm-dd"],
    look_back_days: Annotated[int, "how many days to look back"],
) -> str:

    spec = get_spec(indicator)  # unknown names raise ValueError
    params = effective_params(indicator)  # spec defaults + config overrides

    # TD-9 is not a stockstats handler — compute it from OHLCV across timeframes
    # and return the tiered block (weekly > monthly > daily) directly.
    if spec.custom == "td_9":
        counts = td_setup_by_timeframe(load_ohlcv(symbol, as_of_date), as_of_date)
        return format_td_setup_block(counts) + "\n\n" + spec.description

    # Z-Score is likewise computed from OHLCV across timeframes, not stockstats.
    if spec.custom == "z_score":
        zscores = zscore_by_timeframe(
            load_ohlcv(symbol, as_of_date), as_of_date, window=params["window"]
        )
        return format_zscore_block(zscores, window=params["window"]) + "\n\n" + spec.description

    # SuperTrend resamples OHLCV to weekly/monthly bars itself, so it also
    # returns a tiered block rather than a windowed daily listing.
    if spec.custom == "supertrend":
        snapshots = supertrend_by_timeframe(
            load_ohlcv(symbol, as_of_date), as_of_date, window=params["window"]
        )
        return (
            format_supertrend_block(snapshots, window=params["window"]) + "\n\n" + spec.description
        )

    end_date = as_of_date
    as_of_dt = datetime.strptime(as_of_date, "%Y-%m-%d")
    before = as_of_dt - relativedelta(days=look_back_days)

    # Optimized: Get stock data once and calculate indicators for all dates
    try:
        if spec.custom == "obv":
            # OBV is computed from OHLCV (not a stockstats handler) but reads
            # like any daily series, so it renders through the same window.
            obv = compute_obv(load_ohlcv(symbol, as_of_date))
            indicator_data = {
                date.strftime("%Y-%m-%d"): str(float(value)) for date, value in obv.items()
            }
        else:
            indicator_data = _get_stock_stats_bulk(symbol, indicator, as_of_date)

        # Generate the date range we need
        current_dt = as_of_dt
        date_values = []

        while current_dt >= before:
            date_str = current_dt.strftime("%Y-%m-%d")

            # Look up the indicator value for this date
            if date_str in indicator_data:
                indicator_value = indicator_data[date_str]
            else:
                indicator_value = "N/A: Not a trading day (weekend or holiday)"

            date_values.append((date_str, indicator_value))
            current_dt = current_dt - relativedelta(days=1)

        # Build the result string
        ind_string = ""
        for date_str, value in date_values:
            ind_string += f"{date_str}: {value}\n"

    except VendorError:
        raise  # Unknown/delisted symbol or outage — let the router emit the sentinel
    except Exception as e:
        logger.warning("Bulk stockstats fetch failed, falling back per-day: %s", e)
        # Fallback to original implementation if bulk method fails
        ind_string = ""
        as_of_dt = datetime.strptime(as_of_date, "%Y-%m-%d")
        while as_of_dt >= before:
            indicator_value = get_stockstats_indicator(
                symbol, indicator, as_of_dt.strftime("%Y-%m-%d")
            )
            ind_string += f"{as_of_dt.strftime('%Y-%m-%d')}: {indicator_value}\n"
            as_of_dt = as_of_dt - relativedelta(days=1)

    # Show effective params in the header only when they differ from the
    # defaults, so unconfigured output stays byte-identical.
    label = indicator
    if params and params != spec.default_params:
        label += " (" + ", ".join(f"{k}={v}" for k, v in params.items()) + ")"

    result_str = (
        f"## {label} values from {before.strftime('%Y-%m-%d')} to {end_date}:\n\n"
        + ind_string
        + "\n\n"
        + spec.description
    )

    return result_str


def _get_stock_stats_bulk(
    symbol: Annotated[str, "ticker symbol of the company"],
    indicator: Annotated[str, "technical indicator to calculate"],
    as_of_date: Annotated[str, "current date for reference"],
) -> dict:
    """
    Optimized bulk calculation of stock stats indicators.
    Fetches data once and calculates indicator for all available dates.
    Returns dict mapping date strings to indicator values.
    """
    spec = get_spec(indicator)
    column = resolve_column(indicator, effective_params(indicator))

    data = load_ohlcv(symbol, as_of_date)
    df = wrap(data)
    df["Date"] = df["Date"].dt.strftime("%Y-%m-%d")

    # Calculate the indicator for all rows at once
    df[column]  # This triggers stockstats to calculate the indicator

    # Create a dictionary mapping date strings to indicator values
    result_dict = {}
    for _, row in df.iterrows():
        date_str = row["Date"]
        indicator_value = row[column]

        # Handle NaN/None values
        if pd.isna(indicator_value):
            result_dict[date_str] = "N/A"
        elif spec.scale != 1.0:
            result_dict[date_str] = str(float(indicator_value) * spec.scale)
        else:
            result_dict[date_str] = str(indicator_value)

    return result_dict


def get_stockstats_indicator(
    symbol: Annotated[str, "ticker symbol of the company"],
    indicator: Annotated[str, "technical indicator to get the analysis and report of"],
    as_of_date: Annotated[str, "The current trading date you are trading on, YYYY-mm-dd"],
) -> str:

    as_of_dt = datetime.strptime(as_of_date, "%Y-%m-%d")
    as_of_date = as_of_dt.strftime("%Y-%m-%d")

    try:
        spec = get_spec(indicator)
        indicator_value = get_stock_stats(
            symbol,
            resolve_column(indicator, effective_params(indicator)),
            as_of_date,
        )
        # Non-trading days come back as an explanatory string; only numeric
        # readings are rescaled (stockstats mfi is 0-1, reported as 0-100).
        if spec.scale != 1.0 and not isinstance(indicator_value, str):
            indicator_value = float(indicator_value) * spec.scale
    except VendorError:
        raise  # Unknown/delisted symbol or outage — let the router emit the sentinel
    except Exception as e:
        # An empty string renders as "2026-05-08: " in the indicator table, which
        # reads as no value that day rather than a read that failed. Raise so the
        # router can try the next vendor or report the series unavailable.
        raise NoMarketDataError(
            symbol, symbol, f"{indicator} could not be read for {as_of_date}: {e}"
        ) from e

    return str(indicator_value)


def get_stock_stats(
    symbol: Annotated[str, "ticker symbol for the company"],
    indicator: Annotated[
        str, "quantitative indicators based off of the stock data for the company"
    ],
    as_of_date: Annotated[str, "curr date for retrieving stock price data, YYYY-mm-dd"],
):
    data = load_ohlcv(symbol, as_of_date)
    df = wrap(data)
    df["Date"] = df["Date"].dt.strftime("%Y-%m-%d")
    as_of_str = pd.to_datetime(as_of_date).strftime("%Y-%m-%d")

    df[indicator]  # trigger stockstats to calculate the indicator
    matching_rows = df[df["Date"].str.startswith(as_of_str)]

    if not matching_rows.empty:
        indicator_value = matching_rows[indicator].values[0]
        return indicator_value
    else:
        return "N/A: Not a trading day (weekend or holiday)"
