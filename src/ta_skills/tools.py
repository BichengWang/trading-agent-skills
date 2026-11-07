"""The data tools the analysts call, as plain functions.

Each dated tool takes the run's ``trade_date`` from the run state and never serves
data past it, whatever date the model asks for. The instrument (``ticker``) and
``trade_date`` are injected by the caller and are not part of a tool's ``params``;
``call_tool`` is the single entry point and returns text for every vendor outcome,
including the ``NO_DATA_AVAILABLE:`` / ``DATA_UNAVAILABLE:`` sentinels.

A tool's ``description`` is its docstring, verbatim: it is the text the model sees.
"""

import inspect
import os
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

from ta_skills.data.date_window import as_of, as_of_window
from ta_skills.data.errors import NoMarketDataError, VendorUnavailableError
from ta_skills.data.router import no_data_available, route_to_vendor, vendor_unavailable
from ta_skills.data.vendors.yahoo.snapshot import build_verified_market_snapshot

# Marks a Param without a default: the caller must supply it.
REQUIRED = inspect.Parameter.empty

# ``get_indicators`` prints one line per calendar day of the look-back, so a
# model-supplied 3650 would emit 3650 lines. The default bound matches the
# snapshot's own window cap.
_MAX_INDICATOR_DAYS = 30


@dataclass(frozen=True)
class Param:
    """One model-supplied argument of a tool."""

    name: str
    type: type
    default: Any  # REQUIRED when the argument has no default
    help: str


@dataclass(frozen=True)
class ToolSpec:
    """A tool: its verbatim description, model-supplied params, and implementation.

    ``func(ticker, trade_date, **params)`` returns the tool's text output.
    """

    name: str
    description: str
    params: tuple[Param, ...]
    func: Callable[..., str]


def _bounded_lookback(look_back_days) -> int:
    max_days = int(os.getenv("TRADINGAGENTS_TOOL_MAX_INDICATOR_DAYS", str(_MAX_INDICATOR_DAYS)))
    return max(1, min(int(look_back_days), max_days))


def get_stock_data(ticker, trade_date, start_date, end_date):
    """
    Retrieve stock price data (OHLCV) for the instrument under analysis.
    Uses the configured core_stock_apis vendor.
    Args:
        start_date (str): Start date in yyyy-mm-dd format
        end_date (str): End date in yyyy-mm-dd format
    Returns:
        str: A formatted dataframe containing the price data for the instrument over the date range.
    """
    try:
        start_date, end_date = as_of_window(start_date, end_date, trade_date)
    except ValueError as e:
        return str(e)
    return route_to_vendor("get_stock_data", ticker, start_date, end_date)


def get_indicators(ticker, trade_date, indicator, curr_date, look_back_days=30):
    """
    Retrieve a single technical indicator for the instrument under analysis.
    Uses the configured technical_indicators vendor.
    Args:
        indicator (str): A single technical indicator name, e.g. 'rsi', 'macd'. Call this tool once per indicator.
        curr_date (str): The current trading date you are trading on, YYYY-mm-dd
        look_back_days (int): How many days to look back, default is 30
    Returns:
        str: A formatted dataframe containing the technical indicators for the instrument and indicator.
    """
    # LLMs sometimes pass multiple indicators as a comma-separated string;
    # split and process each individually.
    curr_date = as_of(curr_date, trade_date)
    look_back_days = _bounded_lookback(look_back_days)
    indicators = [i.strip().lower() for i in indicator.split(",") if i.strip()]
    results = []
    for ind in indicators:
        try:
            results.append(
                route_to_vendor("get_indicators", ticker, ind, curr_date, look_back_days)
            )
        except ValueError as e:
            results.append(str(e))
    return "\n\n".join(results)


_NO_VERIFIED_PRICES = (
    " Do not claim exact prices, price levels, or indicator values; "
    "report that verified market data is unavailable."
)


def get_verified_market_snapshot(ticker, trade_date, curr_date, look_back_days=30):
    """Deterministic verification snapshot for exact market-data claims.

    Returns the latest OHLCV row on or before curr_date, common technical
    indicators, and recent closes. Call this before making exact claims about
    price levels, Bollinger bands, RSI, MACD, moving averages, support /
    resistance, or historical comparisons, and treat it as the source of truth.
    """
    # A vendor failure must come back as text, not abort the caller.
    try:
        return build_verified_market_snapshot(ticker, as_of(curr_date, trade_date), look_back_days)
    except VendorUnavailableError as exc:
        return vendor_unavailable("get_verified_market_snapshot", exc) + _NO_VERIFIED_PRICES
    except NoMarketDataError as exc:
        return no_data_available(exc) + _NO_VERIFIED_PRICES


def get_fundamentals(ticker, trade_date, curr_date):
    """
    Retrieve comprehensive fundamental data for the instrument under analysis.
    Uses the configured fundamental_data vendor.
    Args:
        curr_date (str): Current date you are trading at, yyyy-mm-dd
    Returns:
        str: A formatted report containing comprehensive fundamental data
    """
    return route_to_vendor("get_fundamentals", ticker, as_of(curr_date, trade_date))


def get_balance_sheet(ticker, trade_date, freq="quarterly", curr_date=None):
    """
    Retrieve balance sheet data for the instrument under analysis.
    Uses the configured fundamental_data vendor.
    Args:
        freq (str): Reporting frequency: annual/quarterly (default quarterly)
        curr_date (str): Current date you are trading at, yyyy-mm-dd
    Returns:
        str: A formatted report containing balance sheet data
    """
    return route_to_vendor("get_balance_sheet", ticker, freq, as_of(curr_date, trade_date))


def get_cashflow(ticker, trade_date, freq="quarterly", curr_date=None):
    """
    Retrieve cash flow statement data for the instrument under analysis.
    Uses the configured fundamental_data vendor.
    Args:
        freq (str): Reporting frequency: annual/quarterly (default quarterly)
        curr_date (str): Current date you are trading at, yyyy-mm-dd
    Returns:
        str: A formatted report containing cash flow statement data
    """
    return route_to_vendor("get_cashflow", ticker, freq, as_of(curr_date, trade_date))


def get_income_statement(ticker, trade_date, freq="quarterly", curr_date=None):
    """
    Retrieve income statement data for the instrument under analysis.
    Uses the configured fundamental_data vendor.
    Args:
        freq (str): Reporting frequency: annual/quarterly (default quarterly)
        curr_date (str): Current date you are trading at, yyyy-mm-dd
    Returns:
        str: A formatted report containing income statement data
    """
    return route_to_vendor("get_income_statement", ticker, freq, as_of(curr_date, trade_date))


def get_news(ticker, trade_date, start_date, end_date):
    """
    Retrieve news data for the instrument under analysis.
    Uses the configured news_data vendor.
    Args:
        start_date (str): Start date in yyyy-mm-dd format
        end_date (str): End date in yyyy-mm-dd format
    Returns:
        str: A formatted string containing news data
    """
    try:
        start_date, end_date = as_of_window(start_date, end_date, trade_date)
    except ValueError as e:
        return str(e)
    return route_to_vendor("get_news", ticker, start_date, end_date)


def get_global_news(ticker, trade_date, curr_date, look_back_days=None, limit=None):
    """
    Retrieve global news data.
    Uses the configured news_data vendor. Defaults for look_back_days and
    limit come from DEFAULT_CONFIG (global_news_lookback_days,
    global_news_article_limit); pass explicit values to override.

    Args:
        curr_date (str): Current date in yyyy-mm-dd format
        look_back_days (int): Number of days to look back; omit to inherit config
        limit (int): Maximum number of articles to return; omit to inherit config

    Returns:
        str: A formatted string containing global news data
    """
    return route_to_vendor("get_global_news", as_of(curr_date, trade_date), look_back_days, limit)


def get_insider_transactions(ticker, trade_date):
    """
    Retrieve insider transaction information about a company.
    Uses the configured news_data vendor.
    Returns:
        str: A report of insider transaction data
    """
    return route_to_vendor("get_insider_transactions", ticker, trade_date or None)


def get_macro_indicators(ticker, trade_date, indicator, curr_date, look_back_days=None):
    """
    Retrieve a macroeconomic indicator time series from FRED (Federal Reserve
    Economic Data): policy rates, Treasury yields, inflation, labor, and growth,
    for the US and, through FRED's mirrored series, the euro area.
    Returns the series title, units, frequency, the latest value, the change
    over the window, and a recent observation table. Uses the configured
    macro_data vendor.

    Args:
        indicator (str): Friendly alias or raw FRED series ID
        curr_date (str): Current date in yyyy-mm-dd format
        look_back_days (int): Trailing window length; omit for a 1-year window

    Returns:
        str: A formatted markdown report of the macro series
    """
    return route_to_vendor(
        "get_macro_indicators", indicator, as_of(curr_date, trade_date), look_back_days
    )


def get_prediction_markets(ticker, trade_date, topic, limit=None):
    """
    Retrieve live, market-implied probabilities for forward-looking events from
    prediction markets (Polymarket): Fed decisions, recession, elections,
    geopolitics, crypto. Returns the most-traded open markets matching the
    topic, each with its implied probability, traded volume, resolution date,
    and recent move. Uses the configured prediction_markets vendor.

    Args:
        topic (str): Event keyword(s) to search
        limit (int): Max markets to return; omit for a default of 6

    Returns:
        str: A formatted markdown report of matching prediction markets
    """
    return route_to_vendor("get_prediction_markets", topic, limit, trade_date or None)


# Model-visible argument descriptions (the annotation strings of the source tools).
_START = Param("start_date", str, REQUIRED, "Start date in yyyy-mm-dd format")
_END = Param("end_date", str, REQUIRED, "End date in yyyy-mm-dd format")
_FREQ = Param("freq", str, "quarterly", "reporting frequency: annual/quarterly")
_STATEMENT_DATE = Param("curr_date", str, None, "current date you are trading at, yyyy-mm-dd")

_PARAMS: dict[str, tuple[Param, ...]] = {
    "get_stock_data": (_START, _END),
    "get_indicators": (
        Param("indicator", str, REQUIRED, "technical indicator to get the analysis and report of"),
        Param(
            "curr_date", str, REQUIRED, "The current trading date you are trading on, YYYY-mm-dd"
        ),
        Param("look_back_days", int, 30, "how many days to look back"),
    ),
    "get_verified_market_snapshot": (
        Param("curr_date", str, REQUIRED, "the current trading date, YYYY-mm-dd"),
        Param(
            "look_back_days",
            int,
            30,
            "number of recent trading rows to include for sanity-checking",
        ),
    ),
    "get_fundamentals": (
        Param("curr_date", str, REQUIRED, "current date you are trading at, yyyy-mm-dd"),
    ),
    "get_balance_sheet": (_FREQ, _STATEMENT_DATE),
    "get_cashflow": (_FREQ, _STATEMENT_DATE),
    "get_income_statement": (_FREQ, _STATEMENT_DATE),
    "get_news": (_START, _END),
    "get_global_news": (
        Param("curr_date", str, REQUIRED, "Current date in yyyy-mm-dd format"),
        Param("look_back_days", int, None, "Days to look back; omit to use the configured default"),
        Param("limit", int, None, "Max articles to return; omit to use the configured default"),
    ),
    "get_insider_transactions": (),
    "get_macro_indicators": (
        Param(
            "indicator",
            str,
            REQUIRED,
            "Macro indicator: a friendly alias such as 'cpi', 'core_pce', "
            "'unemployment', 'fed_funds_rate', '10y_treasury', 'yield_curve', "
            "'real_gdp', 'vix', euro-area aliases such as 'ecb_deposit_rate', "
            "'euro_hicp', 'euro_core_hicp', 'euro_real_gdp', 'germany_10y', "
            "'france_10y', 'eur_usd', or a raw FRED series ID such as 'CPIAUCSL'.",
        ),
        Param(
            "curr_date",
            str,
            REQUIRED,
            "Current date in yyyy-mm-dd format; the end of the window",
        ),
        Param(
            "look_back_days",
            int,
            None,
            "Trailing window length in days; omit for a 1-year window",
        ),
    ),
    "get_prediction_markets": (
        Param(
            "topic",
            str,
            REQUIRED,
            "Event topic/keyword, e.g. 'Fed rate cut', 'recession 2026', "
            "'US election', or a sector/company event.",
        ),
        Param("limit", int, None, "Max markets to return; omit for a default of 6"),
    ),
}

_FUNCS = (
    get_stock_data,
    get_indicators,
    get_verified_market_snapshot,
    get_fundamentals,
    get_balance_sheet,
    get_cashflow,
    get_income_statement,
    get_news,
    get_global_news,
    get_insider_transactions,
    get_macro_indicators,
    get_prediction_markets,
)

TOOLS: dict[str, ToolSpec] = {
    func.__name__: ToolSpec(
        name=func.__name__,
        description=inspect.getdoc(func),
        params=_PARAMS[func.__name__],
        func=func,
    )
    for func in _FUNCS
}


def _coerce(param: Param, value):
    """A value as the param's type; None and already-typed values pass through."""
    if value is None or isinstance(value, param.type):
        return value
    try:
        return param.type(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{param.name} must be {param.type.__name__}, got {value!r}") from exc


def call_tool(name: str, ticker: str, trade_date: str, **kwargs) -> str:
    """Run the tool ``name`` for ``ticker`` as of ``trade_date`` and return its text.

    Dates are clamped to ``trade_date`` by the tool itself. Vendor failures come
    back as the sentinel text (``NO_DATA_AVAILABLE:`` / ``DATA_UNAVAILABLE:``) or,
    for a vendor error that is neither, as an ``Error:`` line the agent can read,
    so a data problem never aborts the caller. An unknown tool or parameter, or a
    missing required parameter, is a caller error and raises ``ValueError``.
    """
    spec = TOOLS.get(name)
    if spec is None:
        raise ValueError(f"Unknown tool {name!r}. Available: {sorted(TOOLS)}")
    by_name = {p.name: p for p in spec.params}
    unknown = sorted(set(kwargs) - set(by_name))
    if unknown:
        raise ValueError(f"{name} has no parameter(s) {unknown}; it takes {sorted(by_name)}")
    values = {}
    for param in spec.params:
        if kwargs.get(param.name) is not None:
            values[param.name] = _coerce(param, kwargs[param.name])
        elif param.default is REQUIRED:
            raise ValueError(f"{name} requires parameter {param.name!r}")
    try:
        return spec.func(ticker, trade_date, **values)
    except Exception as exc:  # noqa: BLE001 — the agent reads the failure and carries on
        return f"Error: {exc!r}\n Please fix your mistakes."
