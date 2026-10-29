"""Instrument identity and asset type: what the agents are told the instrument is."""

import functools
import logging
from typing import Any

from ta_skills.data.symbols import normalize_symbol
from ta_skills.data.vendors.yahoo.fundamentals import get_company_profile

logger = logging.getLogger(__name__)

# Canonical Yahoo symbols ending in one of these are crypto pairs.
CRYPTO_SUFFIXES = ("-USD", "-USDT", "-USDC", "-BTC", "-ETH")


def detect_asset_type(ticker: str) -> str:
    """``"crypto"`` for a crypto pair, else ``"stock"`` (ETFs and indices are stocks here).

    Classifies on the canonical symbol so e.g. BTCUSD and BTC-USDT both read as
    crypto (#981/#982), matching what the data path will actually fetch.
    """
    try:
        canonical = normalize_symbol(ticker)
    except Exception:
        canonical = ticker.strip().upper()
    return "crypto" if canonical.endswith(CRYPTO_SUFFIXES) else "stock"


def _clean_identity_value(value: Any) -> str | None:
    """Return a trimmed string, or None for empty / placeholder-ish values."""
    if not isinstance(value, str):
        return None
    cleaned = value.strip()
    if not cleaned or cleaned.lower() in {"none", "n/a", "nan", "null"}:
        return None
    return cleaned


def resolve_instrument_identity(
    ticker: str, asset_type: str = "stock", trade_date: str | None = None
) -> dict | None:
    """Resolve deterministic identity metadata (company name, sector, ...) for a ticker.

    This exists to stop the pipeline from hallucinating a *different* company
    when a chart pattern suggests a different industry than the real one
    (#814): without a ground-truth name, the market analyst would pattern-match
    the price action to a narrative and invent an identity that then cascaded
    through every downstream agent.

    Returns ``{"company_name", "sector", "industry", "exchange", "quote_type"}``
    (the fields Yahoo has), or None when nothing could be resolved. Best-effort
    by design: if yfinance is unavailable, rate-limited, or doesn't recognise
    the ticker, the caller falls back to ticker-only context rather than failing
    before analysis starts. An answer is cached for the process; a failed lookup
    is asked again next time.

    Identity resolves for the same instrument the price path fetches
    (``XAUUSD`` -> ``GC=F``, #983). Yahoo's profile has no historical vintage:
    ``asset_type`` and ``trade_date`` do not change the lookup, they are the run's
    context for the caller, whose ``build_instrument_context`` decides how much of
    a profile an earlier-dated run may be told.
    """
    try:
        return _identity(ticker) or None
    except Exception as exc:  # noqa: BLE001 — fail open, never block the run
        logger.debug("Could not resolve instrument identity for %s: %s", ticker, exc)
        return None


@functools.lru_cache(maxsize=256)
def _identity(ticker: str) -> dict:
    """The vendor's identity fields for ``ticker``; raises if the lookup fails."""
    info = get_company_profile(ticker)
    identity: dict[str, str] = {}
    company_name = _clean_identity_value(info.get("longName")) or _clean_identity_value(
        info.get("shortName")
    )
    if company_name:
        identity["company_name"] = company_name
    for source_key, target_key in (
        ("sector", "sector"),
        ("industry", "industry"),
        ("exchange", "exchange"),
        ("quoteType", "quote_type"),
    ):
        value = _clean_identity_value(info.get(source_key))
        if value:
            identity[target_key] = value
    return identity
