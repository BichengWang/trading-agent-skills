"""Shared fixtures for the run-state tests: an offline run and canned role outputs."""

from __future__ import annotations

import json
from pathlib import Path

from ta_skills import run

TRADE_DATE = "2025-03-14"

IDENTITY = {
    "company_name": "NVIDIA Corporation", "sector": "Technology", "industry": "Semiconductors",
    "exchange": "NMS",
}
SENTIMENT_INPUTS = {
    "start_date": "2025-03-07", "end_date": TRADE_DATE, "news_block": "NEWS-BLOCK",
    "stocktwits_block": "ST-BLOCK", "reddit_block": "REDDIT-BLOCK",
    "subreddits": ["wallstreetbets", "stocks", "investing"],
    "character": "r/wallstreetbets is often contrarian/exuberant; r/stocks more measured; "
                 "r/investing longer-term",
}

SENTIMENT = {
    "overall_band": "Bullish", "overall_score": 7.2, "confidence": "medium",
    "narrative": "StockTwits leans bullish.\n\n| Source | Direction |\n|---|---|\n| news | up |",
}
RESEARCH_PLAN = {
    "recommendation": "Buy", "rationale": "The bull case carried the debate.",
    "strategic_actions": "Build the position in two tranches.",
}
TRADER = {
    "action": "Buy", "reasoning": "Plan and price structure agree.", "entry_price": 100.5,
    "stop_loss": 95.0, "position_sizing": "5% of portfolio",
}
PM = {
    "rating": "Overweight", "executive_summary": "Add gradually.",
    "investment_thesis": "Street consensus rating: Hold. Our target: 12x forward FCF.",
    "price_target": 120.0, "current_price": 101.0, "confidence": "medium", "time_horizon": "3-6 months",
}

STRUCTURED_JSON = {
    "sentiment": SENTIMENT, "research_manager": RESEARCH_PLAN, "trader": TRADER, "portfolio_manager": PM,
}
FREE_TEXT = {
    "market": "**Current Price**: 101.0\n\n# Market view\n\nTrend is up.",
    "news": "News body.",
    "fundamentals": "Fundamentals body.",
}


def canned(role: str, turn: int = 1) -> tuple[str, str]:
    """``(extension, content)`` of a valid submission for ``role``."""
    if role in STRUCTURED_JSON:
        return "json", json.dumps(STRUCTURED_JSON[role])
    if role in FREE_TEXT:
        return "md", FREE_TEXT[role]
    return "md", f"{role} argument, turn {turn}."


def submit(run_dir, role: str, content: str | None = None, ext: str | None = None, turn: int = 1):
    """Write a draft and submit it; ``content=None`` submits the canned output."""
    if role == "memory":
        return run.submit_role(run_dir, "memory")
    default_ext, default_content = canned(role, turn)
    path = Path(run_dir) / "drafts" / f"{role}.{ext or default_ext}"
    path.write_text(default_content if content is None else content, encoding="utf-8")
    return run.submit_role(run_dir, role, path)


def drive(run_dir) -> list[str]:
    """Submit canned output for every runnable role until the run is done; return the order run."""
    order: list[str] = []
    while True:
        phase, roles = run.next_phase(run.load_state(run_dir))
        if phase == "done":
            return order
        assert roles, phase
        for role in roles:
            turn = len([r for r in order if r == role]) + 1
            submit(run_dir, role, turn=turn)
            order.append(role)


def new_run(**kwargs) -> Path:
    kwargs.setdefault("ticker", "NVDA")
    kwargs.setdefault("trade_date", TRADE_DATE)
    return run.init_run(**kwargs)
