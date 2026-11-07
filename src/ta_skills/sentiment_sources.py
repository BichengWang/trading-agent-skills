"""The sentiment analyst's pre-fetch: news, StockTwits and Reddit, trimmed to the analysis window.

The analyst reports on data it was given rather than inventing posts, so the three
sources are fetched before it runs and put in its prompt. Each fetcher degrades
gracefully and returns a string, so the analyst always sees either real data or a
clear placeholder. With a TypeSafe key, the social posts are screened by Jev first
(see ``post_screen``). These feeds serve recent items and are not archived, so a
historical run's sentiment inputs are not point-in-time.
"""

from datetime import datetime, timedelta

from ta_skills.config import get_config
from ta_skills.data.vendors.reddit import (
    CRYPTO_SUBREDDITS,
    DEFAULT_SUBREDDITS,
    fetch_reddit_posts,
    subreddits_for,
)
from ta_skills.data.vendors.stocktwits import fetch_stocktwits_messages
from ta_skills.post_screen import jev_screen
from ta_skills.tools import call_tool


def _seven_days_back(trade_date: str) -> str:
    return (datetime.strptime(trade_date, "%Y-%m-%d") - timedelta(days=7)).strftime("%Y-%m-%d")


def _character(subreddits: tuple[str, ...]) -> str:
    """How the analyst is told to read the subreddits it was given."""
    if subreddits == DEFAULT_SUBREDDITS:
        return "r/wallstreetbets is often contrarian/exuberant; r/stocks more measured; r/investing longer-term"
    if len(subreddits) > len(CRYPTO_SUBREDDITS):
        return f"r/{subreddits[0]} leans toward the coin's holders; r/CryptoCurrency and r/CryptoMarkets are broader"
    return "r/CryptoCurrency and r/CryptoMarkets are broad crypto communities"


def build_sentiment_inputs(ticker: str, trade_date: str) -> dict:
    """The sentiment analyst's data blocks for ``ticker`` on ``trade_date``.

    Returns ``{"start_date", "end_date", "news_block", "stocktwits_block",
    "reddit_block", "subreddits", "character"}``: the window is the seven days up
    to and including ``trade_date``; ``subreddits`` is the list the Reddit block
    was drawn from and ``character`` the line describing their tone.
    """
    end_date = trade_date
    start_date = _seven_days_back(end_date)

    # Pre-fetch all three sources. Each fetcher degrades gracefully and returns
    # a string, so the LLM always sees something — either real data or a clear
    # placeholder.
    news_block = call_tool("get_news", ticker, trade_date, start_date=start_date, end_date=end_date)
    # Pass the analysis window so a historical run trims social posts to it
    # instead of leaking today's chatter into a backtest (#1220).
    screen = jev_screen(ticker)
    stocktwits_block = fetch_stocktwits_messages(
        ticker, limit=30, start_date=start_date, end_date=end_date, screen=screen
    )
    subreddits = subreddits_for(ticker)
    include_reddit = get_config().get("sentiment_include_reddit", True)
    if isinstance(include_reddit, str):
        include_reddit = include_reddit.strip().lower() not in {"0", "false", "no", "off"}
    reddit_block = (
        fetch_reddit_posts(
            ticker, subreddits, start_date=start_date, end_date=end_date, screen=screen
        )
        if include_reddit
        else "<reddit skipped: disabled by sentiment_include_reddit config>"
    )

    return {
        "start_date": start_date,
        "end_date": end_date,
        "news_block": news_block,
        "stocktwits_block": stocktwits_block,
        "reddit_block": reddit_block,
        "subreddits": list(subreddits),
        "character": _character(subreddits),
    }
