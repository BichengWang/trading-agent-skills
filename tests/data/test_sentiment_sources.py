"""The sentiment analyst's pre-fetch: window, sources, screen and the reddit switch."""

import pytest

from ta_skills import sentiment_sources
from ta_skills.config import set_config
from ta_skills.data.vendors.reddit import CRYPTO_SUBREDDITS, DEFAULT_SUBREDDITS

KEYS = {
    "start_date",
    "end_date",
    "news_block",
    "stocktwits_block",
    "reddit_block",
    "subreddits",
    "character",
}


@pytest.fixture
def sources(monkeypatch):
    """The three fetchers stubbed; ``calls`` records what each was handed."""
    calls = {}
    screen = object()
    monkeypatch.setattr(sentiment_sources, "jev_screen", lambda ticker: screen)

    def news(name, ticker, trade_date, **kwargs):
        calls["news"] = (name, ticker, trade_date, kwargs)
        return "news"

    def stocktwits(ticker, **kwargs):
        calls["stocktwits"] = (ticker, kwargs)
        return "stocktwits"

    def reddit(ticker, subreddits, **kwargs):
        calls["reddit"] = (ticker, subreddits, kwargs)
        return "reddit"

    monkeypatch.setattr(sentiment_sources, "call_tool", news)
    monkeypatch.setattr(sentiment_sources, "fetch_stocktwits_messages", stocktwits)
    monkeypatch.setattr(sentiment_sources, "fetch_reddit_posts", reddit)
    calls["screen"] = screen
    return calls


@pytest.mark.unit
def test_every_source_is_trimmed_to_the_seven_days_up_to_the_trade_date(sources):
    out = sentiment_sources.build_sentiment_inputs("NVDA", "2026-01-09")

    assert set(out) == KEYS
    assert (out["start_date"], out["end_date"]) == ("2026-01-02", "2026-01-09")
    assert sources["news"] == (
        "get_news",
        "NVDA",
        "2026-01-09",
        {"start_date": "2026-01-02", "end_date": "2026-01-09"},
    )
    window = {"start_date": "2026-01-02", "end_date": "2026-01-09"}
    assert sources["stocktwits"] == ("NVDA", {"limit": 30, "screen": sources["screen"], **window})
    assert sources["reddit"] == (
        "NVDA",
        DEFAULT_SUBREDDITS,
        {"screen": sources["screen"], **window},
    )
    assert (out["news_block"], out["stocktwits_block"], out["reddit_block"]) == (
        "news",
        "stocktwits",
        "reddit",
    )


@pytest.mark.unit
def test_the_default_subreddits_and_their_character(sources):
    out = sentiment_sources.build_sentiment_inputs("NVDA", "2026-01-09")
    assert out["subreddits"] == list(DEFAULT_SUBREDDITS)
    assert out["character"] == (
        "r/wallstreetbets is often contrarian/exuberant; r/stocks more measured; r/investing longer-term"
    )


@pytest.mark.unit
def test_a_coin_with_its_own_subreddit_and_other_crypto(sources):
    btc = sentiment_sources.build_sentiment_inputs("BTC-USD", "2026-01-09")
    assert btc["subreddits"][0] == "Bitcoin" and len(btc["subreddits"]) > len(CRYPTO_SUBREDDITS)
    assert btc["character"] == (
        "r/Bitcoin leans toward the coin's holders; r/CryptoCurrency and r/CryptoMarkets are broader"
    )
    other = sentiment_sources.build_sentiment_inputs("DOGE-USD", "2026-01-09")
    assert other["subreddits"] == list(CRYPTO_SUBREDDITS)
    assert other["character"] == "r/CryptoCurrency and r/CryptoMarkets are broad crypto communities"


@pytest.mark.unit
@pytest.mark.parametrize("setting", [False, "false", "0", "off"])
def test_reddit_can_be_switched_off(sources, setting):
    set_config({"sentiment_include_reddit": setting})
    out = sentiment_sources.build_sentiment_inputs("NVDA", "2026-01-09")
    assert out["reddit_block"] == "<reddit skipped: disabled by sentiment_include_reddit config>"
    assert "reddit" not in sources
    assert out["stocktwits_block"] == "stocktwits"
