"""Suite-wide isolation, set before ta_skills is first imported.

``ta_skills.config`` folds the TRADINGAGENTS_* environment into DEFAULT_CONFIG when
it is imported, so these must be in place before any test module imports the package.
"""

import os
import tempfile


def _blank_settings_overlay():
    """Blank every TRADINGAGENTS_* setting so a contributor's own settings do not
    become the defaults the suite asserts on. A blank value is still present, so
    load_dotenv leaves it alone, and the overlay reads it as unset."""
    from dotenv import dotenv_values, find_dotenv

    names = set(os.environ)
    names |= set(dotenv_values(find_dotenv(".env", usecwd=True)))
    for name in names:
        if name.startswith("TRADINGAGENTS_"):
            os.environ[name] = ""


def _own_file_locations():
    """Keep the suite's results, cache, runs and memory log in a directory of its own:
    a test reading the user's cache would see what their runs fetched."""
    home = tempfile.mkdtemp(prefix="tradingagents-tests-")
    os.environ["TRADINGAGENTS_RESULTS_DIR"] = os.path.join(home, "logs")
    os.environ["TRADINGAGENTS_CACHE_DIR"] = os.path.join(home, "cache")
    os.environ["TRADINGAGENTS_RUNS_DIR"] = os.path.join(home, "runs")
    os.environ["TRADINGAGENTS_REPORTS_DIR"] = os.path.join(home, "reports")
    os.environ["TRADINGAGENTS_MEMORY_LOG_PATH"] = os.path.join(home, "memory", "trading_memory.md")


_blank_settings_overlay()
_own_file_locations()


def pytest_configure(config):
    # yfinance caches each symbol's time zone on disk, in the home directory
    # by default; the suite keeps it in a directory of its own.
    import yfinance

    yfinance.set_tz_cache_location(tempfile.mkdtemp(prefix="tradingagents-tests-yf-"))
