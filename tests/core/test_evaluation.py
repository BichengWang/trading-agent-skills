"""A forecast is scored on what it said, against prices it could not have seen.

The ledger must read the Portfolio Manager's own fields, not a rating argued in
the debate before them. Outcomes must enter after the analysis date, stay
pending until the whole window has traded, give an instrument that is its own
benchmark no alpha, and rescale a target written before a split. A shrunk
target may learn its factor only from target windows that had already closed.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from ta_skills.evaluation import outcomes
from ta_skills.evaluation.ledger import forecast_from_page, parse_horizon
from ta_skills.evaluation.metrics import cohort_ic, decisions, target_section
from ta_skills.evaluation.outcomes import score_forecast, score_ledger

PATH = "NVDA/20260603_model-a_20260603_120000/complete_report/index.html"
PAGE = """<article>
<h2 id="iv-risk-management-team-decision">IV. Risk</h2>
<p><strong>Rating</strong>: Buy</p>
<h2 id="v-portfolio-manager-decision">V. Portfolio Manager Decision</h2>
<p><strong>Rating</strong>: Underweight</p>
<p><strong>Current Price</strong>: $1,234.50 (2026-06-03)</p>
<p><strong>Price Target</strong>: 1,100</p>
<p><strong>Confidence</strong>: Medium</p>
<p><strong>Time Horizon</strong>: 3–6 months</p>
</article>"""


def _prices(start: str, closes: list[float], splits: dict[int, float] | None = None) -> pd.DataFrame:
    index = pd.bdate_range(start, periods=len(closes))
    frame = pd.DataFrame({"close": closes, "adj_close": closes, "high": closes, "low": closes,
                          "splits": 0.0}, index=index)
    for i, ratio in (splits or {}).items():
        frame.iloc[i, frame.columns.get_loc("splits")] = ratio
    return frame


def _forecast(**overrides) -> dict:
    base = {"path": PATH, "ticker": "NVDA", "date": "2026-06-03", "model": "model-a",
            "generated_at": "2026-06-03T12:00:00", "rating": "Overweight", "confidence": None,
            "time_horizon": None, "horizon_min_td": None, "horizon_max_td": None,
            "reported_price": None, "price_target": None}
    return {**base, **overrides}


@pytest.mark.unit
def test_the_ledger_reads_the_final_decision_not_the_debate_before_it():
    f = forecast_from_page(PATH, PAGE, "abc")
    assert (f.rating, f.reported_price, f.price_target, f.confidence) == (
        "Underweight", 1234.5, 1100.0, "medium")
    assert (f.date, f.generated_at) == ("2026-06-03", "2026-06-03T12:00:00")
    assert (f.horizon_min_td, f.horizon_max_td) == (63, 126)


@pytest.mark.unit
@pytest.mark.parametrize("text, expected", [
    ("1-3 months", (21, 63)),
    ("2–4 weeks", (10, 20)),
    ("12-month horizon", (252, 252)),
    ("6 to 12 months", (126, 252)),
    ("Through the next earnings release", None),
])
def test_stated_horizons_become_trading_days(text, expected):
    assert parse_horizon(text) == expected


@pytest.mark.unit
def test_entry_is_the_first_close_after_the_analysis_date_and_short_windows_stay_pending():
    stock = _prices("2026-06-01", [100.0, 101.0, 102.0, 110.0, 111.0, 112.0, 113.0, 114.0, 115.0])
    bench = _prices("2026-06-01", [50.0] * 9)
    row = score_forecast(_forecast(), stock, bench, pd.Timestamp("2026-06-11"), horizons=(5, 21))
    assert row["entry_date"] == "2026-06-04"   # 2026-06-03 is the analysis date: enter after it
    assert row["p0"] == 102.0
    assert row["raw5"] == pytest.approx(115.0 / 110.0 - 1)
    assert row["alpha5"] == pytest.approx(115.0 / 110.0 - 1)
    assert "raw21" not in row


@pytest.mark.unit
def test_bars_after_the_as_of_date_are_not_used():
    stock = _prices("2026-06-01", [100.0] * 12)
    row = score_forecast(_forecast(), stock, _prices("2026-06-01", [50.0] * 12),
                         pd.Timestamp("2026-06-10"), horizons=(5,))
    assert "raw5" not in row   # the window has traded since, but not by the as-of date


@pytest.mark.unit
def test_an_instrument_that_is_its_own_benchmark_has_no_alpha():
    stock = _prices("2026-06-01", [100.0 + i for i in range(10)])
    row = score_forecast(_forecast(ticker="SPY"), stock, None, pd.Timestamp("2026-06-30"),
                         horizons=(5,))
    assert row["raw5"] > 0 and "alpha5" not in row


@pytest.mark.unit
def test_a_benchmark_that_has_not_traded_through_the_exit_leaves_alpha_pending():
    stock = _prices("2026-06-01", [100.0 + i for i in range(10)])
    row = score_forecast(_forecast(), stock, _prices("2026-06-01", [50.0] * 6),
                         pd.Timestamp("2026-06-30"), horizons=(5,))
    assert "raw5" in row and "alpha5" not in row


@pytest.mark.unit
def test_a_target_written_before_a_split_is_rescaled_with_the_history():
    # 10-for-1 split on the entry session: Yahoo has divided the earlier closes by 10,
    # so the report's 102 close reads as 10.2 now.
    stock = _prices("2026-06-01", [10.0, 10.1, 10.2, 10.5, 10.6, 10.7, 10.8, 10.9, 11.0],
                    splits={3: 10.0})
    row = score_forecast(_forecast(price_target=110.0, reported_price=102.0,
                                   horizon_min_td=2, horizon_max_td=2),
                         stock, None, pd.Timestamp("2026-06-30"), horizons=())
    assert row["p0"] == 10.2
    assert (row["price_target"], row["reported_price"]) == (pytest.approx(11.0), pytest.approx(10.2))


@pytest.mark.unit
def test_a_split_after_the_as_of_date_still_rescales_the_target(monkeypatch):
    # Like Yahoo: closes already divided by 3 for a 3-for-1 split on 2026-07-01, which
    # is after the as-of date; only rows in [start, end) come back.
    history = _prices("2026-01-01", [100.0] * 260)
    history.loc[pd.Timestamp("2026-07-01"), "splits"] = 3.0

    def fake(symbol, start, end):
        return history[(history.index >= start) & (history.index < end)]

    monkeypatch.setattr(outcomes, "get_price_history", fake)
    forecast = _forecast(ticker="SOXX", price_target=330.0, horizon_min_td=2)
    rows, _ = score_ledger([forecast], {"benchmark_ticker": "SOXX"}, "2026-06-30")
    assert rows.loc[0, "price_target"] == pytest.approx(110.0)   # written when 300 was 100 now


@pytest.mark.unit
def test_scoring_skips_runs_generated_after_entry_and_keeps_the_newest_rerun():
    rows = pd.DataFrame([
        {"ticker": "A", "model": "m", "date": "2026-06-03", "generated_at": "2026-06-03T09:00:00",
         "status": "ok", "retro": False, "rating": "Hold"},
        {"ticker": "A", "model": "m", "date": "2026-06-03", "generated_at": "2026-06-03T15:00:00",
         "status": "ok", "retro": False, "rating": "Buy"},
        {"ticker": "B", "model": "m", "date": "2026-06-03", "generated_at": "2026-06-09T09:00:00",
         "status": "ok", "retro": True, "rating": "Sell"},
    ])
    kept, coverage = decisions(rows)
    assert kept["rating"].tolist() == ["Buy"]
    assert (coverage["retro_excluded"], coverage["reruns_dropped"]) == (1, 1)


@pytest.mark.unit
def test_rank_ic_is_per_cohort_and_skips_cohorts_too_small_to_rank():
    big = pd.DataFrame({"date": "2026-06-03", "score": [2, 1, 0, -1, -2],
                        "alpha": [0.05, 0.02, 0.0, -0.01, -0.04]})
    small = pd.DataFrame({"date": "2026-06-10", "score": [2, -2], "alpha": [-0.1, 0.1]})
    ic = cohort_ic(pd.concat([big, small]), "score", "alpha")
    assert ic.to_dict() == {"2026-06-03": pytest.approx(1.0)}


@pytest.mark.unit
def test_a_target_window_records_the_session_it_closed_on():
    stock = _prices("2026-06-01", [100.0 + i for i in range(10)])
    row = score_forecast(_forecast(price_target=110.0, horizon_min_td=2, horizon_max_td=4),
                         stock, None, pd.Timestamp("2026-06-30"), horizons=())
    # Entry 2026-06-04 (Thu): two sessions on is Mon 06-08, four is Wed 06-10.
    assert (row["pt_exit_min"], row["pt_exit_max"]) == ("2026-06-08", "2026-06-10")
    assert (row["pt_close_min"], row["pt_close_max"]) == (105.0, 107.0)


def _targets(date: str, exit_: str, n: int, implied: float, realized: float) -> pd.DataFrame:
    """n targets on one date, alternating up and down, with the given log moves."""
    sign = np.resize([1.0, -1.0], n)
    return pd.DataFrame({"date": date, "pt_exit_min": exit_, "model": "m", "p0": 100.0,
                         "price_target": 100.0 * np.exp(implied * sign),
                         "pt_close_min": 100.0 * np.exp(realized * sign)})


@pytest.mark.unit
def test_shrunk_targets_learn_their_factor_only_from_windows_already_closed():
    # Targets that ask for four times the move that follows, in the right direction.
    early = _targets("2026-01-05", "2026-02-02", 30, 0.20, 0.05)
    late = _targets("2026-03-02", "2026-04-01", 10, 0.20, 0.05)
    w = target_section(pd.concat([early, late], ignore_index=True), "min")["walk_forward_shrunk"]
    assert w["n"] == 10   # on the early date no window had closed yet
    assert w["latest_factor"] == pytest.approx(0.25)
    assert w["target_skill"] == pytest.approx(-2.0)
    assert w["shrunk_skill"] == pytest.approx(1.0)


@pytest.mark.unit
def test_a_date_never_learns_its_shrink_factor_from_its_own_outcomes():
    same_day = _targets("2026-01-05", "2026-02-02", 40, 0.20, 0.05)
    assert target_section(same_day, "min")["walk_forward_shrunk"] is None


@pytest.mark.unit
def test_targets_that_point_the_wrong_way_shrink_to_the_random_walk():
    early = _targets("2026-01-05", "2026-02-02", 30, 0.20, -0.05)
    late = _targets("2026-03-02", "2026-04-01", 10, 0.20, -0.05)
    w = target_section(pd.concat([early, late], ignore_index=True), "min")["walk_forward_shrunk"]
    assert w["latest_factor"] == 0.0
    assert w["shrunk_skill"] == pytest.approx(0.0)


@pytest.mark.unit
def test_a_target_equal_to_the_price_has_no_direction_despite_float_rounding():
    # The report's 581.70 target against Yahoo's 581.700012 close, rescaled for a split.
    q = pd.DataFrame({"date": "2026-07-09", "model": "m", "p0": [581.700012 / 3, 100.0],
                      "price_target": [581.70 / 3, 110.0], "pt_close_min": [187.76, 105.0]})
    s = target_section(q, "min")
    assert s["flat_targets"] == 1
    assert s["direction_agreement"] == 1.0   # only the 110 target is scored, and it was right
