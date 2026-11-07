"""The scorecard: how accurate matured forecasts were, against baselines, with uncertainty.

Every figure is computed per cohort (one analysis date) first and then averaged,
because forecasts on one date share a market move, and windows from nearby dates
overlap. Confidence intervals come from a circular block bootstrap over cohort
dates, with blocks as long as the horizon spans cohorts.
"""

from __future__ import annotations

import math

import numpy as np
import pandas as pd

from ta_skills.evaluation.outcomes import HORIZONS
from ta_skills.rating import RATINGS_5_TIER

MIN_COHORT = 5      # forecasts on a date before its cross-section is ranked
MIN_GROUP = 30      # forecasts before a model or confidence level is reported on its own
FLAT_MOVE = 1e-6    # |log(target / price)| below this is a target equal to the price
BOOTSTRAP_REPS = 2000
PERMUTATION_REPS = 500


def _num(value):
    if value is None or isinstance(value, bool):
        return value
    value = float(value)
    return None if math.isnan(value) else round(value, 4)


def decisions(outcomes: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """Scoreable decisions: priced, generated before entry, newest rerun per ticker/model/date."""
    status = outcomes["status"].value_counts().to_dict()
    priced = outcomes[outcomes["status"] == "ok"]
    live = priced[~priced["retro"].astype(bool)]
    kept = live.sort_values("generated_at").drop_duplicates(["ticker", "model", "date"], keep="last")
    return kept, {
        "rated": len(outcomes), "status": status,
        "retro_excluded": len(priced) - len(live), "reruns_dropped": len(live) - len(kept),
        "decisions": len(kept),
    }


def _ranked_cohorts(d: pd.DataFrame, signal: str, target: str) -> dict:
    """Per date, the (signal ranks, target ranks) of a cross-section that can be ranked."""
    cohorts = {}
    for date, g in d.dropna(subset=[signal, target]).groupby("date"):
        if len(g) >= MIN_COHORT and g[signal].nunique() > 1 and g[target].nunique() > 1:
            cohorts[date] = (g[signal].rank().to_numpy(), g[target].rank().to_numpy())
    return dict(sorted(cohorts.items()))


def _pearson(a: np.ndarray, b: np.ndarray) -> float:
    a, b = a - a.mean(), b - b.mean()
    den = math.sqrt(float((a * a).sum() * (b * b).sum()))
    return float((a * b).sum() / den) if den else float("nan")


def cohort_ic(d: pd.DataFrame, signal: str, target: str) -> pd.Series:
    """Spearman rank correlation of signal and outcome, one value per cohort date."""
    cohorts = _ranked_cohorts(d, signal, target)
    return pd.Series({date: _pearson(s, t) for date, (s, t) in cohorts.items()}, dtype=float)


def block_length(dates, horizon: int) -> int:
    """Cohorts one horizon spans: forecasts closer than that share part of their window."""
    days = pd.DatetimeIndex(pd.to_datetime(list(dates)))
    if len(days) < 2:
        return 1
    gap = max(1.0, float(np.median(np.diff(days.values).astype("timedelta64[D]").astype(int))) * 5 / 7)
    return int(min(max(1, math.ceil(horizon / gap)), max(1, len(days) // 3)))


def block_ci(values, block: int, reps: int = BOOTSTRAP_REPS, seed: int = 0):
    """95% interval of the mean by circular block bootstrap; None below three values."""
    values = np.asarray(values, dtype=float)
    n = len(values)
    if n < 3:
        return None
    rng = np.random.default_rng(seed)
    k = math.ceil(n / block)
    idx = (rng.integers(0, n, (reps, k))[:, :, None] + np.arange(block)) % n
    means = values[idx.reshape(reps, -1)[:, :n]].mean(axis=1)
    return [_num(np.percentile(means, 2.5)), _num(np.percentile(means, 97.5))]


def permutation_p(d: pd.DataFrame, signal: str, target: str, reps: int = PERMUTATION_REPS,
                  seed: int = 0) -> float | None:
    """Share of within-date shuffles of the signal whose mean IC is at least the observed one."""
    cohorts = list(_ranked_cohorts(d, signal, target).values())
    if not cohorts:
        return None
    observed = np.mean([_pearson(s, t) for s, t in cohorts])
    rng = np.random.default_rng(seed)
    hits = sum(
        np.mean([_pearson(rng.permutation(s), t) for s, t in cohorts]) >= observed
        for _ in range(reps)
    )
    return _num((hits + 1) / (reps + 1))


def _summary(series: pd.Series, horizon: int) -> dict:
    series = series.dropna()
    return {"mean": _num(series.mean()) if len(series) else None, "cohorts": len(series),
            "ci95": block_ci(series.to_numpy(), block_length(series.index, horizon))}


def _hit(frame: pd.DataFrame, target: str) -> float | None:
    called = frame[frame["score"] != 0].dropna(subset=[target])
    if called.empty:
        return None
    return _num((np.sign(called["score"]) == np.sign(called[target])).mean())


def rating_section(d: pd.DataFrame, h: int, permutation_reps: int = PERMUTATION_REPS) -> dict | None:
    """How well ratings ordered and signed benchmark-relative returns over h sessions."""
    alpha, raw = f"alpha{h}", f"raw{h}"
    if alpha not in d:
        return None
    m = d.dropna(subset=[alpha]).copy()
    if m.empty:
        return None
    # Return relative to the cohort's own universe (each ticker once): removes "the
    # tickers chosen beat the benchmark" from hit rates.
    universe = m.drop_duplicates(["date", "ticker"]).groupby("date")[raw].mean()
    m["vs_universe"] = m[raw] - m["date"].map(universe)

    ic = cohort_ic(m, "score", alpha)
    spread = m.groupby("date").apply(
        lambda g: g.loc[g["score"] > 0, alpha].mean() - g.loc[g["score"] < 0, alpha].mean(),
        include_groups=False,
    ).dropna()
    within = m[m.groupby("ticker")["score"].transform("nunique") > 1]
    within_corr = None
    if len(within) >= MIN_GROUP:
        s = within["score"] - within.groupby("ticker")["score"].transform("mean")
        a = within[alpha] - within.groupby("ticker")[alpha].transform("mean")
        within_corr = _num(s.rank().corr(a.rank()))

    by_model = []
    for model, g in m.groupby("model"):
        called = g[g["score"] != 0]
        by_model.append({
            "model": model, "n": len(g), "called": len(called), "hit_rate": _hit(g, alpha),
            "mean_signed_alpha": _num((np.sign(called["score"]) * called[alpha]).mean())
            if len(called) else None,
        })
    # Where models split on one ticker and date, was the more bullish side right?
    split = m[m.groupby(["ticker", "date"])["score"].transform("nunique") > 1].copy()
    split["lean"] = split["score"] - split.groupby(["ticker", "date"])["score"].transform("mean")
    split = split[split["lean"] != 0]
    wins = (np.sign(split["lean"]) == np.sign(split[alpha])).groupby(split["model"]).agg(["count", "mean"])
    for row in by_model:
        if row["model"] in wins.index:
            row["disagreements"] = int(wins.loc[row["model"], "count"])
            row["disagreement_win_rate"] = _num(wins.loc[row["model"], "mean"])

    return {
        "n": len(m), "cohorts": int(m["date"].nunique()),
        "by_rating": [
            {"rating": r, "n": int((m["rating"] == r).sum()),
             "mean_alpha": _num(m.loc[m["rating"] == r, alpha].mean()),
             "median_alpha": _num(m.loc[m["rating"] == r, alpha].median())}
            for r in RATINGS_5_TIER if (m["rating"] == r).any()
        ],
        "hit_rate": _hit(m, alpha),
        "hit_rate_vs_universe": _hit(m, "vs_universe"),
        "rank_ic": {**_summary(ic, h), "permutation_p": permutation_p(m, "score", alpha, permutation_reps)},
        "long_short_alpha": _summary(spread, h),
        "baselines": {
            "always_hold_ic": 0.0,
            "momentum21_ic": _summary(cohort_ic(m, "mom21", alpha), h),
            "momentum63_ic": _summary(cohort_ic(m, "mom63", alpha), h),
        },
        "rating_vs_momentum63_rank_corr": _num(m["score"].rank().corr(m["mom63"].rank())),
        "within_ticker_rank_corr": within_corr,
        "ic_by_month": {
            month: {"mean": _num(v.mean()), "cohorts": len(v)}
            for month, v in ic.groupby(pd.to_datetime(ic.index).strftime("%Y-%m"))
        },
        "by_model": sorted(by_model, key=lambda r: -r["n"]),
        "by_confidence": [
            {"confidence": c, "n": int((m["confidence"] == c).sum()),
             "hit_rate": _hit(m[m["confidence"] == c], alpha)}
            for c in ("low", "medium", "high") if (m["confidence"] == c).any()
        ],
    }


def shrunk_targets(q: pd.DataFrame, implied: pd.Series, realized: pd.Series,
                   exit_col: str) -> dict | None:
    """Targets with their implied move scaled by a factor learned only from earlier windows.

    For each analysis date, the factor is the least-squares slope (through the origin,
    clipped to 0..1) of realized on implied log moves over targets whose window had
    closed by that date. Dates with fewer than MIN_GROUP such targets are not scored.
    A positive skill means the targets point the right way once their size is corrected;
    at a factor of 0 the shrunk target is the random walk.
    """
    if exit_col not in q:
        return None
    exits, dates = pd.to_datetime(q[exit_col]), pd.to_datetime(q["date"])
    shrunk = pd.Series(np.nan, index=q.index)
    factor = None
    for date in sorted(dates.unique()):
        known = exits <= date
        if known.sum() < MIN_GROUP:
            continue
        x, y = implied[known], realized[known]
        den = float((x * x).sum())
        factor = min(1.0, max(0.0, float((x * y).sum()) / den)) if den else 0.0
        shrunk[dates == date] = factor * implied[dates == date]
    scored = shrunk.notna()
    if not scored.any():
        return None
    walk = realized[scored].abs().mean()
    return {
        "n": int(scored.sum()), "latest_factor": _num(factor),
        "target_skill": _num(1 - (realized - implied)[scored].abs().mean() / walk),
        "shrunk_skill": _num(1 - (realized - shrunk)[scored].abs().mean() / walk),
    }


def target_section(d: pd.DataFrame, end: str) -> dict | None:
    """Price targets against the close at the declared horizon's start (min) or end (max)."""
    close = f"pt_close_{end}"
    if close not in d:
        return None
    q = d.dropna(subset=["price_target", "p0", close])
    if q.empty:
        return None
    implied, realized = np.log(q["price_target"] / q["p0"]), np.log(q[close] / q["p0"])
    # Yahoo's float rounding (581.700012 for 581.70) must not give a flat target a direction.
    moving = implied.abs() > FLAT_MOVE
    model_err, walk_err = np.log(q[close] / q["price_target"]).abs(), realized.abs()
    touch = q.get(f"pt_touch_{end}", pd.Series(dtype=object)).dropna()
    by_model = []
    for model, g in q.groupby("model"):
        if len(g) >= MIN_GROUP:
            ge, gw = model_err[g.index].mean(), walk_err[g.index].mean()
            by_model.append({"model": model, "n": len(g), "skill_vs_random_walk": _num(1 - ge / gw)})
    return {
        "n": len(q), "flat_targets": int((~moving).sum()),
        "median_abs_implied_move": _num(implied.abs().median()),
        "direction_agreement": _num((np.sign(implied[moving]) == np.sign(realized[moving])).mean()),
        "mae_log_target": _num(model_err.mean()), "mae_log_random_walk": _num(walk_err.mean()),
        "skill_vs_random_walk": _num(1 - model_err.mean() / walk_err.mean()),
        "calibration_slope": _num(np.polyfit(implied, realized, 1)[0])
        if len(q) >= 3 and implied.var() > 0 else None,
        "touch_rate": _num(touch.astype(bool).mean()) if len(touch) else None,
        "by_model": sorted(by_model, key=lambda r: -r["n"]),
        "walk_forward_shrunk": shrunk_targets(q, implied, realized, f"pt_exit_{end}"),
    }


def integrity_section(d: pd.DataFrame) -> dict:
    """Is the report's own price right, and does its target point the way its rating does?"""
    priced = d.dropna(subset=["reported_price", "p0"])
    off = (priced["reported_price"] / priced["p0"] - 1).abs()
    t = d.dropna(subset=["price_target", "p0"])
    t = t[t["score"] != 0]
    reference = t["reported_price"].fillna(t["p0"])
    gap = t["price_target"] - reference
    return {
        "reported_price_n": len(priced),
        "reported_price_off_1pct": int((off > 0.01).sum()),
        "reported_price_off_5pct": int((off > 0.05).sum()),
        "directional_with_target": len(t),
        "target_against_rating": int(((gap != 0) & (np.sign(gap) != np.sign(t["score"]))).sum()),
        "target_equals_price": int((gap == 0).sum()),
    }


def build_scorecard(outcomes: pd.DataFrame, as_of: str, source: dict,
                    permutation_reps: int = PERMUTATION_REPS) -> dict:
    d, coverage = decisions(outcomes)
    coverage["matured_alpha"] = {h: int(d[f"alpha{h}"].notna().sum()) if f"alpha{h}" in d else 0
                                 for h in HORIZONS}
    coverage["self_benchmark_excluded"] = int(d["benchmark"].isna().sum())
    coverage["rating_mix"] = d["rating"].value_counts().to_dict()
    coverage["models"] = int(d["model"].nunique())
    coverage["tickers"] = int(d["ticker"].nunique())
    coverage["analysis_dates"] = [d["date"].min(), d["date"].max()] if len(d) else None
    targets = {}
    for end, label in (("min", "declared_start"), ("max", "declared_end")):
        if (section := target_section(d, end)) is not None:
            targets[label] = section
    return {
        "as_of": as_of, "source": source, "coverage": coverage,
        "ratings": {h: s for h in HORIZONS
                    if (s := rating_section(d, h, permutation_reps)) is not None},
        "targets": targets,
        "integrity": integrity_section(d),
    }


# --- markdown ---------------------------------------------------------------------------------


def _pct(v, signed=True):
    return "–" if v is None else (f"{v:+.1%}" if signed else f"{v:.0%}")


def _f(v, digits=3):
    return "–" if v is None else f"{v:.{digits}f}"


def _ci(s):
    return "–" if not s.get("ci95") else f"{_f(s['ci95'][0])} to {_f(s['ci95'][1])}"


def render_markdown(card: dict) -> str:
    cov = card["coverage"]
    lines = [
        f"# Forecast accuracy scorecard (as of {card['as_of']})", "",
        f"Source: forecast ledger `{card['source']['ledger']}`, published site at commit "
        f"`{card['source']['commit'][:10]}`. "
        f"{cov['decisions']} decisions from {cov['models']} models on {cov['tickers']} tickers, "
        f"analysis dates {cov['analysis_dates'][0]} to {cov['analysis_dates'][1]}; "
        f"{cov['retro_excluded']} generated after entry and {cov['reruns_dropped']} superseded "
        f"reruns excluded; {cov['self_benchmark_excluded']} are their own benchmark (no alpha).", "",
        "Rating mix: " + ", ".join(f"{k} {v}" for k, v in cov["rating_mix"].items()), "",
        "## Ratings vs benchmark-relative return", "",
        "| horizon | n | cohorts | rank IC | 95% CI | perm. p | hit rate | hit vs universe "
        "| long–short alpha | momentum63 IC |",
        "|---|---|---|---|---|---|---|---|---|---|",
    ]
    for h, s in card["ratings"].items():
        ic = s["rank_ic"]
        lines.append(
            f"| {h}d | {s['n']} | {s['cohorts']} | {_f(ic['mean'])} | {_ci(ic)} | "
            f"{_f(ic['permutation_p'])} | {_pct(s['hit_rate'], False)} | "
            f"{_pct(s['hit_rate_vs_universe'], False)} | {_pct(s['long_short_alpha']['mean'])} | "
            f"{_f(s['baselines']['momentum63_ic']['mean'])} |")
    for h, s in card["ratings"].items():
        lines += ["", f"### {h} sessions", "",
                  "| rating | n | mean alpha | median alpha |", "|---|---|---|---|"]
        lines += [f"| {r['rating']} | {r['n']} | {_pct(r['mean_alpha'])} | {_pct(r['median_alpha'])} |"
                  for r in s["by_rating"]]
        lines += ["", f"Rating vs trailing 63d alpha (rank corr): {_f(s['rating_vs_momentum63_rank_corr'])}; "
                  f"within-ticker rank corr (timing): {_f(s['within_ticker_rank_corr'])}; "
                  "IC by month: " + ", ".join(f"{k} {_f(v['mean'])} ({v['cohorts']})"
                                              for k, v in s["ic_by_month"].items()), ""]
        models = [r for r in s["by_model"] if r["called"] >= MIN_GROUP]
        if models:
            lines += ["| model | n | non-Hold | hit rate | mean signed alpha | split calls won |",
                      "|---|---|---|---|---|---|"]
            lines += [f"| {r['model']} | {r['n']} | {r['called']} | {_pct(r['hit_rate'], False)} | "
                      f"{_pct(r['mean_signed_alpha'])} | "
                      f"{_pct(r.get('disagreement_win_rate'), False)} ({r.get('disagreements', 0)}) |"
                      for r in models]
        if s["by_confidence"]:
            lines += ["", "Confidence: " + ", ".join(
                f"{c['confidence']} n={c['n']} hit {_pct(c['hit_rate'], False)}"
                for c in s["by_confidence"])]
    lines += ["", "## Price targets", "",
              "| at | n | direction right | target error | random-walk error | skill | slope | touched |",
              "|---|---|---|---|---|---|---|---|"]
    for k, s in card["targets"].items():
        lines.append(
            f"| {k.replace('_', ' ')} | {s['n']} | {_pct(s['direction_agreement'], False)} | "
            f"{_f(s['mae_log_target'])} | {_f(s['mae_log_random_walk'])} | "
            f"{_pct(s['skill_vs_random_walk'])} | {_f(s['calibration_slope'], 2)} | "
            f"{_pct(s['touch_rate'], False)} |")
    lines += ["", "Errors are mean |log(close / target)|; skill = 1 − target error / random-walk "
              "error (above 0 beats \"no change\")."]
    shrunk = {k: s["walk_forward_shrunk"] for k, s in card["targets"].items()
              if s.get("walk_forward_shrunk")}
    if shrunk:
        lines += ["", "| at | n | latest factor | target skill | shrunk-target skill |",
                  "|---|---|---|---|---|"]
        lines += [f"| {k.replace('_', ' ')} | {w['n']} | {_f(w['latest_factor'], 2)} | "
                  f"{_pct(w['target_skill'])} | {_pct(w['shrunk_skill'])} |"
                  for k, w in shrunk.items()]
        lines += ["", "Shrunk target: the target's implied move times a factor fitted only on "
                  "targets whose window had closed by the report's date (slope of realized on "
                  f"implied move through the origin, clipped to 0–1; dates with fewer than {MIN_GROUP} "
                  "such targets are skipped). Both skills are on that same subset. A shrunk-target "
                  "skill above 0 means the targets point the right way once their size is corrected."]
    i = card["integrity"]
    lines += ["", "## Integrity", "",
              f"- Reported price more than 1% from the actual close: {i['reported_price_off_1pct']} of "
              f"{i['reported_price_n']} ({i['reported_price_off_5pct']} more than 5%).",
              f"- Target pointing against the rating: {i['target_against_rating']} of "
              f"{i['directional_with_target']} directional ratings; target equal to price: "
              f"{i['target_equals_price']}.", ""]
    return "\n".join(lines)
