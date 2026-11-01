"""The forecast ledger: every published portfolio decision as structured fields.

The record of past forecasts is the TradingAgents site's deploy branch
(``gh-pages``): one ``<TICKER>/<YYYYMMDD>_<model>_<YYYYMMDD>_<HHMMSS>/complete_report/index.html``
per run. A run path is immutable, so each decision is read once and appended to
``forecasts.jsonl``; a later pull only adds runs it has not seen, and a decision
already scored can never change under the evaluation.
"""

from __future__ import annotations

import hashlib
import html
import json
import re
import subprocess
from dataclasses import asdict, dataclass
from pathlib import Path

from ta_skills.data.files import locked
from ta_skills.rating import RATINGS_5_TIER

REPORT_PATH = re.compile(
    r"^([A-Z0-9][A-Z0-9.^-]{0,31})/(\d{8})_([A-Za-z0-9][A-Za-z0-9._-]{0,127})_(\d{8})_(\d{6})"
    r"/complete_report/index\.html$"
)
# Buy +2 .. Sell -2: the position a rating asks for, relative to the benchmark.
RATING_SCORE = {r: 2 - i for i, r in enumerate(RATINGS_5_TIER)}

_FIELD_RE = re.compile(r"<p>\s*<strong>([^<]+?)</strong>(.*?)</p>", re.DOTALL)
_LABELS = {
    "Rating": "rating", "Current Price": "current_price", "Price Target": "price_target",
    "Confidence": "confidence", "Time Horizon": "time_horizon",
}
_RATING_RE = re.compile(r"\W*(buy|overweight|hold|underweight|sell)\b", re.IGNORECASE)
_NUMBER_RE = re.compile(r"\d[\d,]*(?:\.\d+)?")
_TRADING_DAYS = {"day": 1, "week": 5, "month": 21, "quarter": 63, "year": 252}
_HORIZON_RE = re.compile(
    r"(\d+(?:\.\d+)?)(?:\s*(?:-|–|—|to)\s*(\d+(?:\.\d+)?))?[\s-]*(?:trading[\s-]+)?"
    r"(day|week|month|quarter|year)s?\b",
    re.IGNORECASE,
)


@dataclass(frozen=True)
class Forecast:
    """One published decision. Prices are as written in the report, in its quote currency."""

    path: str
    ticker: str
    date: str               # analysis date, YYYY-MM-DD
    model: str
    generated_at: str       # run start, YYYY-MM-DDTHH:MM:SS, the runner's local clock
    rating: str | None      # one of RATINGS_5_TIER, or None when the report states none
    reported_price: float | None
    price_target: float | None
    confidence: str | None
    time_horizon: str | None
    horizon_min_td: int | None
    horizon_max_td: int | None
    section_sha256: str     # of the final decision section as published
    source_commit: str


def final_section(page: str) -> str | None:
    """The Portfolio Manager decision section of a published report, or None when it has none.

    Sliced out before parsing: earlier debate sections carry Rating labels of their own.
    """
    anchor = page.find('id="v-portfolio-manager-decision"')
    if anchor < 0:
        return None
    start, end = page.rfind("<h", 0, anchor), page.find("</article>", anchor)
    return page[start:end] if start >= 0 and end >= 0 else None


def final_fields(section: str) -> dict[str, str | None]:
    """The labelled fields (``<p><strong>Label</strong>: value</p>``), first occurrence of each."""
    fields: dict[str, str | None] = dict.fromkeys(_LABELS.values())
    for label, rest in _FIELD_RE.findall(section):
        key = _LABELS.get(html.unescape(label).strip().rstrip(":").strip())
        if key and fields[key] is None:
            text = html.unescape(re.sub(r"<[^>]+>", "", rest))
            fields[key] = re.sub(r"\s+", " ", text).strip().lstrip(":").strip() or None
    return fields


def parse_rating(value: str | None) -> str | None:
    match = _RATING_RE.match(value or "")
    return match[1].capitalize() if match else None


def parse_price(value: str | None) -> float | None:
    """The first number in a price field (``$389.28``, ``221.71 (2026-10-21)``), if positive."""
    match = _NUMBER_RE.search(value or "")
    price = float(match[0].replace(",", "")) if match else None
    return price if price else None


def parse_horizon(value: str | None) -> tuple[int, int] | None:
    """A stated horizon as (min, max) trading days: ``3-6 months`` -> (63, 126)."""
    match = _HORIZON_RE.search(value or "")
    if not match:
        return None
    unit = _TRADING_DAYS[match[3].lower()]
    low = float(match[1])
    high = float(match[2]) if match[2] else low
    return round(low * unit), round(high * unit)


def forecast_from_page(path: str, page: str, source_commit: str) -> Forecast | None:
    """The forecast a published report page makes; None for a path that is not a report run."""
    match = REPORT_PATH.match(path)
    if not match:
        return None
    ticker, day, model, gen_day, gen_time = match.groups()
    section = final_section(page) or ""
    fields = final_fields(section)
    horizon = parse_horizon(fields["time_horizon"])
    confidence = (fields["confidence"] or "").split(" ")[0].strip(".,;").lower()
    return Forecast(
        path=path, ticker=ticker, date=f"{day[:4]}-{day[4:6]}-{day[6:]}", model=model,
        generated_at=(f"{gen_day[:4]}-{gen_day[4:6]}-{gen_day[6:]}T"
                      f"{gen_time[:2]}:{gen_time[2:4]}:{gen_time[4:]}"),
        rating=parse_rating(fields["rating"]),
        reported_price=parse_price(fields["current_price"]),
        price_target=parse_price(fields["price_target"]),
        confidence=confidence if confidence in {"low", "medium", "high"} else None,
        time_horizon=fields["time_horizon"],
        horizon_min_td=horizon[0] if horizon else None,
        horizon_max_td=horizon[1] if horizon else None,
        section_sha256=hashlib.sha256(section.encode("utf-8")).hexdigest(),
        source_commit=source_commit,
    )


def load_ledger(path: Path) -> list[dict]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def _git(repo: Path, *args: str, stdin: bytes | None = None) -> bytes:
    return subprocess.run(["git", "-C", str(repo), *args], input=stdin, capture_output=True,
                          check=True).stdout


def _read_blobs(repo: Path, ref: str, paths: list[str], chunk: int = 100):
    """Yield ``(path, content)`` at ``ref``, one ``git cat-file --batch`` per chunk of paths.

    Chunked so a full history (2,000+ pages of ~350 KB) never sits in memory at once.
    """
    for start in range(0, len(paths), chunk):
        batch = paths[start:start + chunk]
        out = _git(repo, "cat-file", "--batch",
                   stdin="".join(f"{ref}:{p}\n" for p in batch).encode("utf-8"))
        pos = 0
        for path in batch:
            header_end = out.index(b"\n", pos)
            size = int(out[pos:header_end].split()[2])
            yield path, out[header_end + 1:header_end + 1 + size].decode("utf-8", errors="replace")
            pos = header_end + 1 + size + 1


def pull(ledger_path: Path, repo: Path, remote: str = "origin", branch: str = "gh-pages",
         fetch: bool = True) -> dict:
    """Append every published run not yet in the ledger, read from ``remote/branch`` of ``repo``.

    Only the remote-tracking ref is updated: the checkout's working tree and
    branches are left alone.
    """
    repo = Path(repo).expanduser()
    ref = f"{remote}/{branch}"
    if fetch:
        _git(repo, "fetch", "-q", remote, branch)
    commit = _git(repo, "rev-parse", ref).decode().strip()
    listed = [p for p in _git(repo, "ls-tree", "-r", "--name-only", ref).decode().splitlines()
              if REPORT_PATH.match(p)]
    ledger_path.parent.mkdir(parents=True, exist_ok=True)
    with locked(ledger_path):
        known = {row["path"] for row in load_ledger(ledger_path)}
        new = sorted(p for p in listed if p not in known)
        forecasts = [forecast_from_page(p, page, commit) for p, page in _read_blobs(repo, ref, new)]
        with open(ledger_path, "a", encoding="utf-8") as fh:
            for forecast in forecasts:
                fh.write(json.dumps(asdict(forecast), ensure_ascii=False) + "\n")
    return {
        "source": f"{repo}@{ref}", "commit": commit, "published": len(listed),
        "added": len(forecasts), "ledger_total": len(known) + len(forecasts),
        "added_without_rating": sum(f.rating is None for f in forecasts),
        "ledger": str(ledger_path),
    }
