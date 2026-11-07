"""The run state machine: init, next, prompt rendering, submit/validate, tools, memory and finish.

A run is a directory holding ``state.json`` (the source ``AgentState`` field for
field, plus a ``run`` block), the rendered ``prompts/``, the accepted
``outputs/`` and each analyst's ``tools/<role>.log``. Every read-modify-write of
``state.json`` holds a file lock, because the analysts submit concurrently.

Each ``submit`` reproduces the state write of the source node it replaces, and
``next`` reproduces ``graph/conditional_logic.py`` and the graph's edges, so the
orchestrating agent only ever runs what ``next`` lists.
"""

from __future__ import annotations

import json
import os
import re
from collections.abc import Iterator
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path

from pydantic import BaseModel, ValidationError

from ta_skills import prompts
from ta_skills.config import get_config
from ta_skills.context import (
    TOOLS_SPENT,
    WRAP_UP,
    build_instrument_context,
    get_instrument_context_from_state,
    get_language_instruction,
    get_portfolio_context_from_state,
    opponent_argument_or_opening,
    report_or_absent,
)
from ta_skills.data.date_window import get_current_date, is_historical
from ta_skills.data.files import locked, replace_file
from ta_skills.data.symbols import normalize_symbol, safe_ticker_component
from ta_skills.memory import TradingMemoryLog
from ta_skills.memory.settlement import due_outcomes
from ta_skills.rating import run_rating
from ta_skills.report_fields import MissingPriceTargetError, require_price_target
from ta_skills.reporting import report_settings, write_report_tree
from ta_skills.schemas import (
    PortfolioDecision,
    ResearchPlan,
    SentimentReport,
    TraderProposal,
    render_pm_decision,
    render_research_plan,
    render_sentiment_report,
    render_trader_proposal,
)

# --- roles ----------------------------------------------------------------------------------

ANALYSTS = ("market", "sentiment", "news", "fundamentals")
DEBATE_ROLES = ("bull", "bear")
RISK_ROLES = ("aggressive", "conservative", "neutral")
ROLES = (
    *ANALYSTS,
    "memory",
    *DEBATE_ROLES,
    "research_manager",
    "trader",
    *RISK_ROLES,
    "portfolio_manager",
)

ROLE_SKILL = {
    "market": "market-analyst",
    "sentiment": "sentiment-analyst",
    "news": "news-analyst",
    "fundamentals": "fundamentals-analyst",
    "bull": "bull-researcher",
    "bear": "bear-researcher",
    "research_manager": "research-manager",
    "trader": "trader",
    "aggressive": "aggressive-risk-analyst",
    "conservative": "conservative-risk-analyst",
    "neutral": "neutral-risk-analyst",
    "portfolio_manager": "portfolio-manager",
    "memory": "decision-reflection",
}

# Roles that do not run twice; ``done_roles`` records these. The debate speakers repeat.
ONE_SHOT = (*ANALYSTS, "memory", "research_manager", "trader", "portfolio_manager")

# The tools each analyst is offered (the source ``TOOLS`` tuples); the sentiment analyst has none.
ANALYST_TOOLS: dict[str, tuple[str, ...]] = {
    "market": ("get_stock_data", "get_indicators", "get_verified_market_snapshot"),
    "sentiment": (),
    "news": ("get_news", "get_global_news", "get_macro_indicators", "get_prediction_markets"),
    "fundamentals": (
        "get_fundamentals",
        "get_balance_sheet",
        "get_cashflow",
        "get_income_statement",
        "get_insider_transactions",
    ),
}

REPORT_KEY = {
    "market": "market_report",
    "sentiment": "sentiment_report",
    "news": "news_report",
    "fundamentals": "fundamentals_report",
}

# Structured roles: the pydantic schema their JSON submission is validated against, and
# the prompt part (system / human) that ends with the generated JSON submission section;
# it is always the last part of the rendered prompt.
STRUCTURED: dict[str, tuple[type[BaseModel], str]] = {
    "sentiment": (SentimentReport, "human"),
    "research_manager": (ResearchPlan, "system"),
    "trader": (TraderProposal, "human"),
    "portfolio_manager": (PortfolioDecision, "system"),
}

# The speaker prefix each debate / risk role writes.
SPEAKER_PREFIX = {
    "bull": "Bull Analyst",
    "bear": "Bear Analyst",
    "aggressive": "Aggressive Analyst",
    "conservative": "Conservative Analyst",
    "neutral": "Neutral Analyst",
}

ANALYST_ALIASES = {"social": "sentiment"}

DEBATE_ROUNDS_ENV = "TRADINGAGENTS_MAX_DEBATE_ROUNDS"
RISK_ROUNDS_ENV = "TRADINGAGENTS_MAX_RISK_ROUNDS"

# What a rejected portfolio decision is told to fix (source portfolio_manager.py retry prompt).
PM_CORRECTION = (
    "Correction required: the previous attempt did not produce a valid "
    "Price Target. Return the full decision with **Price Target**: one positive "
    "absolute number. Derive it from the supplied analyst evidence with the first "
    "method the reports support (valuation objective, technical objective, or the "
    "latest close adjusted by the move your rating and horizon imply) and show the "
    "calculation in the Investment Thesis. Weak or stale evidence means Confidence: "
    "Low, not a missing target. Do not copy the current/entry/stop price."
)


class RunError(Exception):
    """A failed command: ``code`` is the process exit status (2 = submission rejected)."""

    def __init__(self, message: str, code: int = 1):
        super().__init__(message)
        self.code = code


# --- state file -------------------------------------------------------------------------------


def state_path(run_dir) -> Path:
    return Path(run_dir) / "state.json"


def load_state(run_dir) -> dict:
    path = state_path(run_dir)
    if not path.is_file():
        raise RunError(f"not a run directory (no state.json): {run_dir}")
    return json.loads(path.read_text(encoding="utf-8"))


def _write_state(run_dir, state: dict) -> None:
    text = json.dumps(state, indent=2, ensure_ascii=False)
    replace_file(state_path(run_dir), lambda tmp: Path(tmp).write_text(text, encoding="utf-8"))


@contextmanager
def update_state(run_dir) -> Iterator[dict]:
    """Locked read-modify-write of ``state.json``; the block's mutations are saved on a clean exit."""
    path = state_path(run_dir)
    if not path.is_file():
        raise RunError(f"not a run directory (no state.json): {run_dir}")
    with locked(path):
        state = json.loads(path.read_text(encoding="utf-8"))
        yield state
        _write_state(run_dir, state)


def _initial_state(
    ticker: str, trade_date: str, asset_type: str, instrument_context: str, portfolio_context: str
) -> dict:
    """The source's initial graph state (``Propagator.create_initial_state``), minus ``messages``."""
    return {
        "company_of_interest": ticker,
        "asset_type": asset_type,
        "instrument_context": instrument_context,
        "trade_date": trade_date,
        "market_report": "",
        "sentiment_report": "",
        "news_report": "",
        "fundamentals_report": "",
        "investment_debate_state": {
            "bull_history": "",
            "bear_history": "",
            "history": "",
            "current_response": "",
            "count": 0,
        },
        "investment_plan": "",
        "trader_investment_plan": "",
        "risk_debate_state": {
            "aggressive_history": "",
            "conservative_history": "",
            "neutral_history": "",
            "history": "",
            "latest_speaker": "",
            "current_aggressive_response": "",
            "current_conservative_response": "",
            "current_neutral_response": "",
            "count": 0,
        },
        "final_trade_decision": "",
        "final_rating": "",
        "past_context": "",
        "memory_note": "",
        "portfolio_context": portfolio_context,
    }


# --- init -------------------------------------------------------------------------------------


def validate_trade_date(trade_date) -> str:
    """The run date as a canonical ``YYYY-MM-DD`` string no later than today."""
    value = str(trade_date)
    try:
        canonical = datetime.strptime(value, "%Y-%m-%d").strftime("%Y-%m-%d") == value
    except ValueError:
        canonical = False
    if not canonical:
        raise RunError(f"trade_date must be a date in YYYY-MM-DD format, got {trade_date!r}")
    if value > get_current_date():
        raise RunError(f"trade_date cannot be in the future: {value}")
    return value


def parse_analysts(selection) -> list[str]:
    """The selected analyst roles, in the order given; ``social`` is the sentiment analyst's old key."""
    if isinstance(selection, str):
        selection = selection.split(",")
    selected: list[str] = []
    for raw in selection:
        key = raw.strip().lower()
        if not key:
            continue
        key = ANALYST_ALIASES.get(key, key)
        if key not in ANALYSTS:
            raise RunError(f"unknown analyst key: {raw.strip()}")
        if key not in selected:
            selected.append(key)
    if not selected:
        raise RunError("at least one analyst must be selected")
    return selected


def _rounds(explicit: int | None, env_var: str, depth: int | None, configured: int) -> int:
    """One round count: an explicit flag, else the environment's value, else ``--depth``, else the config's.

    The environment beats the depth, as in the source CLI (#977): a value the
    user exported is not silently replaced by the research-depth answer.
    """
    if explicit is not None:
        value = explicit
    elif depth is not None and not os.environ.get(env_var):
        value = depth
    else:
        value = configured
    if value < 1:
        raise RunError(f"round counts must be at least 1, got {value}")
    return value


def init_run(
    ticker: str,
    trade_date: str,
    analysts="market,sentiment,news,fundamentals",
    depth: int | None = None,
    debate_rounds: int | None = None,
    risk_rounds: int | None = None,
    language: str | None = None,
    portfolio: str | None = None,
    asset_type: str = "auto",
    model_label: str | None = None,
    provider: str = "unspecified",
    deep_model: str = "unspecified",
    quick_model: str = "unspecified",
) -> Path:
    """Create a run directory and its initial state; return the run directory."""
    from ta_skills.identity import detect_asset_type, resolve_instrument_identity
    from ta_skills.portfolio import load_portfolio

    trade_date = validate_trade_date(trade_date)
    raw = (ticker or "").strip()
    if not raw:
        raise RunError("ticker must not be empty")
    try:
        symbol = normalize_symbol(raw)
        safe = safe_ticker_component(symbol)
    except ValueError as exc:
        raise RunError(f"invalid ticker {ticker!r}: {exc}") from exc
    if asset_type not in ("auto", "stock", "crypto"):
        raise RunError(f"asset type must be auto, stock or crypto, got {asset_type!r}")
    kind = detect_asset_type(symbol) if asset_type == "auto" else asset_type
    selected = parse_analysts(analysts)
    config = get_config()
    debate = _rounds(debate_rounds, DEBATE_ROUNDS_ENV, depth, config["max_debate_rounds"])
    risk = _rounds(risk_rounds, RISK_ROUNDS_ENV, depth, config["max_risk_discuss_rounds"])

    portfolio_context = ""
    if portfolio:
        try:
            portfolio_context = load_portfolio(portfolio).render(symbol)
        except ValueError as exc:
            raise RunError(str(exc)) from exc

    identity = resolve_instrument_identity(symbol, kind, trade_date)
    state = _initial_state(
        symbol,
        trade_date,
        kind,
        build_instrument_context(symbol, kind, identity, trade_date),
        portfolio_context,
    )
    state["run"] = {
        "selected_analysts": selected,
        "max_debate_rounds": debate,
        "max_risk_discuss_rounds": risk,
        "max_tool_rounds": int(config["max_tool_rounds"]),
        "output_language": language or config.get("output_language") or "English",
        "model_label": model_label or deep_model,
        "provider": provider,
        "deep_model": deep_model,
        "quick_model": quick_model,
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "done_roles": [],
        "tool_calls": {},
        "submit_failures": {},
        "memory_due": [],
        "memory_failed": [],
    }

    parent = Path(config["runs_dir"]).expanduser() / safe
    parent.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y-%m-%d_%H%M%S")
    run_dir, n = parent / stamp, 1
    while True:
        try:
            run_dir.mkdir()
            break
        except FileExistsError:
            n += 1
            run_dir = parent / f"{stamp}_{n}"
    for sub in ("prompts", "outputs", "tools", "drafts"):
        (run_dir / sub).mkdir()
    _write_state(run_dir, state)
    return run_dir


# --- routing ----------------------------------------------------------------------------------


def next_debate_speaker(state: dict) -> str:
    """``bull``, ``bear`` or ``research_manager``: ``ConditionalLogic.should_continue_debate``.

    The first speaker is always the bull (there is no conditional entry edge). The
    count budget is checked before the speaker-prefix routing.
    """
    debate = state["investment_debate_state"]
    if debate["count"] == 0:
        return "bull"
    if debate["count"] >= 2 * state["run"]["max_debate_rounds"]:
        return "research_manager"
    if debate["current_response"].startswith("Bull"):
        return "bear"
    return "bull"


def next_risk_speaker(state: dict) -> str:
    """``aggressive``, ``conservative``, ``neutral`` or ``portfolio_manager``: ``should_continue_risk_analysis``.

    The risk debate always opens with the aggressive analyst (the Trader's only edge).
    """
    risk = state["risk_debate_state"]
    if risk["count"] == 0 and not risk["latest_speaker"]:
        return "aggressive"
    if risk["count"] >= 3 * state["run"]["max_risk_discuss_rounds"]:
        return "portfolio_manager"
    if risk["latest_speaker"].startswith("Aggressive"):
        return "conservative"
    if risk["latest_speaker"].startswith("Conservative"):
        return "neutral"
    return "aggressive"


def next_phase(state: dict) -> tuple[str, list[str]]:
    """The phase the run is in and the roles to run now."""
    run = state["run"]
    done = set(run["done_roles"])
    pending = [r for r in (*run["selected_analysts"], "memory") if r not in done]
    if pending:
        return "analysts", pending
    if "research_manager" not in done:
        speaker = next_debate_speaker(state)
        if speaker == "research_manager":
            return "research_manager", ["research_manager"]
        return "research_debate", [speaker]
    if "trader" not in done:
        return "trader", ["trader"]
    if "portfolio_manager" not in done:
        speaker = next_risk_speaker(state)
        if speaker == "portfolio_manager":
            return "portfolio_manager", ["portfolio_manager"]
        return "risk_debate", [speaker]
    return "done", []


def turn_of(state: dict, role: str) -> int | None:
    """The 1-based speech number of a repeating role about to speak; ``None`` for one-shot roles."""
    if role in DEBATE_ROLES:
        return state["investment_debate_state"]["count"] // 2 + 1
    if role in RISK_ROLES:
        return state["risk_debate_state"]["count"] // 3 + 1
    return None


def _require_runnable(state: dict, role: str) -> None:
    if role not in ROLES:
        raise RunError(f"unknown role {role!r}; roles: {', '.join(ROLES)}")
    phase, roles = next_phase(state)
    if role not in roles:
        raise RunError(
            f"role {role!r} is not runnable now (phase {phase!r}, runnable: {roles or 'none'})"
        )


# --- prompts ----------------------------------------------------------------------------------


def sentiment_inputs(run_dir, state: dict) -> dict:
    """The sentiment analyst's pre-fetched sources, fetched once per run and cached beside the state."""
    path = Path(run_dir) / "sentiment_inputs.json"
    if path.is_file():
        return json.loads(path.read_text(encoding="utf-8"))
    from ta_skills.sentiment_sources import build_sentiment_inputs

    inputs = build_sentiment_inputs(state["company_of_interest"], state["trade_date"])
    text = json.dumps(inputs, indent=2, ensure_ascii=False)
    replace_file(path, lambda tmp: Path(tmp).write_text(text, encoding="utf-8"))
    return inputs


def _reports(state: dict) -> dict[str, str]:
    """The four reports, each replaced by an explicit marker when its analyst produced none."""
    return {
        "market_research_report": report_or_absent(state["market_report"], "market"),
        "sentiment_report": report_or_absent(state["sentiment_report"], "sentiment"),
        "news_report": report_or_absent(state["news_report"], "news"),
        "fundamentals_report": report_or_absent(state["fundamentals_report"], "fundamentals"),
    }


def prompt_values(role: str, state: dict, run_dir=None) -> dict[str, str]:
    """The ``{{placeholder}}`` values of ``role``'s templates, derived from the state."""
    run = state["run"]
    language = get_language_instruction(run["output_language"])
    instrument = get_instrument_context_from_state(state)

    if role in ("market", "news", "fundamentals"):
        spent = run["tool_calls"].get(role, 0) >= run["max_tool_rounds"]
        values = {
            "tool_names": TOOLS_SPENT if spent else ", ".join(ANALYST_TOOLS[role]),
            "current_date": state["trade_date"],
            "instrument_context": instrument,
            "language_instruction": language,
            "company_of_interest": state["company_of_interest"],
        }
        if role == "market":
            from ta_skills.data.indicator_registry import render_prompt_section

            values["indicator_menu"] = render_prompt_section()
        if role == "news":
            values["asset_label"] = (
                "company" if state.get("asset_type", "stock") == "stock" else "asset"
            )
        return values

    if role == "sentiment":
        inputs = sentiment_inputs(run_dir, state)
        return {
            "current_date": state["trade_date"],
            "instrument_context": instrument,
            "language_instruction": language,
            "company_of_interest": state["company_of_interest"],
            "ticker": state["company_of_interest"],
            "start_date": inputs["start_date"],
            "end_date": inputs["end_date"],
            "news_block": inputs["news_block"],
            "stocktwits_block": inputs["stocktwits_block"],
            "reddit_block": inputs["reddit_block"],
            "subreddits": ", ".join(f"r/{sub}" for sub in inputs["subreddits"]),
            "character": inputs["character"],
        }

    if role in DEBATE_ROLES:
        debate = state["investment_debate_state"]
        stock = state.get("asset_type", "stock") == "stock"
        opponent = "bear analyst" if role == "bull" else "bull analyst"
        return {
            "target_label": "stock" if stock else "asset",
            "fundamentals_label": (
                "Company fundamentals report"
                if stock
                else "Asset fundamentals report (may be unavailable for crypto)"
            ),
            "instrument_context": instrument,
            **_reports(state),
            "language_instruction": language,
            "history": debate.get("history", "") or "(the debate has not started yet)",
            "current_response": opponent_argument_or_opening(
                debate.get("current_response", ""), opponent
            ),
        }

    if role == "research_manager":
        return {
            "instrument_context": instrument,
            "history": state["investment_debate_state"].get("history", ""),
            "language_instruction": language,
        }

    if role == "trader":
        market_report = (state["market_report"] or "").strip()
        if market_report:
            grounding = (
                "Ground concrete price levels (entry, stop-loss, position sizing) in the technical "
                "market report's price structure -- current price, support/resistance, ATR, and "
                "volatility -- and use the research plan for direction and strategy. "
            )
            report_section = f"Technical Market Report:\n{market_report}\n\n"
        else:
            grounding = ""
            report_section = ""
        return {
            "grounding": grounding,
            "language_instruction": language,
            "company_name": state["company_of_interest"],
            "instrument_context": instrument,
            "report_section": report_section,
            "portfolio_context": get_portfolio_context_from_state(state),
            "investment_plan": state["investment_plan"],
        }

    if role in RISK_ROLES:
        risk = state["risk_debate_state"]
        others = [r for r in RISK_ROLES if r != role]
        values = {
            "trader_decision": state["trader_investment_plan"],
            "instrument_context": instrument,
            "portfolio_context": get_portfolio_context_from_state(state),
            **_reports(state),
            "language_instruction": language,
            "history": risk.get("history", "") or "(none yet)",
        }
        for other in others:
            key = f"current_{other}_response"
            values[key] = opponent_argument_or_opening(risk.get(key, ""), f"{other} analyst")
        return values

    if role == "portfolio_manager":
        past_context = state.get("past_context", "")
        return {
            "instrument_context": instrument,
            "portfolio_context": get_portfolio_context_from_state(state),
            "research_plan": state["investment_plan"],
            "trader_plan": state["trader_investment_plan"],
            "market_report": state.get("market_report") or "No technical market report supplied.",
            "fundamentals_report": state.get("fundamentals_report")
            or "No fundamentals report supplied.",
            "lessons_line": (
                f"- Lessons from prior decisions and outcomes:\n{past_context}\n"
                if past_context
                else ""
            ),
            "history": state["risk_debate_state"]["history"],
            "language_instruction": language,
        }
    raise RunError(f"role {role!r} has no prompt")


def render_role_prompt(role: str, state: dict, run_dir=None) -> str:
    """The full prompt for ``role``: system part, ``---HUMAN---``, human part (when it has one)."""
    system, human = prompts.render_skill(ROLE_SKILL[role], prompt_values(role, state, run_dir))
    if role in STRUCTURED:
        schema, part = STRUCTURED[role]
        section = prompts.json_output_section(schema)
        if part == "human" and human is not None:
            human = f"{human}\n\n{section}"
        else:
            system = f"{system}\n\n{section}"
    return prompts.join_parts(system, human)


def role_label(role: str, turn: int | None) -> str:
    return role if turn is None else f"{role}_{turn}"


def prompt_for(run_dir, role: str) -> str:
    """Render ``role``'s prompt at its current turn, save it under ``prompts/`` and return it."""
    state = load_state(run_dir)
    if role == "memory":
        raise RunError(
            "the memory role's prompts come from `ta memory due`, one per pending decision"
        )
    _require_runnable(state, role)
    text = render_role_prompt(role, state, run_dir)
    path = Path(run_dir) / "prompts" / f"{role_label(role, turn_of(state, role))}.md"
    path.write_text(text, encoding="utf-8")
    return text


# --- submit -----------------------------------------------------------------------------------


def _read_submission(path) -> str:
    try:
        return Path(path).read_text(encoding="utf-8")
    except OSError as exc:
        raise RunError(f"cannot read submission file {path}: {exc}") from exc


def _reject(state: dict, role: str, message: str) -> RunError:
    """A rejected submission (exit 2); the first rejection is the one correction attempt."""
    failures = state["run"].setdefault("submit_failures", {})
    failures[role] = failures.get(role, 0) + 1
    if failures[role] > 1:
        message += (
            "\nThis was the correction attempt and it failed too: do not resubmit; "
            "report the failure to the orchestrator."
        )
    else:
        message += "\nFix the submission and resubmit once."
    return RunError(message, code=2)


def _validation_message(exc: ValidationError) -> str:
    lines = []
    for err in exc.errors():
        where = ".".join(str(p) for p in err["loc"]) or "(object)"
        lines.append(f"- {where}: {err['msg']}")
    return "The submission does not match the schema:\n" + "\n".join(lines)


def _parse_structured(role: str, text: str) -> tuple[BaseModel, str]:
    """The validated schema instance of a JSON submission and its rendered markdown."""
    schema = STRUCTURED[role][0]
    try:
        data = json.loads(text)
    except json.JSONDecodeError as exc:
        raise ValueError(
            f"The submission must be one JSON object, with no code fences or surrounding text: {exc}"
        ) from exc
    if not isinstance(data, dict):
        raise ValueError("The submission must be one JSON object.")
    try:
        obj = schema.model_validate(data)
    except ValidationError as exc:
        raise ValueError(_validation_message(exc)) from exc
    render = {
        "sentiment": render_sentiment_report,
        "research_manager": render_research_plan,
        "trader": render_trader_proposal,
        "portfolio_manager": render_pm_decision,
    }[role]
    rendered = render(obj)
    if role == "portfolio_manager":
        try:
            require_price_target(rendered)
        except MissingPriceTargetError as exc:
            raise ValueError(f"{exc}\n{PM_CORRECTION}") from exc
    return obj, rendered


def _apply_debate(state: dict, role: str, text: str) -> None:
    """Bull / Bear: the source node's returned ``investment_debate_state``."""
    debate = state["investment_debate_state"]
    argument = f"{SPEAKER_PREFIX[role]}: {text}"
    state["investment_debate_state"] = {
        "history": debate.get("history", "") + "\n" + argument,
        "bull_history": debate.get("bull_history", "")
        + ("\n" + argument if role == "bull" else ""),
        "bear_history": debate.get("bear_history", "")
        + ("\n" + argument if role == "bear" else ""),
        "current_response": argument,
        "count": debate["count"] + 1,
    }


def _apply_risk(state: dict, role: str, text: str) -> None:
    """Aggressive / Conservative / Neutral: the source node's returned ``risk_debate_state``."""
    risk = state["risk_debate_state"]
    argument = f"{SPEAKER_PREFIX[role]}: {text}"
    new = {
        "history": risk.get("history", "") + "\n" + argument,
        "aggressive_history": risk.get("aggressive_history", ""),
        "conservative_history": risk.get("conservative_history", ""),
        "neutral_history": risk.get("neutral_history", ""),
        "latest_speaker": role.capitalize(),
        "current_aggressive_response": risk.get("current_aggressive_response", ""),
        "current_conservative_response": risk.get("current_conservative_response", ""),
        "current_neutral_response": risk.get("current_neutral_response", ""),
        "count": risk["count"] + 1,
    }
    new[f"{role}_history"] += "\n" + argument
    new[f"current_{role}_response"] = argument
    state["risk_debate_state"] = new


def _apply_research_manager(state: dict, plan: str) -> None:
    """The Research Manager overwrites ``current_response`` with its plan; counts are unchanged."""
    debate = state["investment_debate_state"]
    state["investment_debate_state"] = {
        "history": debate.get("history", ""),
        "bear_history": debate.get("bear_history", ""),
        "bull_history": debate.get("bull_history", ""),
        "current_response": plan,
        "count": debate["count"],
    }
    state["investment_plan"] = plan


def _apply_portfolio_manager(state: dict, decision: str, rating: str) -> None:
    """The Portfolio Manager marks ``latest_speaker`` as ``Judge``; its typed rating is the run's."""
    risk = state["risk_debate_state"]
    state["risk_debate_state"] = {**risk, "latest_speaker": "Judge"}
    state["final_trade_decision"] = decision
    state["final_rating"] = rating


def submit_role(run_dir, role: str, file=None) -> dict:
    """Validate and apply ``role``'s submission, mark it done, and return what was accepted."""
    if role == "memory":
        return submit_memory(run_dir)
    if file is None:
        raise RunError(f"submit {role} needs --file PATH")
    text = _read_submission(file)
    with update_state(run_dir) as state:
        _require_runnable(state, role)
        turn = turn_of(state, role)
        failed: RunError | None = None
        rendered = text.strip()
        obj = None
        try:
            if role in STRUCTURED:
                obj, rendered = _parse_structured(role, text)
            elif not rendered:
                raise ValueError(f"{role}: expected report text, received an empty submission")
        except ValueError as exc:
            failed = _reject(state, role, str(exc))
        if failed is not None:
            _persist_failure_count(run_dir, state)
            raise failed

        out = Path(run_dir) / "outputs"
        label = role_label(role, turn)
        if role in STRUCTURED:
            (out / f"{label}.json").write_text(text, encoding="utf-8")
        (out / f"{label}.md").write_text(rendered, encoding="utf-8")

        if role in ANALYSTS:
            state[REPORT_KEY[role]] = rendered
        elif role in DEBATE_ROLES:
            _apply_debate(state, role, rendered)
        elif role in RISK_ROLES:
            _apply_risk(state, role, rendered)
        elif role == "research_manager":
            _apply_research_manager(state, rendered)
        elif role == "trader":
            state["trader_investment_plan"] = rendered
        elif role == "portfolio_manager":
            _apply_portfolio_manager(state, rendered, obj.rating.value)
        state["run"]["submit_failures"].pop(role, None)
        if role in ONE_SHOT:
            state["run"]["done_roles"].append(role)
        phase, roles = next_phase(state)
    return {"role": role, "accepted": True, "phase": phase, "roles": roles}


def _persist_failure_count(run_dir, state: dict) -> None:
    """Keep the rejection counter: ``update_state`` saves only on a clean exit, a rejection raises."""
    _write_state(run_dir, state)


# --- memory -----------------------------------------------------------------------------------


def _memory_log() -> TradingMemoryLog:
    return TradingMemoryLog(get_config())


def memory_as_of(trade_date: str) -> str | None:
    """Point-in-time cutoff for past-context lessons (#1251): the trade date of a historical run."""
    return str(trade_date) if is_historical(trade_date) else None


def reflection_prompt(item: dict) -> str:
    """The rendered reflection prompt for one settled decision (system, ``---HUMAN---``, human)."""
    system, human = prompts.render_skill(
        ROLE_SKILL["memory"],
        {
            "holding_days": str(item["holding_days"]),
            "raw_return": f"{item['raw']:+.1%}",
            "alpha_return": f"{item['alpha']:+.1%}",
            "benchmark_name": item["benchmark"],
            "final_decision": item["decision"],
        },
    )
    return prompts.join_parts(system, human)


def memory_due(run_dir) -> list[dict]:
    """Settle price outcomes of every pending decision whose holding window has traded.

    Returns one item per decision awaiting its reflection, with the rendered
    reflection prompt. Price failures are recorded for the run's memory note.
    """
    load_state(run_dir)
    config = get_config()
    failed: list[list[str]] = []
    items: list[dict] = []
    error = ""
    try:
        settlement = due_outcomes(_memory_log(), config)
        failed = [list(f) for f in settlement.failed]
        for due in settlement.due:
            items.append(
                {
                    "ticker": due.ticker,
                    "date": due.date,
                    "raw": due.raw,
                    "alpha": due.alpha,
                    "holding_days": due.holding_days,
                    "benchmark": due.benchmark,
                    "resolution_date": due.resolution_date,
                    "decision": due.decision,
                }
            )
    except Exception as exc:  # noqa: BLE001 — a memory failure never ends the run
        error = f"Past decisions could not be settled this run ({type(exc).__name__}); they stay pending."
    with update_state(run_dir) as state:
        state["run"]["memory_due"] = items
        state["run"]["memory_failed"] = failed
        state["run"]["memory_error"] = error
    return [
        {
            "ticker": i["ticker"],
            "date": i["date"],
            "raw": i["raw"],
            "alpha": i["alpha"],
            "holding_days": i["holding_days"],
            "benchmark": i["benchmark"],
            "prompt": reflection_prompt(i),
        }
        for i in items
    ]


def memory_resolve(run_dir, ticker: str, date: str, file) -> bool:
    """Write the reflection for one settled decision; ``False`` when it was already resolved."""
    reflection = _read_submission(file).strip()
    if not reflection:
        raise RunError("the reflection is empty", code=2)
    state = load_state(run_dir)
    item = next(
        (
            i
            for i in state["run"].get("memory_due", [])
            if i["ticker"] == ticker and i["date"] == date
        ),
        None,
    )
    if item is None:
        raise RunError(
            f"{ticker} {date} is not in this run's due list; run `ta memory due --run DIR` first"
        )
    return _memory_log().update_with_outcome(
        ticker,
        date,
        item["raw"],
        item["alpha"],
        item["holding_days"],
        reflection,
        resolution_date=item["resolution_date"],
    )


def submit_memory(run_dir) -> dict:
    """Compute ``past_context`` and ``memory_note`` and mark the memory step done (``_memory_step``)."""
    state = load_state(run_dir)
    run = state["run"]
    note = run.get("memory_error", "")
    past_context = ""
    try:
        log = _memory_log()
        unsettled = len(run.get("memory_failed", []))
        if run.get("memory_due"):
            pending = {(e["ticker"], e["date"]) for e in log.get_pending_entries()}
            unsettled += sum(1 for i in run["memory_due"] if (i["ticker"], i["date"]) in pending)
        if unsettled and not note:
            note = f"{unsettled} past decision(s) could not be settled this run and stay pending."
        past_context = log.get_past_context(
            state["company_of_interest"], as_of=memory_as_of(state["trade_date"])
        )
    except Exception as exc:  # noqa: BLE001 — a memory failure never ends the run
        past_context = ""
        note = note or f"The memory log could not be read this run ({type(exc).__name__})."
    with update_state(run_dir) as state:
        if "memory" in state["run"]["done_roles"]:
            raise RunError("memory is already done")
        state["past_context"] = past_context
        state["memory_note"] = note
        state["run"]["done_roles"].append("memory")
        phase, roles = next_phase(state)
    return {"role": "memory", "accepted": True, "phase": phase, "roles": roles}


# --- tools ------------------------------------------------------------------------------------


def run_tool(run_dir, name: str, role: str, params: dict[str, str]) -> str:
    """Run an analyst's tool for the run's ticker and date, counting the call against its budget.

    Once the role's ``max_tool_rounds`` are spent the tool is not run and the
    wrap-up instruction is returned instead (the source's ``WRAP_UP`` turn).
    """
    from ta_skills.tools import TOOLS, call_tool

    if role not in ANALYSTS:
        raise RunError(f"{role!r} is not an analyst role; analysts: {', '.join(ANALYSTS)}")
    offered = ANALYST_TOOLS[role]
    if name not in offered:
        raise RunError(
            f"{name!r} is not a tool of the {role} analyst; its tools: {', '.join(offered) or 'none'}"
        )
    if name not in TOOLS:
        raise RunError(f"unknown tool {name!r}")
    with update_state(run_dir) as state:
        calls = state["run"]["tool_calls"]
        if calls.get(role, 0) >= state["run"]["max_tool_rounds"]:
            return WRAP_UP
        calls[role] = calls.get(role, 0) + 1
        ticker, trade_date = state["company_of_interest"], state["trade_date"]
    try:
        output = call_tool(name, ticker, trade_date, **params)
    except ValueError as exc:
        raise RunError(str(exc), code=2) from exc
    log = Path(run_dir) / "tools" / f"{role}.log"
    with locked(log), open(log, "a", encoding="utf-8") as fh:
        args = ", ".join(f"{k}={v}" for k, v in params.items())
        fh.write(f"[{datetime.now().isoformat(timespec='seconds')}] {name}({args})\n{output}\n\n")
    return output


# --- finish -----------------------------------------------------------------------------------


def _slug(value: str) -> str:
    """A model label as a path component (``/ : .`` become ``-``, like the report guard)."""
    return re.sub(r"[/:.]", "-", value).strip() or "model"


def _state_log(final_state: dict, settings: dict) -> dict:
    """The source ``_log_state`` entry."""
    debate = final_state["investment_debate_state"]
    risk = final_state["risk_debate_state"]
    return {
        "company_of_interest": final_state["company_of_interest"],
        "trade_date": final_state["trade_date"],
        "market_report": final_state["market_report"],
        "sentiment_report": final_state["sentiment_report"],
        "news_report": final_state["news_report"],
        "fundamentals_report": final_state["fundamentals_report"],
        "investment_debate_state": {
            "bull_history": debate["bull_history"],
            "bear_history": debate["bear_history"],
            "history": debate["history"],
            "current_response": debate["current_response"],
        },
        "trader_investment_plan": final_state["trader_investment_plan"],
        "risk_debate_state": {
            "aggressive_history": risk["aggressive_history"],
            "conservative_history": risk["conservative_history"],
            "neutral_history": risk["neutral_history"],
            "history": risk["history"],
        },
        "investment_plan": final_state["investment_plan"],
        "final_trade_decision": final_state["final_trade_decision"],
        "final_rating": run_rating(final_state),
        "run_settings": settings,
    }


def finish_run(run_dir, reports_dir=None, html: bool = True) -> dict:
    """Write the report tree and the state log, record the decision in the memory log."""
    state = load_state(run_dir)
    phase, roles = next_phase(state)
    if phase != "done":
        raise RunError(f"the run is not finished (phase {phase!r}, runnable: {roles})")
    config = get_config()
    run = state["run"]
    final_state = {k: v for k, v in state.items() if k != "run"}
    ticker, trade_date = final_state["company_of_interest"], final_state["trade_date"]
    settings = report_settings(config, run)

    base = Path(reports_dir or config["reports_dir"]).expanduser()
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    name = f"{trade_date.replace('-', '')}_{_slug(run['model_label'])}_{stamp}"
    report_dir = base / safe_ticker_component(ticker) / name
    try:
        complete = write_report_tree(
            final_state, ticker, report_dir, settings=settings, html=html, normalize_headings=True
        )
    except MissingPriceTargetError as exc:
        raise RunError(str(exc)) from exc

    log_dir = (
        Path(config["results_dir"]).expanduser()
        / safe_ticker_component(ticker)
        / "TradingAgentsStrategy_logs"
    )
    log_dir.mkdir(parents=True, exist_ok=True)
    entry = _state_log(final_state, settings)
    # Reports can be in any language and this file is read by a person.
    (log_dir / f"full_states_log_{trade_date}.json").write_text(
        json.dumps(entry, indent=4, ensure_ascii=False), encoding="utf-8"
    )

    rating = run_rating(final_state)
    decision = final_state.get("final_trade_decision")
    if decision:
        _memory_log().store_decision(
            ticker=ticker,
            trade_date=trade_date,
            final_trade_decision=decision,
            rating=rating,
        )
    return {"report_dir": str(report_dir), "complete_report": str(complete), "rating": rating}
