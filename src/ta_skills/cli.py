"""The ``ta`` command: data tools, run state, prompt rendering, submissions, memory and reports.

Every command prints machine-readable output on stdout; errors go to stderr with
a non-zero exit status (2 = a submission was rejected and may be corrected).
"""

from __future__ import annotations

import argparse
import json
import sys

from ta_skills import __version__, run


def _print_json(value) -> None:
    print(json.dumps(value, ensure_ascii=False, indent=2))


# --- ta run -----------------------------------------------------------------------------------


def _cmd_init(args) -> None:
    run_dir = run.init_run(
        ticker=args.ticker,
        trade_date=args.date,
        analysts=args.analysts,
        depth=args.depth,
        debate_rounds=args.debate_rounds,
        risk_rounds=args.risk_rounds,
        language=args.language,
        portfolio=args.portfolio,
        asset_type=args.asset_type,
        model_label=args.model_label,
        provider=args.provider,
        deep_model=args.deep_model,
        quick_model=args.quick_model,
    )
    print(run_dir)


def _cmd_next(args) -> None:
    phase, roles = run.next_phase(run.load_state(args.run))
    _print_json({"phase": phase, "roles": roles})


def _cmd_prompt(args) -> None:
    if args.role == "memory":
        # The memory role has no single prompt: one per settled decision, rendered by `memory due`.
        _print_json(run.memory_due(args.run))
        return
    sys.stdout.write(run.prompt_for(args.run, args.role) + "\n")


def _cmd_submit(args) -> None:
    _print_json(run.submit_role(args.run, args.role, args.file))


def _cmd_finish(args) -> None:
    _print_json(run.finish_run(args.run, reports_dir=args.reports_dir, html=not args.no_html))


# --- ta memory --------------------------------------------------------------------------------


def _cmd_memory_due(args) -> None:
    _print_json(run.memory_due(args.run))


def _cmd_memory_resolve(args) -> None:
    resolved = run.memory_resolve(args.run, args.ticker, args.date, args.file)
    _print_json({"ticker": args.ticker, "date": args.date, "resolved": resolved})


# --- ta tool ----------------------------------------------------------------------------------


def _tool_signature(spec, required) -> str:
    parts = []
    for p in spec.params:
        default = "" if p.default is required else f" = {p.default!r}"
        parts.append(f"--{p.name} <{p.type.__name__}>{default}")
    return " ".join(parts)


def _list_tools(role: str | None) -> None:
    from ta_skills.tools import REQUIRED, TOOLS

    if role is not None and role not in run.ANALYSTS:
        raise run.RunError(f"unknown analyst role {role!r}; analysts: {', '.join(run.ANALYSTS)}")
    names = list(TOOLS) if role is None else list(run.ANALYST_TOOLS[role])
    if not names:
        print(f"The {role} analyst has no tools: its data is pre-fetched into its prompt.")
        return
    blocks = []
    for name in names:
        spec = TOOLS[name]
        lines = [f"{name} {_tool_signature(spec, REQUIRED)}".rstrip()]
        for p in spec.params:
            need = "required" if p.default is REQUIRED else f"default {p.default!r}"
            lines.append(f"  --{p.name} ({p.type.__name__}, {need}): {p.help}")
        lines.append("")
        lines.extend(f"  {line}" if line else "" for line in spec.description.splitlines())
        blocks.append("\n".join(lines))
    print("\n\n".join(blocks))


def _tool_params(spec, extras: list[str]) -> dict[str, str]:
    """``--name value`` pairs of a tool call, checked against the tool's own parameters."""
    allowed = {p.name for p in spec.params}
    params: dict[str, str] = {}
    i = 0
    while i < len(extras):
        arg = extras[i]
        if not arg.startswith("--"):
            raise run.RunError(f"unexpected argument {arg!r}; tool parameters are --name value pairs")
        name, eq, value = arg[2:].partition("=")
        if not eq:
            i += 1
            if i >= len(extras):
                raise run.RunError(f"--{name} needs a value")
            value = extras[i]
        if name.replace("-", "_") not in allowed:
            raise run.RunError(
                f"{spec.name} has no parameter --{name}; it takes: "
                f"{', '.join('--' + n for n in sorted(allowed)) or 'none'}"
            )
        params[name.replace("-", "_")] = value
        i += 1
    return params


def _cmd_tool(args, extras: list[str]) -> None:
    if args.name == "list":
        if extras:
            raise run.RunError(f"unexpected arguments: {' '.join(extras)}")
        _list_tools(args.role)
        return
    if not args.run or not args.role:
        raise run.RunError("ta tool NAME needs --run DIR and --role ROLE")
    from ta_skills.tools import TOOLS

    spec = TOOLS.get(args.name)
    if spec is None:
        raise run.RunError(f"unknown tool {args.name!r}; see `ta tool list`")
    sys.stdout.write(run.run_tool(args.run, args.name, args.role, _tool_params(spec, extras)) + "\n")


# --- parser -----------------------------------------------------------------------------------


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="ta",
        allow_abbrev=False,
        description="trading-agent-skills: deterministic data tools, run state, prompts and reports "
        "for the TradingAgents role skills.",
    )
    parser.add_argument("--version", action="version", version=f"ta {__version__}")
    top = parser.add_subparsers(dest="group", required=True, metavar="{run,tool,memory}")

    # ta run ...
    run_p = top.add_parser("run", help="create and drive a research run")
    run_sub = run_p.add_subparsers(dest="command", required=True)

    p = run_sub.add_parser("init", help="create a run directory; prints its path")
    p.add_argument("--ticker", required=True)
    p.add_argument("--date", required=True, help="analysis date, YYYY-MM-DD, not in the future")
    p.add_argument("--analysts", default="market,sentiment,news,fundamentals",
                   help="comma-separated: market,sentiment,news,fundamentals (social = sentiment)")
    p.add_argument("--depth", type=int, help="rounds for both debates; an exported "
                   f"{run.DEBATE_ROUNDS_ENV} / {run.RISK_ROUNDS_ENV} wins over it")
    p.add_argument("--debate-rounds", type=int, help="bull/bear rounds (explicit; beats the environment)")
    p.add_argument("--risk-rounds", type=int, help="risk-analyst rounds (explicit; beats the environment)")
    p.add_argument("--language", help="output language (default: config / TRADINGAGENTS_OUTPUT_LANGUAGE)")
    p.add_argument("--portfolio", metavar="FILE", help="portfolio JSON: cash, currency, positions")
    p.add_argument("--asset-type", choices=("auto", "stock", "crypto"), default="auto")
    p.add_argument("--model-label", help="label for the report folder (default: the deep model)")
    p.add_argument("--provider", default="unspecified")
    p.add_argument("--deep-model", default="unspecified")
    p.add_argument("--quick-model", default="unspecified")
    p.set_defaults(func=_cmd_init)

    p = run_sub.add_parser("next", help="print the phase and the roles to run now, as JSON")
    p.add_argument("--run", required=True, metavar="DIR")
    p.set_defaults(func=_cmd_next)

    p = run_sub.add_parser("prompt", help="print a role's fully rendered prompt")
    p.add_argument("role", choices=run.ROLES)
    p.add_argument("--run", required=True, metavar="DIR")
    p.set_defaults(func=_cmd_prompt)

    p = run_sub.add_parser("submit", help="validate and apply a role's output")
    p.add_argument("role", choices=run.ROLES)
    p.add_argument("--run", required=True, metavar="DIR")
    p.add_argument("--file", metavar="PATH", help="markdown (free-text roles) or JSON (structured "
                   "roles); omitted for the memory role")
    p.set_defaults(func=_cmd_submit)

    p = run_sub.add_parser("finish", help="write the report tree and record the decision")
    p.add_argument("--run", required=True, metavar="DIR")
    p.add_argument("--reports-dir", metavar="D")
    p.add_argument("--no-html", action="store_true", help="skip complete_report.html")
    p.set_defaults(func=_cmd_finish)

    # ta tool ...
    p = top.add_parser(
        "tool", help="run an analyst's data tool (`ta tool list` shows them)",
        description="ta tool NAME --run DIR --role ROLE [--<param> value ...]   run a tool\n"
        "ta tool list [--role ROLE]                                 list tools",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("name", metavar="NAME|list")
    p.add_argument("--run", metavar="DIR")
    p.add_argument("--role", metavar="ROLE")
    p.set_defaults(func=_cmd_tool, takes_extras=True)

    # ta memory ...
    mem = top.add_parser("memory", help="settle and reflect on past decisions")
    mem_sub = mem.add_subparsers(dest="command", required=True)
    p = mem_sub.add_parser("due", help="price outcomes of settled decisions, with reflection prompts")
    p.add_argument("--run", required=True, metavar="DIR")
    p.set_defaults(func=_cmd_memory_due)
    p = mem_sub.add_parser("resolve", help="write a reflection into the memory log")
    p.add_argument("--run", required=True, metavar="DIR")
    p.add_argument("--ticker", required=True)
    p.add_argument("--date", required=True)
    p.add_argument("--file", required=True, metavar="PATH")
    p.set_defaults(func=_cmd_memory_resolve)
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args, extras = parser.parse_known_args(argv)
    if extras and not getattr(args, "takes_extras", False):
        parser.error(f"unrecognized arguments: {' '.join(extras)}")
    try:
        if getattr(args, "takes_extras", False):
            args.func(args, extras)
        else:
            args.func(args)
    except run.RunError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return exc.code
    except Exception as exc:  # noqa: BLE001 — a command failure is a message and a status, not a traceback
        from ta_skills.prompts import PromptError

        print(f"error: {exc}", file=sys.stderr)
        return 3 if isinstance(exc, PromptError) else 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
