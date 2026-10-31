# AGENTS.md

trading-agent-skills: TradingAgents' research pipeline as Agent Skills plus a deterministic `ta` CLI. LLM reasoning lives in the agent runtime (skills); data, state, validation, reports and memory live in Python (`src/ta_skills`).

## Rules

- Run everything from the repo root through uv: `uv run ta ...`, `uv run pytest`, `uv run ruff check .`.
- Never hand-edit run state (`state.json`, `outputs/`, `prompts/` under a run directory) or the memory log. Change them only via `ta run ...` / `ta memory ...`.
- Prompts live in `skills/*/prompt*.md` (verbatim TradingAgents text, `{{name}}` placeholders) and are rendered by `ta run prompt ROLE --run DIR`. Skills never embed prompt text and agents never read `prompt*.md` directly. Changing a prompt changes research behavior: keep it faithful to upstream unless asked.
- `skills/<name>/SKILL.md` stays short (procedure and rules). `.claude/skills` and `.agents/skills` are symlinks to `skills/`; `.claude/agents/<skill>.md` are the Claude Code subagent definitions (`sonnet` for quick roles, `opus` for deep roles). Keep names in sync with the role table in `skills/trading-research/SKILL.md`.
- No LLM SDKs or LangChain/LangGraph in Python. Network tests are marked `integration` and excluded by default.
- Python 3.11+, ruff line length 100.

## Generated report Markdown (`docs/`)

- `docs/<TICKER>/<YYYYMMDD>_<MODEL>_<YYYYMMDD>_<HHMMSS>/` holds report runs in the TradingAgents layout (`1_analysts/` .. `5_portfolio/`, `complete_report.md`/`.html`); `ta run finish` writes them there via the `trading-research` skill.
- Do not recursively scan or read report Markdown under `docs/**/*.md` into context. Treat `docs/index.md`, `docs/<TICKER>/index.md` and run folders as generated output; never hand-edit them.
- Refresh the site with `uv run python scripts/report_workflow.py --analysis-date YYYYMMDD --allow-incomplete` (regenerates the homepage and ticker hubs, validates the summary rows, compiles `_site/` with `mkdocs build --strict`). Needs the `docs` extra: `uv pip install -e '.[dev,docs]'`.
- Report Markdown, the derived indices and `_site/` are gitignored, as in TradingAgents: they stay in the local checkout. Verify docs with bounded commands (`grep -c`, the workflow validators), not by dumping reports.

## Before a PR

`uv run ruff check .` and `uv run pytest -q` must pass (CI runs both on Python 3.11-3.13). Add the `auto-merge` label to a PR that should merge itself once checks pass.
