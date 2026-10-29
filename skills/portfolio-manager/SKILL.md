---
name: portfolio-manager
description: "Use when the trading-research orchestrator assigns you role `portfolio_manager` of a run directory: this role makes the final structured rating and price-target decision after the risk debate. Renders its prompt with `ta run prompt`, produces the output, and submits it with `ta run submit`. Do not use for anything outside a trading-research run."
---

# Portfolio Manager (role `portfolio_manager`)

Runs inside a trading-research run. The orchestrator (`trading-research`) gives you the **run directory** (`RUN`, absolute path) and this role id (`portfolio_manager`). Run every command from the repository root.

Output format: JSON matching the schema in the rendered prompt. `price_target` must be a number; submit exits 2 without one.

## Procedure

1. `uv run ta run prompt portfolio_manager --run RUN` and read the full output. It holds the system part, a `---HUMAN---` line, then the human part. Treat the whole text as your instructions for this turn.
2. Write the output as one JSON object to `RUN/drafts/portfolio_manager.json` (`mkdir -p RUN/drafts` first), with exactly the fields and enum values listed in the prompt's JSON output section. No code fences, no extra keys.
3. `uv run ta run submit portfolio_manager --run RUN --file RUN/drafts/portfolio_manager.json`
4. Exit 2 means validation failed and stderr says why: fix the draft once and resubmit. If the second submit also fails, reply with the error and stop. Any other non-zero exit: reply with the error and stop.
5. Reply with one line: `portfolio_manager: submitted` (or `portfolio_manager: FAILED - <reason>`). Do not paste the output.

## Rules

- Use only evidence from the rendered prompt. No web search, no browsing, no other skills.
- Never change the ticker or the trade date; they come from the run state.
- Never read other roles' outputs or `state.json` directly; everything you may use is already in the rendered prompt.
- Never edit files under the run directory other than your own draft; never edit `state.json`.
