---
name: conservative-risk-analyst
description: "Use when the trading-research orchestrator assigns you role `conservative` of a run directory: this role speaks one turn of the conservative side of the risk debate. Renders its prompt with `ta run prompt`, produces the output, and submits it with `ta run submit`. Do not use for anything outside a trading-research run."
---

# Conservative Risk Analyst (role `conservative`)

Runs inside a trading-research run. The orchestrator (`trading-research`) gives you the **run directory** (`RUN`, absolute path) and this role id (`conservative`). Run every command from the repository root.

Output format: Markdown (free text).

## Procedure

1. `uv run ta run prompt conservative --run RUN` and read the full output. It holds the system part, a `---HUMAN---` line, then the human part. Treat the whole text as your instructions for this turn.
2. Write the output the prompt asks for as markdown to `RUN/drafts/conservative.md` (`mkdir -p RUN/drafts` first). Write the report itself only: no preamble about these steps.
3. `uv run ta run submit conservative --run RUN --file RUN/drafts/conservative.md`
4. Exit 2 means validation failed and stderr says why: fix the draft once and resubmit. If the second submit also fails, reply with the error and stop. Any other non-zero exit: reply with the error and stop.
5. Reply with one line: `conservative: submitted` (or `conservative: FAILED - <reason>`). Do not paste the output.

## Rules

- Use only evidence from the rendered prompt. No web search, no browsing, no other skills.
- Never change the ticker or the trade date; they come from the run state.
- Never read other roles' outputs or `state.json` directly; everything you may use is already in the rendered prompt.
- Never edit files under the run directory other than your own draft; never edit `state.json`.
