---
name: decision-reflection
description: "Use when the trading-research orchestrator assigns you role `memory` of a run directory: settles past trading decisions whose holding window has elapsed by writing a reflection for each, then computes the run's past-context memory note. Do not use outside a trading-research run."
---

# Decision Reflection (role `memory`)

Runs at the start of a trading-research run, in parallel with the analysts. The orchestrator gives you the **run directory** (`RUN`, absolute path) and the role id `memory`. Run every command from the repository root.

Output format: markdown reflection text per settled decision; no JSON.

## Procedure

1. `uv run ta memory due --run RUN` prints a JSON list. Each item has `ticker`, `date`, `raw`, `alpha`, `holding_days`, `benchmark` and `prompt` (the rendered reflection prompt: system part, a `---HUMAN---` line, then the human part). An empty list means nothing is due: go to step 4.
2. For each item, follow its `prompt` as your instructions and write the reflection it asks for as markdown to `RUN/drafts/memory_<ticker>_<date>.md` (`mkdir -p RUN/drafts` first). Base it only on that prompt.
3. `uv run ta memory resolve --run RUN --ticker T --date D --file RUN/drafts/memory_<ticker>_<date>.md` for each item (`T` and `D` are the item's `ticker` and `date`). Exit 2 or any other failure: fix the draft once and retry; if it still fails, skip that item and mention it in your reply.
4. `uv run ta run submit memory --run RUN` (no `--file`). It computes the past context and memory note and marks `memory` done.
5. Reply with one line: `memory: submitted (N reflections)` (or `memory: FAILED - <reason>`).

## Rules

- Use only evidence from the due items' prompts. No web search, no other skills.
- Never change a ticker or date; copy them from the item.
- Never edit `state.json` or the memory log by hand; only `ta memory resolve` writes it.
