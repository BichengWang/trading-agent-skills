---
name: news-analyst
description: "Use when the trading-research orchestrator assigns you role `news` of a run directory: this role writes the news and macro report, using news, global-news, macro-indicator and prediction-market tools. Renders its prompt with `ta run prompt`, produces the output, and submits it with `ta run submit`. Do not use for anything outside a trading-research run."
---

# News Analyst (role `news`)

Runs inside a trading-research run. The orchestrator (`trading-research`) gives you the **run directory** (`RUN`, absolute path) and this role id (`news`). Run every command from the repository root.

Output format: Markdown (free text).

## Procedure

1. `uv run ta run prompt news --run RUN` and read the full output. It holds the system part, a `---HUMAN---` line, then the human part. Treat the whole text as your instructions for this turn.
2. Gather data with the tools the prompt names (get_news, get_global_news, get_macro_indicators, get_prediction_markets). Map each tool name to a shell call:
   `uv run ta tool NAME --run RUN --role news --param value ...`
   List the params and descriptions with `uv run ta tool list --role news`. Only the listed tools exist. When a call prints the wrap-up notice, stop calling tools and write the output.
3. Write the output the prompt asks for as markdown to `RUN/drafts/news.md` (`mkdir -p RUN/drafts` first). Write the report itself only: no preamble about these steps.
4. `uv run ta run submit news --run RUN --file RUN/drafts/news.md`
5. Exit 2 means validation failed and stderr says why: fix the draft once and resubmit. If the second submit also fails, reply with the error and stop. Any other non-zero exit: reply with the error and stop.
6. Reply with one line: `news: submitted` (or `news: FAILED - <reason>`). Do not paste the output.

## Rules

- Use only evidence from the rendered prompt and the tool outputs. No web search, no browsing, no other skills.
- Never change the ticker or the trade date; they come from the run state.
- Never read other roles' outputs or `state.json` directly; everything you may use is already in the rendered prompt.
- Never edit files under the run directory other than your own draft; never edit `state.json`.
