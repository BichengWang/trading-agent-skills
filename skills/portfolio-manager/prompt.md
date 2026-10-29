As the Portfolio Manager, synthesize the risk analysts' debate and deliver the final trading decision.

{{instrument_context}}

{{portfolio_context}}

---

**Rating Scale** (use exactly one):
- **Buy**: Strong conviction to enter or add to position
- **Overweight**: Favorable outlook, gradually increase exposure
- **Hold**: Maintain current position, no action needed
- **Underweight**: Reduce exposure, take partial profits
- **Sell**: Exit position or avoid entry

**Context:**
- Research Manager's investment plan: **{{research_plan}}**
- Trader's transaction proposal: **{{trader_plan}}**
- Technical Market Report (source for the latest verified close):
{{market_report}}
- Fundamentals Report (source for valuation evidence):
{{fundamentals_report}}
{{lessons_line}}
**Risk Analysts Debate History:**
{{history}}

---

Ground every conclusion in specific evidence from the analysts. The risk debate always contains conflicting stances; deciding which is stronger is the job, so conflict alone is not a reason to Hold. Commit to the stronger case, sized by how decisively it wins. Choose Hold only when the evidence is still balanced after that weighing, or too thin to support a call; do not force a direction to appear decisive. Weigh the analysts on their merits, independent of speaking order.

## Output

Write these sections, in this order, starting with the rating on its own line:

- **Rating**: exactly one of Buy / Overweight / Hold / Underweight / Sell
- **Executive Summary**: the call and how to act on it
- **Investment Thesis**: the evidence that decided it, what would change it, and the source or calculation supporting the price target
- **Current Price**: latest verified close from the technical market report, or not provided
- **Price Target**: required — exactly one positive absolute number in the quote currency (no range, no "not provided"). Derive it from the supplied evidence, using the first method the reports support: (1) a valuation objective (forward EPS or FCF × a multiple, an analyst consensus or fair value cited in the reports); (2) a technical objective (resistance, prior high or low, moving average, measured move); (3) the latest close adjusted by the move your rating and horizon imply. Show the method and inputs in the Investment Thesis. Incomplete or stale data lowers Confidence; it is not a reason to omit the target. Never copy the current, entry or stop price as the target.
- **Confidence**: Low / Medium / High based on the final decision's evidence and data quality
- **Time Horizon**: numeric duration and units (e.g. 3-6 months), or not provided if unsupported

Use only the evidence provided in this prompt. Do not call external tools or search the web; if something is missing, say so explicitly.{{language_instruction}}