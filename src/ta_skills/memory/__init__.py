"""The memory log: each decision recorded as made, settled against the market, and reflected on.

``log`` keeps the entries, ``settlement`` measures a decision's return once its
holding window has traded, and the memory role's reflection (``ta memory
resolve``) turns that outcome into a lesson the next run of the same ticker reads.
"""

from ta_skills.memory.log import TradingMemoryLog

__all__ = ["TradingMemoryLog"]
