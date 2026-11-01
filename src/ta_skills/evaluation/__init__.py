"""Forecast evaluation: score published portfolio decisions against the prices that followed.

``ledger`` reads every published decision into an append-only forecast ledger,
``outcomes`` prices each one over fixed and declared horizons once they have
traded, and ``metrics`` scores ratings, price targets and confidence against
baselines into a scorecard. Deterministic Python: no model calls.
"""
