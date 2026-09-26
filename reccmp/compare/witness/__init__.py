"""Concrete differential execution: find inputs that refute equivalence.

Requires the optional ``unicorn`` dependency (``pip install reccmp[witness]``).
"""

from .machine import SideMachine
from .replay import ReplayResult, replay
from .search import SearchResult, Translator, find_witness

__all__ = [
    "ReplayResult",
    "SearchResult",
    "SideMachine",
    "Translator",
    "find_witness",
    "replay",
]
