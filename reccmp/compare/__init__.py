"""Comparison package; load the top-level comparator only when requested.

Analysis modules import comparison primitives without needing ``core`` (which
also imports those analyses) during package initialization.
"""

from typing import TYPE_CHECKING

__all__ = ["Compare"]

if TYPE_CHECKING:
    from .core import Compare


def __getattr__(name: str):
    if name == "Compare":
        # pylint: disable-next=import-outside-toplevel
        from .core import Compare

        return Compare
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
