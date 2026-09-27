"""Presentation rows for a decoded function.

Data and jump tables belong to ``FunctionImage``, not its instruction stream.
This module interleaves them only when constructing a human-facing diff.
"""

from __future__ import annotations

from collections.abc import Hashable
from dataclasses import dataclass

from .ir import FunctionImage, instruction_match_key, instruction_semantic_key


@dataclass(frozen=True)
class RenderedRow:
    address: int | None
    display: str
    match_key: Hashable
    semantic_key: Hashable
    code_index: int | None = None


def render_function_rows(image: FunctionImage) -> tuple[RenderedRow, ...]:
    """Interleave instructions and embedded data without feeding it to analysis."""
    pending: list[tuple[int, int, RenderedRow]] = []
    for index, row in enumerate(image.instructions):
        assert row.address is not None
        pending.append(
            (
                row.address,
                1,
                RenderedRow(
                    row.address,
                    row.display,
                    instruction_match_key(row),
                    instruction_semantic_key(row),
                    index,
                ),
            )
        )
    for table in image.jump_tables:
        pending.append(
            (
                table.address,
                0,
                RenderedRow(None, "Jump table:", ("jump_header",), ("jump_header",)),
            )
        )
        for entry_addr, target in table.entries:
            offset = target - image.start_addr
            key = ("case", offset)
            pending.append(
                (
                    entry_addr,
                    1,
                    RenderedRow(
                        entry_addr,
                        f"start + 0x{offset:x}",
                        key,
                        key,
                    ),
                )
            )
    for region in image.data_regions:
        pending.append(
            (
                region.address,
                0,
                RenderedRow(None, "Data table:", ("data_header",), ("data_header",)),
            )
        )
        for offset, value in enumerate(region.data):
            key = ("byte", value)
            pending.append(
                (
                    region.address + offset,
                    1,
                    RenderedRow(region.address + offset, hex(value), key, key),
                )
            )
    return tuple(
        row for _address, _priority, row in sorted(pending, key=lambda item: item[:2])
    )
