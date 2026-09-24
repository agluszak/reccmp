"""Attributes FunctionComparator provides to the mixins it is built from.

The mixins keep fork-specific comparison logic out of functions.py; this
base only declares what they may use, for type checking.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from reccmp.compare.asm.parse import ParseAsm
    from reccmp.compare.call_facts import CallFacts
    from reccmp.compare.db import EntityDb
    from reccmp.compare.inlines import HelperCatalogEntry
    from reccmp.compare.lines import LinesDb
    from reccmp.cvdump.analysis import CvdumpNode
    from reccmp.cvdump.types import CvdumpTypesParser
    from reccmp.formats import Image
    from reccmp.source import SourceIndex


class ComparatorState:
    # pylint: disable=too-few-public-methods,too-many-instance-attributes
    db: EntityDb
    lines_db: LinesDb
    orig_bin: Image
    recomp_bin: Image
    types: CvdumpTypesParser
    func_nodes: dict[int, CvdumpNode]
    equivalence_groups: dict[int, int]
    source_index: SourceIndex | None
    is_32bit: bool
    orig_sanitize: ParseAsm
    recomp_sanitize: ParseAsm
    _call_facts_cache: dict[str, CallFacts | None] | None
    _fp_cache: dict
    _helper_catalog: list[HelperCatalogEntry] | None
    _helper_by_orig: dict[int, HelperCatalogEntry | None]
    _helper_identity_index: dict[str, int] | None
    _helper_identity_ambiguous: set[str] | None

    # Provided by FunctionComparator, which comes earlier in the MRO;
    # declared here only for type checking.
    def rebuild_lookups(self) -> None: ...
