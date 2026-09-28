"""Source capability wiring for datacmp."""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import MagicMock

from reccmp.compare.db import EntityDb
from reccmp.compare.source_capability import (
    require_source_index,
    resolve_source_index_path,
)
from reccmp.compare.variables import VariableComparator
from reccmp.cvdump.types import CvdumpTypesParser
from reccmp.source import (
    keyed,
    SourceAbi,
    SourceClass,
    SourceField,
    SourceIndex,
    SourceVariable,
)
from reccmp.types import ImageId

from tests.raw_image import RawImage
from tests.test_variable_comparator import create_matched_variable, get_match


def test_require_source_index_scopes_and_enriches_datacmp_path(
    tmp_path: Path,
):
    """Compare session path: load index → target view → VariableComparator."""
    db = EntityDb()
    types = CvdumpTypesParser()
    document = SourceIndex(
        declarations={},
        markers=(),
        abi=SourceAbi(
            target_triple="i386-pc-windows-msvc",
            pointer_width=32,
            ms_abi=True,
        ),
        target_abis={
            "GAME": SourceAbi(
                target_triple="i386-pc-windows-msvc",
                pointer_width=32,
                ms_abi=True,
            )
        },
        classes={
            **keyed(
                (
                    SourceClass(
                        semantic_id="record:Foo",
                        qualified_name="Foo",
                        bases=(),
                        fields=(
                            SourceField(
                                name="bar",
                                type="int",
                                pointer_depth=0,
                                storage_kind="scalar",
                                source_file="a.h",
                                line=2,
                                offset=0,
                                size=4,
                            ),
                            SourceField(
                                name="ptr",
                                type="int *",
                                storage_kind="pointer",
                                source_file="a.h",
                                line=3,
                                offset=4,
                                size=4,
                                pointer_depth=1,
                            ),
                        ),
                        virtual_declarations=(),
                        source_file="a.h",
                        line=1,
                        end_line=4,
                        size=8,
                        alignment=4,
                        layout_trusted=True,
                    ),
                ),
                "GAME",
            ),
            **keyed(
                (
                    SourceClass(
                        semantic_id="record:Foo",
                        qualified_name="Foo",
                        bases=(),
                        fields=(),
                        virtual_declarations=(),
                        source_file="other.h",
                        line=1,
                        end_line=2,
                        size=4,
                        layout_trusted=True,
                    ),
                ),
                "OTHER",
            ),
        },
        variables={
            **keyed(
                (
                    SourceVariable(
                        semantic_id="gFoo",
                        qualified_name="gFoo",
                        type="Foo",
                        storage_kind="embedded_record",
                        record_semantic_id="record:Foo",
                        linkage="external",
                        storage_class="none",
                        definition_kind="definition",
                        source_file="a.cpp",
                        line=1,
                        end_line=1,
                    ),
                ),
                "GAME",
            )
        },
    ).to_dict()
    index_path = tmp_path / "source-index.json"
    index_path.write_text(json.dumps(document), encoding="utf-8")

    target = MagicMock()
    target.target_id = "GAME"
    target.recompiled_path = tmp_path / "game.exe"
    target.source_paths = ()
    target.source_index = None

    assert resolve_source_index_path(target, explicit=index_path) == index_path
    scoped = require_source_index(target, explicit=index_path)
    assert scoped is not None
    source_class = scoped.class_named("Foo")
    assert source_class is not None and source_class.size == 8
    assert scoped.abi is not None
    assert scoped.abi.pointer_width == 32

    create_matched_variable(db, 0, size=8)
    with db.batch() as batch:
        batch.set(ImageId.RECOMP, 0, name="gFoo")

    orig = RawImage.from_memory(b"\x01\x00\x00\x00\xaa\xbb\xcc\xdd")
    recomp = RawImage.from_memory(b"\x01\x00\x00\x00\xaa\xbb\xcc\xde")
    comparator = VariableComparator(db, types, orig, recomp, source_index=scoped)
    result = comparator.compare_variable(get_match(db, 0))
    assert result is not None
    assert result.raw_only is False
    names = [item.name for item in result.compared]
    assert names == ["bar", "ptr"]
