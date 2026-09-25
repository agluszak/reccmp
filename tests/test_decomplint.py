"""decomplint reads markers from the Clang source index."""

from pathlib import Path

import pytest

from reccmp.parser.error import AlertCode
from reccmp.source import SourceIndex, SourceIndexError
from reccmp.source.index import source_digest
from reccmp.tools.decomplint import DecomplintTarget, lint_all_targets


def _block(source_file: str, line: int, *comments: str, candidates=()) -> dict:
    return {
        "source_file": source_file,
        "comments": [
            {"text": text, "line": line + i, "column": 1, "offset": 10 * (line + i)}
            for i, text in enumerate(comments)
        ],
        "anchor": {
            "line": line + len(comments),
            "column": 1,
            "candidates": list(candidates),
            "string": None,
        },
    }


def _function(name: str, line: int) -> dict:
    return {
        "kind": "function",
        "semantic_id": f"?{name}@@YAXXZ",
        "qualified_name": name,
        "is_definition": True,
        "line": line,
        "end_line": line,
    }


@pytest.fixture(name="project")
def fixture_project(tmp_path: Path) -> tuple[Path, Path]:
    source = tmp_path / "src" / "game.cpp"
    source.parent.mkdir()
    source.write_text(
        "// FUNCTION: GAME 0x2000\n"
        "void Two() {}\n"
        "// FUNCTION: GAME 0x1000\n"
        "void One() {}\n"
        "#if 0\n"
        "// FUNCTION: GAME 0x3000\n"
        "void Three() {}\n"
        "#endif\n"
        "// GLOBAL: GAME 0x4000\n"
        "void NotAVariable() {}\n",
        encoding="utf-8",
    )
    blocks = [
        _block(
            "src/game.cpp",
            1,
            "// FUNCTION: GAME 0x2000",
            candidates=[_function("Two", 2)],
        ),
        _block(
            "src/game.cpp",
            3,
            "// FUNCTION: GAME 0x1000",
            candidates=[_function("One", 4)],
        ),
        _block(
            "src/game.cpp",
            9,
            "// GLOBAL: GAME 0x4000",
            candidates=[_function("NotAVariable", 10)],
        ),
    ]
    document = SourceIndex(
        declarations={},
        classes={},
        markers=(),
        source_digests={"src/game.cpp": source_digest(source)},
    ).to_dict()
    document["marker_blocks"] = blocks
    index = tmp_path / "source-index.json"
    SourceIndex.from_dict(document).write(index)
    return source, index


def _alerts(source: Path, index: Path | None) -> list[tuple[AlertCode, int]]:
    target = DecomplintTarget((source,), "GAME", "utf-8", source_index=index)
    return sorted(
        ((alert.code, alert.line_number) for alert in lint_all_targets((target,))),
        key=lambda item: (item[1], item[0].value),
    )


def test_markers_and_alerts_come_from_the_index(project: tuple[Path, Path]):
    source, index = project
    assert _alerts(source, index) == [
        (AlertCode.FUNCTION_OUT_OF_ORDER, 4),
        (AlertCode.MARKER_NOT_COMPILED, 6),
        (AlertCode.GLOBAL_NOT_VARIABLE, 9),
    ]


def test_changed_sources_make_the_index_stale(project: tuple[Path, Path]):
    source, index = project
    source.write_text(source.read_text() + "\n", encoding="utf-8")
    assert (AlertCode.STALE_SOURCE_INDEX, -1) in _alerts(source, index)


def test_linting_needs_an_index(project: tuple[Path, Path]):
    source, _ = project
    with pytest.raises(SourceIndexError, match="source index"):
        _alerts(source, None)
