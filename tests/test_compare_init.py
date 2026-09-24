"""Testing constructors of the Compare core"""

from pathlib import Path
from unittest.mock import patch
import pytest
from reccmp.compare import Compare
from reccmp.project.detect import RecCmpTarget, GhidraConfig, ReportConfig
from reccmp.cvdump.parser import CvdumpParser
from reccmp.source import SourceIndex, SourceIndexError
from reccmp.source.index import source_digest
from .raw_image import RawImage


@pytest.fixture(name="source_dir")
def fixture_source_dir(tmp_path_factory) -> Path:
    """Create a basic source root with files in two directories."""
    src_dir = tmp_path_factory.mktemp("src")
    (src_dir / "hello.cpp").write_text("")
    (src_dir / "hello.hpp").write_text("")
    (src_dir / "test").mkdir()
    (src_dir / "test" / "game.cpp").write_text("")
    (src_dir / "test" / "game.hpp").write_text("")

    return src_dir


def _index_of(source_dir: Path) -> SourceIndex:
    """An empty but current source index of every file under the root."""
    return SourceIndex(
        declarations=(),
        classes=(),
        markers=(),
        source_digests={
            path.relative_to(source_dir).as_posix(): source_digest(path)
            for path in source_dir.rglob("*.?pp")
        },
    )


def _target(source_paths: tuple[Path, ...]) -> RecCmpTarget:
    return RecCmpTarget(
        target_id="TEST",
        filename="TEST.exe",
        sha256="",
        encoding="utf-8",
        source_paths=source_paths,
        original_path=Path("TEST.exe"),
        recompiled_path=Path("build/TEST.exe"),
        recompiled_pdb=Path("build/TEST.pdb"),
        ghidra_config=GhidraConfig(),
        report_config=ReportConfig(),
    )


def _patched():
    return (
        patch(
            "reccmp.compare.target_analysis.detect_image",
            new=lambda **_: RawImage.from_memory(),
        ),
        patch(
            "reccmp.compare.target_analysis.Cvdump.run",
            new=lambda _: CvdumpParser(),
        ),
    )


def test_nested_paths(source_dir: Path):
    """Compare core will eliminate duplicate code file paths
    if the list of source paths contains any that are nested."""
    target = _target((source_dir, source_dir / "test"))

    # Patch detect_image: don't open the file, just return a RawImage
    # Patch Cvdump.run: don't subprocess.run, just return an empty Cvdump result
    with (
        patch(
            "reccmp.compare.target_analysis.detect_image",
            new=lambda **_: RawImage.from_memory(),
        ),
        patch(
            "reccmp.compare.target_analysis.Cvdump.run",
            new=lambda _: CvdumpParser(),
        ),
    ):
        c = Compare.from_target(
            target, use_cache=False, source_index=_index_of(source_dir)
        )

        # If path walks were just combined, we would have 6 files.
        assert len(c.code_files) == 4

        # Verify that paths are sorted
        assert [f.path.name for f in c.code_files] == [
            "hello.cpp",
            "hello.hpp",
            "game.cpp",
            "game.hpp",
        ]


def test_markers_need_a_source_index(source_dir: Path, monkeypatch):
    monkeypatch.delenv("RECCMP_SOURCE_INDEX", raising=False)
    first, second = _patched()
    with first, second, pytest.raises(SourceIndexError, match="no source index"):
        Compare.from_target(_target((source_dir,)), use_cache=False)


def test_a_stale_source_index_is_refused(source_dir: Path):
    index = _index_of(source_dir)
    (source_dir / "test" / "game.cpp").write_text("// FUNCTION: TEST 0x1000\n")
    first, second = _patched()
    with first, second, pytest.raises(SourceIndexError, match="game.cpp"):
        Compare.from_target(_target((source_dir,)), use_cache=False, source_index=index)
