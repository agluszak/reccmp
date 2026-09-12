"""Exercise the actual collector inside the pinned LLVM 19 analysis environment."""

import json
import os
from pathlib import Path

import pytest

from reccmp.source import SourceIndex, SourceIndexError


def test_native_batch_records_cache_and_errors(tmp_path: Path) -> None:
    if not os.environ.get("RECCMP_SOURCE_INDEXER") and not Path(
        "/usr/lib/llvm-19/include/clang/AST/ASTConsumer.h"
    ).is_file():
        pytest.skip(
            "run inside the pinned analysis image (LLVM 19 + reccmp-source-indexer)"
        )
    repository = tmp_path / "source with spaces"
    repository.mkdir()
    header = repository / "owner.h"
    header.write_text(
        "struct Owner {\n"
        "  int **pointers;\n"
        "  int (*callback)(int);\n"
        "  int *elements[2];\n"
        "  int &reference;\n"
        '};\nstatic_assert(sizeof(Owner) == 20, "size");\n',
        encoding="utf-8",
    )
    sources = [repository / name for name in ("first.cpp", "second.cpp", "empty.cpp")]
    clang_cl = next(
        (
            candidate
            for candidate in ("/usr/bin/clang-cl", "/usr/bin/clang-cl-19")
            if Path(candidate).is_file()
        ),
        None,
    )
    if clang_cl is None:
        # Debian's clang package may omit the cl driver name; the indexer still
        # selects CL mode from a path that ends in clang-cl.
        clang_cl = str(repository / "clang-cl")
        Path(clang_cl).symlink_to("/usr/bin/clang-19")
    for path, target in zip(sources, ("FIRST", "SECOND")):
        local_type = "int" if target == "FIRST" else "long"
        path.write_text(
            '#include "owner.h"\n'
            "extern int gShared;\n"
            "static int gLocal = 1;\n"
            f"int g{target} = 0;\n"
            f"// FUNCTION: {target} 0x00401000\n"
            f"int {target}() {{ {local_type} scratch = 0; "
            f"return gShared + gLocal + g{target} + (int)scratch; }}\n",
            encoding="utf-8",
        )
    sources[2].write_text(
        "// A successful unit may emit no records.\n", encoding="utf-8"
    )
    database = repository / "compile_commands.json"
    database.write_text(
        json.dumps(
            [
                {
                    "directory": str(repository),
                    "file": str(path),
                    "arguments": [
                        clang_cl,
                        "--target=i686-pc-windows-msvc",
                        "/c",
                        str(path),
                    ],
                }
                for path in sources
            ]
        ),
        encoding="utf-8",
    )
    cache = tmp_path / "cache with spaces"
    previous_root = os.environ.get("RECCMP_SOURCE_ROOT")
    os.environ["RECCMP_SOURCE_ROOT"] = str(repository)

    def collect():
        return SourceIndex.from_compile_database(
            repository,
            database,
            {"FIRST": [header, sources[0], sources[2]], "SECOND": [sources[1]]},
            cache_dir=cache,
            jobs=2,
        )

    try:
        index = collect()
        owners = [item for item in index.classes if item.qualified_name == "Owner"]
        assert len(owners) == 2
        assert {item.target for item in owners} == {"FIRST", "SECOND"}
        assert all(item.asserted_size == 20 for item in owners)
        assert [field.pointer_depth for field in owners[0].fields] == [2, 1, 0, 0]
        assert index.functions_by_address(target="FIRST")[0x401000].name == "FIRST"
        assert index.functions_by_address(target="SECOND")[0x401000].name == "SECOND"
        variables = {
            (item.target, item.qualified_name): item for item in index.variables
        }
        assert variables[("FIRST", "gFIRST")].definition_kind == "definition"
        assert variables[("FIRST", "gFIRST")].is_external
        assert variables[("FIRST", "gShared")].definition_kind == "declaration"
        assert variables[("FIRST", "gShared")].is_external
        assert variables[("SECOND", "gSECOND")].is_external
        assert not any(item.qualified_name == "gLocal" for item in index.variables)
        assert not index.conflicts
        declaration = index.functions_by_address(target="FIRST")[0x401000].declaration
        assert declaration is not None
        assert declaration.linkage == "external"
        assert (
            SourceIndex.from_dict(json.loads(json.dumps(index.to_dict()))).to_dict()
            == index.to_dict()
        )
        tu_cache = cache / "tu"
        before = sorted(path.stat().st_mtime_ns for path in tu_cache.glob("*.ndjson"))
        assert before
        assert collect().to_dict() == index.to_dict()
        after = sorted(path.stat().st_mtime_ns for path in tu_cache.glob("*.ndjson"))
        assert after == before
        header.write_text(
            header.read_text().replace("**pointers", "*pointers"), encoding="utf-8"
        )
        refreshed = collect()
        assert all(
            item.fields[0].pointer_depth == 1
            for item in refreshed.classes
            if item.qualified_name == "Owner"
        )
        sources[0].write_text("this is not valid C++;\n", encoding="utf-8")
        with pytest.raises(SourceIndexError, match="first.cpp"):
            collect()
    finally:
        if previous_root is None:
            os.environ.pop("RECCMP_SOURCE_ROOT", None)
        else:
            os.environ["RECCMP_SOURCE_ROOT"] = previous_root
