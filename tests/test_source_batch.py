"""Exercise the actual collector inside the pinned LLVM 19 analysis environment."""

import json
import os
from pathlib import Path

import pytest

from reccmp.source import SourceIndex, SourceIndexError


def test_native_batch_records_cache_and_errors(tmp_path: Path) -> None:
    # pylint: disable=too-many-statements
    if (
        not os.environ.get("RECCMP_SOURCE_INDEXER")
        and not Path("/usr/lib/llvm-19/include/clang/AST/ASTConsumer.h").is_file()
    ):
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
        '};\nstatic_assert(sizeof(Owner) == 20, "size");\n'
        "struct W8First { int unknown_04; int values[4]; };\n"
        "struct W8Second { int unknown_04; };\n"
        "struct W8Payload { int value; };\n"
        "struct W8Record { W8Payload payload; };\n"
        "struct SurrenderOnly { int unknown_04; };\n"
        "template<class T> struct DependentRecord { T value; };\n"
        "inline int ReadW8Field(W8First& object, int index) {\n"
        "  int value = object.unknown_04;\n"
        "  object.values[index] = value;\n"
        "  return value;\n"
        "}\n"
        "inline int ReadW8Second(const W8Second& object) { return object.unknown_04; }\n"
        "inline double ConvertW8Field(const W8First& object) {\n"
        "  return static_cast<double>(object.unknown_04);\n"
        "}\n"
        "inline int* AddressW8Field(W8First& object) { return &object.unknown_04; }\n"
        "inline int ReadW8Element(const W8First& object) { return object.values[2]; }\n"
        "inline void CopyW8Record(W8Record& dest, const W8Record& source) {\n"
        "  dest.payload = source.payload;\n"
        "}\n"
        "template<class T> int ReadDependentMember(T& object) { return object.unknown_dependent; }\n",
        encoding="utf-8",
    )
    sources = [
        repository / name
        for name in ("first.cpp", "second.cpp", "empty.cpp", "repeat.cpp")
    ]
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
    for path, target in zip(sources, ("WIZ8", "SURRENDER")):
        local_type = "int" if target == "WIZ8" else "long"
        path.write_text(
            '#include "owner.h"\n'
            "extern int gShared;\n"
            "static int gLocal = 1;\n"
            f"int g{target} = 0;\n"
            f"// FUNCTION: {target} 0x00401000\n"
            f"int {target}() {{ {local_type} scratch = 0; "
            + (
                "SurrenderOnly surrender{}; "
                "return gShared + gLocal + gSURRENDER + surrender.unknown_04 + (int)scratch; }\n"
                if target == "SURRENDER"
                else "return gShared + gLocal + gWIZ8 + (int)scratch; }\n"
            ),
            encoding="utf-8",
        )
    sources[2].write_text(
        "// A successful unit may emit no records.\n", encoding="utf-8"
    )
    sources[3].write_text(
        '#include "owner.h"\n'
        "// FUNCTION: WIZ8_REPEAT 0x00401004\n"
        "int WIZ8_REPEAT() { return 0; }\n",
        encoding="utf-8",
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
            {
                "WIZ8": [header, sources[0], sources[2], sources[3]],
                "SURRENDER": [sources[1]],
            },
            cache_dir=cache,
            jobs=2,
        )

    try:
        index = collect()
        owners = [item for item in index.classes if item.qualified_name == "Owner"]
        assert len(owners) == 2
        assert {item.target for item in owners} == {"WIZ8", "SURRENDER"}
        assert all(item.asserted_size == 20 for item in owners)
        assert [field.pointer_depth for field in owners[0].fields] == [2, 1, 0, 0]
        dependent_records = [
            item for item in index.classes if item.qualified_name == "DependentRecord"
        ]
        assert len(dependent_records) == 2
        assert all(item.size is None for item in dependent_records)
        assert all(item.fields[0].offset is None for item in dependent_records)
        assert index.functions_by_address(target="WIZ8")[0x401000].name == "WIZ8"
        assert (
            index.functions_by_address(target="SURRENDER")[0x401000].name == "SURRENDER"
        )
        variables = {
            (item.target, item.qualified_name): item for item in index.variables
        }
        assert variables[("WIZ8", "gWIZ8")].definition_kind == "definition"
        assert variables[("WIZ8", "gWIZ8")].is_external
        assert variables[("WIZ8", "gShared")].definition_kind == "declaration"
        assert variables[("WIZ8", "gShared")].is_external
        assert variables[("SURRENDER", "gSURRENDER")].is_external
        assert not any(item.qualified_name == "gLocal" for item in index.variables)
        assert not index.conflicts
        declaration = index.functions_by_address(target="WIZ8")[0x401000].declaration
        assert declaration is not None
        assert declaration.linkage == "external"
        wiz8_uses = index.for_target("WIZ8").member_uses
        surrender_uses = index.for_target("SURRENDER").member_uses
        first_uses = [
            item
            for item in wiz8_uses
            if item.owner == "W8First"
            and item.name == "unknown_04"
            and item.function == "ReadW8Field"
        ]
        assert len(first_uses) == 1
        assert first_uses[0].owner_identity
        assert first_uses[0].field_usr
        assert first_uses[0].offset_bits == 0
        assert first_uses[0].extent_bits == 32
        assert first_uses[0].operations == ("read",)
        first_field_identity = first_uses[0].field_identity
        assert any(
            item.owner == "W8Second" and item.name == "unknown_04" for item in wiz8_uses
        )
        assert not any(item.owner == "SurrenderOnly" for item in wiz8_uses)
        assert any(item.owner == "SurrenderOnly" for item in surrender_uses)
        assert any(item.owner_status == "unknown" for item in wiz8_uses)
        assert any(
            "array-index" in item.operations and "write" in item.operations
            for item in wiz8_uses
        )
        assert any(
            item.owner == "W8First"
            and item.name == "values"
            and item.array_indices
            and not item.array_indices[0].constant
            for item in wiz8_uses
        )
        assert any(
            item.owner == "W8First"
            and item.name == "values"
            and item.array_indices
            and item.array_indices[0].constant
            and item.array_indices[0].value == "2"
            for item in wiz8_uses
        )
        assert any(
            item.function == "ConvertW8Field"
            and any(
                conversion.destination_type == "double"
                for conversion in item.conversions
            )
            for item in wiz8_uses
        )
        assert any(
            item.function == "AddressW8Field" and "address-taken" in item.operations
            for item in wiz8_uses
        )
        assert any(
            "copy-memory-source" in item.operations
            for item in wiz8_uses
            if item.owner == "W8Record"
        )
        assert any(
            "copy-memory-destination" in item.operations
            for item in wiz8_uses
            if item.owner == "W8Record"
        )
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
        header.write_text(
            header.read_text()
            .replace(
                "struct W8First { int unknown_04; int values[4]; }",
                "struct W8First { int state; int values[4]; }",
            )
            .replace("int value = object.unknown_04;", "int value = object.state;")
            .replace(
                "static_cast<double>(object.unknown_04)",
                "static_cast<double>(object.state)",
            )
            .replace("&object.unknown_04", "&object.state"),
            encoding="utf-8",
        )
        renamed = collect()
        renamed_uses = [
            item
            for item in renamed.for_target("WIZ8").member_uses
            if item.owner == "W8First"
            and item.name == "state"
            and item.function == "ReadW8Field"
        ]
        assert len(renamed_uses) == 1
        assert renamed_uses[0].field_identity == first_field_identity
        sources[0].write_text("this is not valid C++;\n", encoding="utf-8")
        with pytest.raises(SourceIndexError, match="first.cpp"):
            collect()
    finally:
        if previous_root is None:
            os.environ.pop("RECCMP_SOURCE_ROOT", None)
        else:
            os.environ["RECCMP_SOURCE_ROOT"] = previous_root
