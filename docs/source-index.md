# Compiler-backed source index

`SourceIndex.from_compile_database()` collects declarations and layouts directly
from Clang, then binds reccmp markers to those records. Run reccmp inside the
pinned analysis image (LLVM 19 + prebuilt `reccmp-source-indexer`): there is no
Docker orchestration inside the Python package.

```python
from pathlib import Path
from reccmp.source import SourceIndex

repo = Path.cwd()
index = SourceIndex.from_compile_database(
    repo,
    repo / "build/compile_commands.json",
    {"GAME": list((repo / "src").rglob("*.cpp")) + list((repo / "include").rglob("*.h"))},
    jobs=4,
)
index.write(repo / "build/source-index.json")
owners = index.functions_by_address(target="GAME")
```

Set `RECCMP_SOURCE_INDEXER` (or put `reccmp-source-indexer` on `PATH`) to a
collector built against LLVM 19. Without that, the first collection compiles
`indexer.cpp` into the cache using the host's LLVM 19 development libraries.
`RECCMP_SOURCE_ROOT` must be the repository root the compile database paths use.
`clang` optionally overrides the compiler named in the database.

Each wanted translation unit is cached by indexer identity, compile command,
main-file contents, and Clang's reported dependency digests. Shared headers are
hashed at most once per invocation. Python merge code and marker aliases do not
invalidate those artifacts: only Clang is expensive. Validated TU records are
streamed once and aggregated into the final index. `force=True` rebuilds every
wanted unit.

Records retain compiler-owned source signatures, parameter reference forms, and
field pointer depth. Declarations carry linkage, storage class, and variadic
status; only external-linkage variables are indexed. Markers store a
`(target, semantic_id)` declaration key in the JSON projection rather than a
nested copy of the declaration. Conflicting size assertions are errors inside
one link namespace.

`TranslationUnitRecords` holds one unit's observations. `derive_namespace()` /
`SourceIndex.from_units()` partition by target, group external entities by
`semantic_id` and non-external functions by `(unit_id, semantic_id)`, then
derive winners and conflicts. Compile-database entries that are not owned by any
requested target are skipped before cache lookup or Clang.

Run the collector integration test inside the pinned image:

```sh
docker run --rm -v "$PWD:/work" -w /work \
  -e RECCMP_SOURCE_INDEXER=/usr/local/bin/reccmp-source-indexer \
  reccmp-source-test \
  uv run --group test pytest tests/test_source_batch.py
```
