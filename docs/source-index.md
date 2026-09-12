# Compiler-backed source index

`SourceIndex.from_compile_database()` collects declarations and layouts directly
from Clang, in parallel, then binds reccmp markers to those records. It replaces
the serial `from_compilation_database` and `from_compilation_database_targets`
APIs. No whole-AST JSON files or downstream collector subclasses are needed.

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

The collector currently builds against the Linux LLVM 19 development libraries
(`clang++`, `/usr/lib/llvm-19`, `libclang-cpp.so.19`, and `libLLVM-19.so.1`),
which is what Debian trixie ships. Its C++ source ships in the Python package;
the executable stays in the project's disposable cache. `RECCMP_LLVM_VERSION`
overrides the LLVM release for images that ship another one. Native commands use
each compile entry's working directory.
For container compile databases, pass `container_image`, `mounts={host_path:
"/container/path"}`, and `compilation_root=Path("/container/repo")`. The whole
batch runs in one container with the supplied mounts read-only. `clang` optionally
overrides the compiler named in the database. No emitter or plugin hooks exist.

Each translation unit is cached by its compile command, source contents,
container/compiler identity, and the dependency set Clang reports. Host paths
are remapped through `mounts`; toolchain includes are covered by the image
identity and are not hashed on the host. Validated TU artifacts are aggregated
into the final index. `force=True` refreshes every unit. Concurrent builders
serialize through the cache lock; compiler failures include the translation unit
and diagnostics. Empty successful output is valid. Writing an unchanged index
preserves its file timestamp.

Records retain compiler-owned source signatures, parameter reference forms, and
field pointer depth (arrays and references are not peeled). Declarations carry
their computed linkage, written storage class, and whether they are variadic;
variables carry their canonical type, linkage, storage class, and definition
kind. Only external-linkage variables are indexed. Markers retain their target
and folded status. `SourceIndex.from_dict()` reads the JSON projection back into
these same types; `functions_by_address(target=...)` selects the unfolded owner
and refuses ambiguous ownership. Use `marker.declaration` for function semantics
and `marker.name` for either a declaration or a named non-body emission.

For already-collected data, `SourceCollector.collect_record()` accepts one
declaration, variable, class, or size-assertion observation tagged with its
translation unit; `collect_records()` accepts NDJSON. Observations are not
merged while they arrive. `SourceCollector.derive()` / `SourceIndex.from_collector()`
partition by link namespace (the TUs owned by a target), group external entities
by `semantic_id` and non-external functions by `(unit_id, semantic_id)`, then
derive winners and conflicts. Definitions replace declarations, an initialized
definition beats a tentative one, and the first located class wins. Conflicting
size assertions inside one namespace are errors. Neither method mutates the
supplied record. `ast_command()` exposes the compile-argument normalization used
by the direct-record collector.

Run `RECCMP_SOURCE_TEST_IMAGE=<image> uv run --group test pytest
tests/test_source_batch.py` to exercise actual compilation, multi-target ownership,
structured layouts, cache invalidation, empty units, and compiler errors.
