# Compiler-backed source index

`SourceIndex.from_compile_database()` collects declarations, layouts and reccmp
markers directly from Clang. Run reccmp inside the pinned analysis image
(LLVM 19 + prebuilt `reccmp-source-indexer`): there is no Docker orchestration
inside the Python package.

The index is the only source of markers: `reccmp-reccmp`, `Compare.from_target`
and `reccmp-decomplint` read them from it. Point reccmp at the index with
`source-index` in `reccmp-build.yml` (relative to that file), or with
`RECCMP_SOURCE_INDEX`:

```yml
project: ..
source-index: ../build/source-index.json
targets:
  GAME:
    path: GAME.EXE
    pdb: GAME.PDB
```

## Markers

The indexer's preprocessor comment handler sees every `//` comment in active
code. Each run of such comments on consecutive lines that contains something
shaped like a marker becomes a `marker-block` record: the comment lines, and
the declarations that begin at the first code token after them (functions with
their extent and definition status, variables with local-static status and
enclosing function, classes), plus the first string literal on that line as
the bytes the compiler emits. A template header, a brace-less `extern "C"`,
or a macro that expands to nothing on the same line do not separate a marker
from its declaration.

The marker grammar stays in `reccmp.parser`: `marker.py` reads one marker line
and `reader.py` decides what a block annotates. `FUNCTION`/`STUB` bind to the
function definition (one per template instantiation), `GLOBAL` to a variable
(a local static belongs to its function's marker), `VTABLE` to a class,
`STRING` to the string literal; a name comment after the markers completes
them by name instead. `LINE` markers stand alone.

Markers the compiler never saw — in files no translation unit includes, or in
code the preprocessor skipped — are reported by `reccmp-decomplint` as
`marker_not_compiled`. The index records the sha256 of every target source file
it was collected from; reccmp refuses an index that is older than the sources,
and decomplint reports `stale_source_index`. Pass every file under the
target's source roots to `from_compile_database`.

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

Each wanted translation unit is cached by indexer identity (the collector
binary plus the Clang libraries it reports with `--version`), compile command,
main-file contents, and the digests of every file Clang reported it including.
File digests persist in `digests.json`, reused while a file's size, mtime,
inode and ctime are unchanged (`paranoid=True` rehashes everything). Python
merge code and marker aliases do not invalidate artifacts. `force=True`
rebuilds every wanted unit.

Misses run on persistent `indexer --serve` workers (one JSON job per stdin
line, one reply per stdout line), each handed the next job when idle, so LLVM
starts once per worker and one slow unit does not hold up others. A wrapper
around the collector (e.g. `docker run`) must keep stdin attached (`-i`).

Several processes may share one cache: artifacts are published by atomic
rename and never locked; only building the collector takes a lock. Every
collection writes `profile.json` to the cache: Python phase times, hits,
misses with reasons (`new`, `forced`, `indexer_changed`, `command_changed`,
`main_file_changed`, `dependency_changed:<path>`), and for fresh units the
indexer's own phases (driver setup, frontend, our consumer, member-use
traversal, marker blocks, serialization) plus records and bytes by kind.

Most of a unit's records describe headers that many units include (on the
Wizardry corpus a declaration line recurs 32 times on average, a class 56
times). Units are loaded through a `RecordPool` that parses each distinct
artifact line once and shares the record; derivation sees each record once.

The index also lists, per unit, the repository files it includes
(`unit_dependencies`).

## Records

Records retain compiler-owned source signatures, parameter reference forms, and
field pointer depth. Declarations carry linkage, storage class, and variadic
status; only external-linkage variables are indexed. Conflicting size
assertions are errors inside one link namespace.

Records are compiler facts: they say nothing about which target or unit they
belong to, so units loaded through the `RecordPool` share them. That context
is the record's `DeclarationKey`: `(target, semantic_id)` for external
entities, plus the defining `unit_id` for TU-local ones, since TU-local
functions of different units can share a mangled name. The index keys
declarations, classes and variables by it, and member uses by the key of the
function making them. Markers carry the key of their declaration. The JSON
projection writes each record with its key's `target` and `unit_id` (member
uses: `function_unit_id`), and markers a `[target, semantic_id, unit_id]`
`declaration_key`. A marker on a TU-local function defined in a header binds
the first including unit's copy: the copies are identical, and nothing states
which one the marker's address is.

### Function facts

Every function declaration carries `call`, what a caller may assume under the
Microsoft x86 ABI: whether ecx and edx carry arguments, the argument bytes the
callee removes (from Clang's parameter sizes; `null` when a record return or a
non-trivial by-value argument leaves it undecided), and the return kind. The
calling convention is the one Clang assigned, not one inferred from spelling.
`SourceIndex.call_facts_for(semantic_id)` returns them as `CallFacts`; the
comparator takes each field from the PDB type record first, then from Clang,
then from the decorated name. On the Wizardry corpus Clang's facts agree with
the PDB and the decorations for all 6,492 recompiled functions it declares.

Member uses state the object of the access (`base`: `this`, `parameter` with
its index, `local`, `global`, `member`, `other`), and each integer, enumeration
or pointer conversion around a use states its widths and whether its source is
signed. `function-facts` records list a body's calls: the callee's semantic id,
whether the call is virtual (with the declaration introducing the slot and the
object's static class), the call's object, and which arguments are plain field
reads. `SourceIndex.function_facts_for(semantic_id)` assembles one function's
call facts, accesses and calls into `FunctionFacts`.

These facts describe the reconstruction. They may explain or constrain the
recompiled side of a comparison; they never prove the original equivalent.

### Derivation

`TranslationUnitRecords` holds one unit's observations. `derive_namespace()` /
`SourceIndex.from_units()` partition by target, group observations by key,
then derive winners and conflicts. Compile-database entries that are not owned by any
requested target are skipped before cache lookup or Clang.

Run the collector integration tests inside the pinned image. The image's
prebuilt collector predates any local change to `indexer.cpp`, so build one
from the working tree:

```sh
docker run --rm -v "$PWD:/work" -w /work reccmp-source-test bash -lc '
  clang++ -O1 -std=c++17 -fno-rtti -fno-exceptions -I/usr/lib/llvm-19/include \
    reccmp/source/indexer.cpp -o /tmp/indexer \
    /usr/lib/llvm-19/lib/libclang-cpp.so.19.1 /usr/lib/llvm-19/lib/libLLVM.so.19.1 &&
  uv venv -q /tmp/venv && uv pip install -q --python /tmp/venv -e . -r requirements-tests.txt &&
  RECCMP_SOURCE_INDEXER=/tmp/indexer /tmp/venv/bin/python -m pytest tests/test_source_batch.py'
```
