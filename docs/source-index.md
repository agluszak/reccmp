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

Every function declaration carries `call`, what a caller may assume: whether
ecx and edx carry arguments, the argument bytes the callee removes, and the
return kind. It is read from Clang's own ABI lowering (`CGFunctionInfo`, from
a `CodeGenModule` that emits nothing), so hidden return pointers, `inalloca`
argument blocks and small records returned in registers are Clang's decision,
not ours. Anything not modelled exactly is `null`, never guessed: conventions
other than cdecl/stdcall/thiscall/fastcall, expanded or coerced aggregates,
constructors with a hidden virtual-base argument, incomplete types. The
calling convention is the one Clang assigned. On the Wizardry corpus the stack
cleanup agrees with the `ret N` MSVC emitted for every recompiled function
checked (7,595).

`SourceIndex.call_facts_for(key)` returns a declaration's facts;
`call_facts_named(semantic_id)` answers by mangled name only when every
declaration with that name agrees. The comparator takes each field from the
PDB type record first, then from Clang (the declaration the function's marker
binds, else the name lookup), then from the decorated name.

Member uses state the object of the access (`base`): its root (`this`,
`parameter` with its index, `local` or `global` with the declaration's
identity, `call`, `other`) and the fields leading from the root to it, each
step saying whether it went through a pointer; `arrow` says whether the access
itself dereferences its base. `this->a.b.c` has root `this` and path `a, b`.
A use's `conversions` are those of the field's own value: the casts wrapping
the use before it becomes an operand of anything else. For integer,
enumeration and pointer values they state widths and source signedness.

`function-facts` records list a body's explicit calls (`CallExpr` nodes, not
constructors, destructors or other implicit calls, so not a call graph): the
callee's semantic id; for virtual calls every declaration introducing a slot
the call may use (more than one under multiple inheritance) and the object's
static class; the call's object; which arguments are plain field reads.
They also list the body's built-in comparisons (overloaded comparison
operators are calls): the operator, the type compared in after the usual
arithmetic conversions, with its width and signedness, and each operand as
written, with its type, field identity and constant value. A
`branch_condition` mismatch shows the comparisons on the recompiled
instruction's source line (`source_comparisons`): whether the source asks for
a signed or an unsigned comparison, and at what width, is usually what a
`jl`/`jb` difference comes down to.

`SourceIndex.function_facts_for(key)` assembles one function's call facts,
accesses, calls and comparisons into `FunctionFacts`.

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
