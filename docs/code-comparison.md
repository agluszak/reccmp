# Comparing code

`reccmp-reccmp` decompiles every annotated function and its original with
Ghidra and diffs the results with [Ghidriff](https://github.com/clearbluejar/ghidriff).
reccmp does not interpret machine instructions itself. It contributes what only
the reconstruction knows: which functions correspond, under which names, and
where their source is.

```sh
export GHIDRA_INSTALL_DIR=/path/to/ghidra_12.1.4_PUBLIC
reccmp-reccmp --target GAME                         # every annotated function
reccmp-reccmp --target GAME --orig-address 0x401000 # one function, diff printed
reccmp-reccmp --target GAME --filter Monster --sxs  # side-by-side HTML too
```

The report directory (`--output`, default `reccmp-<target>`) holds:

- `summary.json`: one result per requested function, the counts, and the
  inputs the run used (binary digests, manifest digest, reccmp, Ghidra and
  Ghidriff versions, the Ghidra project);
- `manifest.json`: the pairs and names handed to Ghidra, with how each pair
  was established;
- `<target>.ghidriff.md` (and `sxs_html/` with `--sxs`): Ghidriff's report of
  the functions with differences;
- `ghidriff.log`.

## Results

Every function the selection requests is accounted for:

| Result | Meaning |
|---|---|
| `differences` | The decompiled code or the data it refers to differs; review the evidence |
| `no-differences` | The analysis completed without a visible difference |
| `unpaired` | reccmp has no counterpart for the original function |
| `analysis-failed` | The comparison did not complete; the failure says why |

An empty decompiled diff is not a proof of equivalence, and a changed
decompilation is not necessarily a bug. The results say what the analyzed
output shows. The exit status does not depend on them.

Analysis failures name their cause: Ghidra has no function at the entry and
could not create one (`no-function`); the entry lies inside a function Ghidra
starts elsewhere (`entry-conflict`, with that function's address); the
decompiler failed (`decompile-error`).

## What both programs receive

Both programs are analyzed the same way, without debug information. Then:

- the original's memory blocks get the write permission of the recompiled
  blocks with the same name. A packed or protected original can have writable
  read-only sections, and the decompiler folds reads of read-only memory into
  constants: without this, a `const` global would show as a name in one
  program and as its value in the other;
- functions are created at every entry the catalog knows, with Ghidra's own
  commands;
- every pair gets one name in both programs, its reccmp name (qualified with
  the original address when several pairs share a name). A duplicate the
  catalog identifies with a pair gets the pair's name;
- unpaired catalog entities keep their own image's name, qualified so it can
  never look like a correspondence;
- string, wide string and float constants the catalog found get the same
  data type on both sides; one-sided string or float typing Ghidra inferred
  for other paired objects is removed, so the decompiler shows the shared name
  on both sides instead of a literal on one;
- a reference into a paired object gets a label for the object and offset,
  so `array + 4` cannot read as `array`.

No reconstruction types, prototypes or calling conventions are applied: the
recovered source must not shape both decompilations toward the same answer.

Correspondence decides names; contents never do. Two different globals with
equal contents keep different identities.

## Referenced data

The decompiler may print two different literals under equal labels. reccmp
therefore compares the contents of the data each function refers to:

- a paired object both sides refer to is compared by identity, over the
  catalog's extent for it (the recompiled PDB's size stands for the original
  when only it is known);
- the remaining references have no counterpart, so only their contents are
  compared, as a multiset. A string on one side and the same bytes, untyped,
  on the other are consistent; so are zero-filled regions of any length. A
  location whose extent neither the catalog nor Ghidra knows, and an address
  stored in data, have no comparable contents.

`unidentified_references` in the summary counts references that were only
compared by contents.

## Pair basis

Each pair records why it exists (`basis` in the manifest and the summary):

- `annotation`: a source annotation or project declaration bound to a
  recompiled debug symbol or line;
- `derived`: binary structure or another pair (entry point, imports,
  exports, thunks, SEH records, CRT startup, vtable slots);
- `content`: equal contents only (string literals).

## Reuse between runs

Ghidra's analysis of both binaries is kept in a Ghidra project
(`--ghidra-projects`, default `.reccmp-cache/ghidra` beside the recompiled
PDB), named by the original binary's digest and the Ghidra and Ghidriff
versions. A new recompiled build replaces the previous one in that project;
the original is analyzed once. The analyzed programs are kept pristine; each
run starts from them and
applies the current catalog, so a changed annotation never needs a new
analysis and never leaves stale names behind.

## Limitations

Large functions with switches and floating-point code often decompile with
structural differences on otherwise equivalent code. Differences in register
use, stack layout and instruction selection mostly disappear in the
decompilation, but not always (a bit test of a different width, a comparison
against an address one past the end of an array). Inspect those with Ghidra:
the project the summary names holds both programs, named.
