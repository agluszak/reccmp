# About this fork

This is `agluszak/reccmp`, a fork of
[isledecomp/reccmp](https://github.com/isledecomp/reccmp). It supplies
reconstruction-specific correspondence, metadata, data checks and source
integration; code comparison is [Ghidriff](https://github.com/clearbluejar/ghidriff)'s.
The fork does not interpret or normalize machine instructions.

```text
target configuration + binaries + PDB + source index
                        |
                        v
          entity catalog (Compare): pairs, pair basis, names, extents
                 |                              |
                 v                              v
   manifest -> Ghidra + Ghidriff        datacmp, vtable, decomplint
   (reccmp-reccmp, reccmp/ghidriff)     (retained checks)
                 |
                 v
   summary.json + Ghidriff report, linked to the source
```

- `reccmp/compare/core.py` prepares the catalog: PDB, annotations, binary
  structure. Every pair records its `PairBasis`. It runs no comparison.
- `reccmp/compare/manifest.py` is what the differ receives: requested
  functions (paired or not), shared names, extents, aliases, input digests.
- `reccmp/ghidriff/` adapts Ghidriff: functions at known entries, names,
  symmetric literal typing, referenced-data contents, one result per
  requested function. See `docs/code-comparison.md`.
- `reccmp/compare/vtables.py`, `variables.py` (datacmp) and the source
  index's layout checks are independent of code comparison.

Generic decompiler or report problems belong in Ghidriff or Ghidra, not here:
`requirements.txt` pins a Ghidriff revision (currently the fork
`agluszak/ghidriff` with the same-basename decompiler-pool fix) and fixes go
upstream from there.

## Deliberate deletions of upstream code

Upstream's assembly comparison and its report are deleted: `compare/asm/`,
`compare/functions.py`, `diff.py`, `pinned_sequences.py`, `report.py`,
`difflib.py`, `tools/asmcmp.py`, `aggregate.py`, `stackcmp.py`, the HTML
report assets and `webui/`. Upstream's handwritten C++ marker reader is
deleted too: markers come only from the Clang source index
(`docs/source-index.md`). When a rebase conflicts on one of these, keep it
deleted; port a marker-grammar change to `reccmp/parser/reader.py` or
`marker.py`.

## Keeping rebases cheap

- **Put new logic in new modules.** Upstream-owned files should only get
  hooks.
- **Use upstream's tooling**: `requirements-tests.txt` and upstream's
  workflows.
- **Don't reformat or tidy upstream code** that the fork doesn't otherwise
  need to change.

## Rebasing onto upstream

```sh
git fetch upstream
git tag fork/$(date +%Y-%m-%d) master   # keep pinned revisions reachable
git config rerere.enabled true          # reuse earlier conflict resolutions
git rebase upstream/master
pytest && pylint reccmp tests && mypy ./reccmp ./tests
```

Downstream projects pin fork revisions by SHA. A rebase rewrites every fork
commit, so tag the old tip before force-pushing.

Before switching a downstream project to a new revision, keep the old
revision's `summary.json` for the same binaries and compare outcomes per
function with the new one. Understand every change before switching.
