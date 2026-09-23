# About this fork

This is `agluszak/reccmp`, a fork of
[isledecomp/reccmp](https://github.com/isledecomp/reccmp). It is kept as a
small stack of feature commits on top of `upstream/master`, one per area:

1. Binary and PDB support
2. Annotations and project files
3. Source index
4. Structured instruction IR
5. Semantic verifier
6. Entity identity
7. Diagnostics and reports
8. Comparator and tools
9. Ghidra
10. This file

A change to the fork amends the commit for its area, not a new commit on
top. `git commit --fixup=<commit>` followed by
`git rebase -i --autosquash upstream/master` does that.

## Keeping rebases cheap

Upstream keeps changing its own files. Every line the fork changes in one of
them can conflict on the next rebase. So:

- **Put new logic in new modules.** Upstream-owned files should only get
  hooks: an import, a call, a base class. For example, `FunctionComparator`
  gets its fork behaviour from mixins in `body_equivalence.py`,
  `function_metadata.py`, `source_pins.py` and `inline_accounting.py`.
  Report JSON lives in `comparison_json.py`, CLI text rendering in
  `tools/asmcmp_text.py`, and SEH/FOLDED matching in `match_folded.py`.
- **Use upstream's tooling.** Use `requirements-tests.txt` and upstream's
  workflows. The only fork delta there is the Ghidra version pin.
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

Before switching a downstream project to a rebased revision, save a report
with the old revision and diff against it with the new one:

```sh
reccmp-reccmp --target <TARGET> --json old.json --silent   # old revision
reccmp-reccmp --target <TARGET> --diff old.json            # new revision
```

Understand any change in the diff before switching.
