# Reccmp Decompilation Toolchain

[![Discord server](https://badgen.net/badge/icon/discord?icon=discord&label)](https://discord.gg/aSKCSXwpNp)
[![Matrix channel](https://badgen.net/badge/icon/matrix?icon=matrix&label)](https://matrix.to/#/#isledecomp:matrix.org)

`reccmp` (recompilation compare) is a collection of tools for decompilation projects. Functions and data are matched based on comments in the source code. For example:

```cpp
// FUNCTION: GAME 0x100b12c0
MxCore* MxObjectFactory::Create(const char* p_name)
{
  // implementation
}
```

This allows you to automatically verify the accuracy of functions, virtual tables, variable offsets and more.

Full documentation available on [our GitHub page](https://github.com/isledecomp/reccmp/).

## Object-to-original comparison

A compiled function need not survive archive extraction or linker dead-code
elimination to be compared. Select its exact COFF linker name and an independently
known original address and extent:

```sh
reccmp-reccmp --target GAME --object build/platform.obj \
    --symbol _InitializePlatform --orig-address 10001230 --size 48
```

This mode reads only the original binary and object file; it does not require a
recompiled executable or PDB. `--json FILE` saves the result. Exit status is zero
for relocation-masked byte identity and one for a difference. The extent is
required, rather than inferred from an original executable's alignment padding.

The same operation supports static functions and initialized, BSS, or common
data symbols. In Python, `reccmp.formats.coff.parse_coff_object` exposes sections,
symbol storage classes, auxiliary records, and relocations with their original
symbol-table indices. `CoffObject.contribution` selects a symbol's bytes and
relocations; `reccmp.compare.exact.compare_object_to_original` compares that
contribution. Data-only translation units are valid objects, not empty failures.

The comparison masks i386 DIR32, DIR32NB and REL32 operands and original PE base
relocations. Unsupported relocation types in the selected extent are rejected.
This result establishes masked byte identity only: it does not establish equal
relocation targets, equivalent source syntax, or semantic equivalence. Section
boundaries, globals, strings, and relocations remain available for those further
questions instead of being discarded by a function-only parser.
