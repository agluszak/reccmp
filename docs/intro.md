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

reccmp pairs each annotated function with its original, decompiles both with Ghidra and reports the differences; it also checks virtual tables, global data and variable layouts.

Full documentation available on [our GitHub page](https://github.com/isledecomp/reccmp/).
