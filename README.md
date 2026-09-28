# Reccmp Decompilation Toolchain

[![Discord server](https://badgen.net/badge/icon/discord?icon=discord&label)](https://discord.gg/aSKCSXwpNp)
[![Matrix channel](https://badgen.net/badge/icon/matrix?icon=matrix&label)](https://matrix.to/#/#isledecomp:matrix.org)

`reccmp` (recompilation compare) is a collection of tools for decompilation projects. It was born from the [decompilation of LEGO Island](https://github.com/isledecomp/isle). Functions and data are matched based on comments in the source code. For example:

```cpp
// FUNCTION: LEGO1 0x100b12c0
MxCore* MxObjectFactory::Create(const char* p_name)
{
  // implementation
}
```

reccmp pairs each annotated function with its original, decompiles both with Ghidra and reports the differences; it also checks virtual tables, global data and variable layouts. [Click to see the full syntax](docs/annotations.md).

You can supplement the code annotations with metadata from CSV files. See the [instructions and syntax](docs/csv.md).

At the moment, C++ compiled to 32-bit x86 with old versions of MSVC (like 4.20) is supported. Work on support for newer MSVC versions is in progress - testing and bug reports are greatly appreciated. Other compilers, languages and architectures are not supported at the moment, but feel free to contribute if you wish to do so!

## Getting started

### Installing / upgrading `reccmp`

1. (Recommended) Set up and activate a virtual Python environment in the directory of your recompilation project (this is different for different operating systems and shells).
2. Install `reccmp`: `pip install reccmp`

The next steps differ based on what kind of project you have.

### Contributing to a project that already uses `reccmp`

1. Compile the C++ project.
2. Run `reccmp-project detect --search-path "path/to/folder/with/original/binaries"`.
3. If there is no `reccmp-build.yml` after building: Navigate to the recompiled binaries folder and run `reccmp-project detect --what recompiled`.
4. Look into `reccmp-project.yml` to see what the target is called.
5. Run `reccmp-reccmp --target <YOURTARGET>`. You should see which functions differ from the original.

### Setting up an existing decompilation project that has not used `reccmp` before

1. Run `reccmp-project create --originals "path/to/original" --scm`. This generates two files `reccmp-project.yml` and `reccmp-user.yml`; the latter will automatically be added to the `.gitignore`.
2. Annotate one function of your existing project as shown above and recompile. Note that the recompiled binary should have the same name file name as the original.
3. Navigate to your recompiled binary and run `reccmp-project detect --what recompiled`. A file `reccmp-build.yml` will be generated. This file should also be user-specific (see below on how to auto-generate this file by the build toolchain).
4. Look into `reccmp-project.yml` to see what the target is called.
5. Run `reccmp-reccmp --target <YOURTARGET>` from the same directory. If all goes well, you will see the result for the function you annotated above.

### Fresh project

1. Run `reccmp-project create --originals "path/to/original/binary" ["path/to/second/original/binary"] --cmake-project`
2. You will see a lot of new files. Set up your C++ compiler and compile the project defined by `CMakeLists.txt`, ideally into a sub-directory like `./build`. Advice on building with old MSVC versions can be found at the [LEGO Island Decompilation project](https://github.com/isledecomp/isle).
3. Look into `reccmp-project.yml` to see what the target is called.
4. Navigate to the build directory and run `reccmp-reccmp --target <YOURTARGET>`.

### Config files

See the [documentation on the config files](./docs/project_files.md) for more information.

## Tooling

All scripts will become available to use in your terminal with the `reccmp-` prefix. Note that these scripts need to be executed in the directory where `reccmp-build.yml` is located.

* [`decomplint`](/reccmp/tools/decomplint.py): Checks the decompilation annotations (see above)
  * e.g. `reccmp-decomplint --target LEGO1`
  * Markers come from the Clang source index (see [docs/source-index.md](/docs/source-index.md)); set `source-index` in `reccmp-build.yml`, `RECCMP_SOURCE_INDEX`, or `--source-index`. `reccmp-reccmp` reads markers from the same index.
* [`reccmp`](/reccmp/tools/compare.py): Decompiles each annotated function and its original with Ghidra and diffs them with Ghidriff. Needs `GHIDRA_INSTALL_DIR`. See [docs/code-comparison.md](/docs/code-comparison.md).
  * Every function: `reccmp-reccmp --target LEGO1`
  * One function, with its diff: `reccmp-reccmp --target LEGO1 --orig-address 0x100ae1a0`
  * Side-by-side HTML for review: `reccmp-reccmp --target LEGO1 --filter LegoWorld --sxs`
  * Parsed PDB and prepared entity analysis is cached in `.reccmp-cache` beside the PDB, Ghidra's analysis of both binaries in `.reccmp-cache/ghidra`. Pass `--no-cache` to diagnose a cache-independent run.
* [`roadmap`](/reccmp/tools/roadmap.py): Compares symbol locations in an original binary with the same symbol locations of a recompiled binary
* [`verexp`](/reccmp/tools/verexp.py): Verifies exports by comparing the exports of the original DLL and the recompiled DLL
* [`vtable`](/reccmp/tools/vtable.py): Asserts virtual table correctness by comparing a recompiled binary with the original
  * e.g. `reccmp-vtable --target LEGO1`
* [`datacmp`](/reccmp/tools/datacmp.py): Compares global data found in the original with the recompiled version
  * e.g. `reccmp-datacmp --target LEGO1`

## Ghidra Import

We also have tooling to import the information from the decompilation into [Ghidra](https://github.com/NationalSecurityAgency/ghidra). See the relevant [README](reccmp/ghidra/README.md) for additional information.

## Best practices

We have established some [best practices](docs/recommendations.md) that have no impact on `reccmp`'s output, but have made a positive impact on the LEGO Island decompilation.

## Contributing

Feel free to contribute to this project if you are interested! More information can be found at [CONTRIBUTING.md](./CONTRIBUTING.md).
