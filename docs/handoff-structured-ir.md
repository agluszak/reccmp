# Structured IR migration

Branch: `refactor/structured-identities`. The starting commit was `6256ed08`;
the work below continues it. The target is that reccmp parses text at input
boundaries and never parses a string it rendered itself.

## Current state

The verifier accepts `DecodedInstruction` rows and `FunctionImage`s only.
`DisasmLiteTuple`, `ResolvedAsm`, `AsmStream`, `Instruction`, `InstructionMeta`,
`with_display`, the text sanitizer, and the production text fallbacks are gone.
The only `parse_instruction()` caller is the test fixture boundary. Direct
control targets, relocations, helper calls, stack offsets, jump tables, and
call facts travel as typed operands or identities. `RawDiffOutput` is used for
reports, not as an analysis input. The `.display` reads in production are for
reporting and diagnostics. Unsupported operands are keyed by instruction
bytes, never Capstone's operand text.

Static locals retain their PDB parent function key through `CvdumpAnalysis`
and `load_cvdump`; matching uses the matched parent address and exact local
name, rejecting ambiguity. The marker's variable semantic ID is retained.
The Clang index emits variable storage kind and base semantic IDs, takes
function return types from `getReturnType()`, and requires structural field,
variable, and base metadata in Python. The old source-index schema is not
accepted. Physical layout conflict checks use structural facts rather than
type spelling. PDB frame offsets are parsed as signed integers at the cvdump
boundary.

The resolver returns display and proof identity together. Call facts use the
identity. Triage uses typed value tags and entity types, and the explanation
path uses symbolic values and identities. `mangled_facts` fills only missing
PDB facts. Importing `reccmp.analysis.crt_startup` before `reccmp.compare`
works; the package's `Compare` export loads lazily.

Exact admission requires equal semantic keys and branch topology, plus either
equal raw bytes or complete operand and control-flow models. An indirect call's
modeled operand completes its control target. A recognized jump table completes
its indirect jump through `FunctionImage.control_flow_complete`. Opaque operands
never gain exact proof from equal display text.

## Validation

- Full local suite: 1,408 passed, 221 skipped, 4 expected failures.
- Updated C++ indexer built with LLVM 21 in the pinned analysis image; source
  collector, source index, and source record integration tests: 30 passed.
- The finished WIZ8/SURRENDER corpus run `refactor-final/run5` had zero
  function status or reason changes against `refactor-3`. WIZ8 took 122.94 s
  and SURRENDER 16.26 s. The previous `refactor-3` WIZ8 run took 126.48 s.
- The corpus's original 150 MB source index uses the old schema. For validation,
  `source-index.json` was regenerated from frozen sources with the updated
  collector. The frozen snapshot lacks its original compile database; a
  rewritten database from the matching Wizardry checkout omitted two frozen
  translation units. The separate validation artifact
  `source-index-merged.json` adds those two units' original marker blocks and
  declarations. This is a corpus-only input artifact, not a compatibility path
  in reccmp.

## Publication

The structured-identities work is on reccmp PR #42 at `71a79dec`. Wizardry
agent6 pins that revision with `[witness]` in PR #693. The PR #42 review
findings are fixed on the function-image branch (see below), not on #42.

## Function-image and graph continuation (2026-09-27)

The continuation is on local branch
`refactor/function-image-graph`, based on `71a79dec`.
It is published as draft PR #43, stacked on PR #42.

Commits on this branch:

- `33207099`: `FunctionImage.instructions` contains only instructions;
  jump tables and embedded data are separate and the renderer interleaves them.
- `c70c24eb`: build one `FunctionGraph` from decoded code and tables; extent
  closure and the product verifier read this graph.
- `84adc0b2`: accept only current structured report format 2; remove the
  format-1 importer, `udiff`, and legacy effective reconstruction.
- `7ee6d0d3`: matched entities link original and recompiled side records
  without merging their dictionaries.
- `b50d1435`: entity catalog owns canonical original identities and project
  equivalence groups.
- `844e318b`: production assembly entry points use pure `decode_function()`;
  the old mutable `ParseAsm.parse_asm()` API is gone.
- `6faafa55`: external conditional graph edges keep their taken label, and
  embedded-data differences block effective proof without hiding earlier
  mismatch diagnostics.
- `c0e595a4`: CRT startup and body-equivalence analysis use the same function
  decoder instead of accessing `InstructGen` directly or decoding a body again.
- `558be1d6`: compiler fact dataclasses live in `source/records.py` rather
  than sharing a file with index orchestration and layout queries.
- `6c947933`: section discovery is private inside `decode_function()`;
  `instgen.py` and direct section-shaped callers are gone.
- `a40e5789`: delete `AsmRole` and all fake table instructions. Test switches
  pass explicit `JumpTable` records. Alias proofs check table case destinations.
- `bb586099`: make the report and graph tests pass CI Pylint.
- `630941a7`: per-TU Clang observation parsing and JSON fact construction
  moved to `source/observations.py`; `source/index.py` retains namespace
  derivation, marker binding, layout queries, and orchestration.
- `cb0d9a77`: comparison gets a fresh `FunctionComparator` after the catalog
  freezes, so its resolver and proof caches start from final identities.
- `774f1d8b`: compile-command normalization moved to `source/commands.py`.
- Marker block merging, declaration binding, and marker JSON projection now
  live in `source/markers.py`; the index passes compiler facts into that
  boundary instead of implementing marker joins itself.
- Trusted class layout and field-path queries now live in `source/layout.py`
  as methods of the same `SourceIndex` object.
- Cross-TU winner selection and conflict derivation now live in
  `source/derive.py`; `SourceIndex` assembles their results.
- The entity catalog now seals side facts when it freezes. Matched entity
  views share the sealed records, and prepared-analysis pickle caching keeps
  them sealed after reload.

Full Python suite: 1,387 passed, 221 skipped, 3 expected failures. Mypy and
repository-wide Pylint pass. The Wizardry WIZ8
corpus retained all 2,684 exact and 284 effective proofs against the pinned
`71a79dec` baseline. The decoder, role-removal, alias-guard, and frozen-
catalog comparator runs had
exactly the same status and reason for all 6,098 entries. Against
the pinned baseline, 14 inconclusive entries became mismatch and 8 mismatches
became inconclusive; those diagnosis changes remain under review. The web UI
lints and builds; Playwright Chromium
could not start because the browser executable is not installed locally.

The positional CFG verifier cannot yet be removed: a product-only probe
retained its successful proofs but lost branch-target diagnoses on 63 WIZ8
functions when the product graph alignment failed. Of 21 effective WIZ8
functions the product verifier could not prove, 17 were proved by positional
lockstep, three by diff-aligned lockstep, and one by relocation followed by
lockstep. Product stopped at nine analysis limits, six alignment failures,
and two state joins. On four more it reported a return-value or memory-address
difference despite a positional lockstep proof. Those conflicting outcomes
need a soundness review before using the product verifier as the sole
strategy. The probe also found 142 stream-verifier proofs that product can
already reproduce (113 positional, 19 diff-aligned, ten relocation).

Still open: replace the separate extent and callee-cleanup walks with
queries over the canonical graph where their evidence permits it; unify
machine and semantic instruction effects; finish separating source index
JSON IO and collector orchestration; finish typed diagnosis
payloads. The catalog's side fact maps are sealed, though entity pairing
objects remain mutable Python objects and the catalog still uses key-value
facts internally.
Keep cvdump and Unicorn as separate input and
execution engines.

Approved disk cleanup removed about 32 GB of Imperialism Rust incremental
artifacts and 124 MB of Wizardry temporary probe artifacts, without touching
source checkouts or retail evidence.

## Review fixes and strategy consolidation (2026-09-27, later)

Stacked on `480064a8`:

- `79415e93`: PR #42 review. Every absolute memory operand is a reference
  (unresolved ones keep a side-local identity); `compare_exact()` is the
  only EXACT gate; switch tables are recognized only for
  `jmp dword ptr [index*4 + table]` (`switch_index_register`); IMPORT
  entities carry `import_module`/`import_name`, and
  `EntityDb.callee_names()` serves both the CRT atexit recognizer and the
  assert fixup; the CRT collector has no address threshold; stack slots
  mismatch symmetrically; inline fingerprints use `operand_match_key()`.
- `efa467eb`: dead code removed (text register swap, mismatch clusters,
  unused comparator/catalog/admission helpers).
- `6d927169`: relocation reads Capstone register/flag access instead of
  its own opcode table; relocation tests are machine code.
- `8a1abe97`: the positional CFG strategy is gone. It made no unique proof
  on WIZ8/SURRENDER; at least 48 of its 58 WIZ8 branch-target mismatches
  paired a branch with a non-branch. Block-pairing conflicts on a branch
  edge are now branch-target differences (60 WIZ8 functions). Diff
  opcodes must cover every row (an empty list used to verify vacuously).
- `198b0dc7`: the Intel text parser lives in `tests/asm_rows.py` only.
- The product's result under an unanchored block pairing is an
  `unanchored_product` strategy attempt (a lead with the whole
  difference), replacing the `product_stop` fact strings.

Validation: full suite, mypy and Pylint pass. Corpus: WIZ8 and SURRENDER
built by Wizardry agent6 (`a0277cff`, reccmp pin `71a79dec`) with its new
source index. Proof counts are unchanged throughout (WIZ8 2,684 exact /
284 effective; SURRENDER 859 / 26); only the diagnosis changes above moved.
WIZ8 takes about 146 s.

Per-strategy probe on WIZ8 (284 effective): the product pairing alone
proves 263; diff-aligned lockstep has 3 unique proofs, relocation 1.
The product fails the others by analysis limit (9), alignment (6),
memory address (3), state join (2) and return value (1).

Still open: product coverage of those 21 so diff-aligned lockstep and
relocation can go; extent discovery and callee cleanup still walk Capstone
themselves (`discover_extent` needs recursive descent, which the linear
section discovery in `decode_function` does not do); block alignment's
instruction classes; typed difference facts.

