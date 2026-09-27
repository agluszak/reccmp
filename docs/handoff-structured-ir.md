# Handoff: structured IR refactor (branch `refactor/structured-identities`)

Goal (from the user): one PR where **no string reccmp produces is ever parsed
by reccmp again**. Three migrations: (1) IR is mandatory, (2) derived analyses
stay structured, (3) compiler/debug facts stay structured. No backwards
compatibility. Base: `origin/master` `df3472c5`. This commit is **work in
progress**: the pipeline runs end to end, but tests are not migrated and
verdicts regressed (see "State").

## Done

### IR is mandatory
- `asm/instgen.py`: `CodeSection.contents` is `list[DecodedInstruction]`;
  `DisasmLiteTuple`, `as_lite_tuple`, `displacement_regex`, `stop_at_int3`,
  `get_disassembler` deleted. Jumps/tables from `branch_target` and typed mem
  operands (`_table_displacement`).
- `asm/parse.py` rewritten: one `ReferenceResolver` (`resolve(addr, exact,
  indirect) -> ResolvedAddress(name, identity)`), no text sanitizer, no
  `_should_sanitize`/`_hex_style_addr`/`from_hex`/regexes; every row goes
  through `sanitize_row`; identity never inferred from display
  (`_OFFSET_PLACEHOLDER` gone). `assert_fixup` replaces operands structurally.
- `asm/replacement.py`: `create_name_lookup` -> `create_resolver`;
  `canonical_callee_name` no longer appends `[CALLEE identity]`.
- `asm/model.py`: `Instruction` class removed; `parse_instruction` returns
  `(prefix, mnemonic, operands)` and is a text boundary for test fixtures only;
  `STACK_ENTRY_REGEX` removed; `ResolvedAddress` added.
- `asm/ir.py`: removed `ResolvedAsm`, `AsmStream`, `resolve_asm_stream`,
  `instruction_at`, `is_data_row`, `with_display`, `as_effective`,
  `from_effective`, `raw_operands`, `raw_op_str`, `excerpt_*`,
  `rewrite_stack_displacements`, string branches of key functions. Table
  markers carry payloads (`("case", offset)`, `("byte", b)`). Match key uses
  identity for resolved references (display only for side-local ones).
  New `local_branch_targets(rows)`.
- Verifier (`asm/verifier/*`) takes `DecodedInstruction` rows: lockstep, cfg,
  cfg_build, iso_cfg, relocation, block_align, schedule, obligations,
  evidence, semantics. `analyze_effective_match(codes, orig: FunctionImage,
  recomp: FunctionImage, metadata)`. `InstructionMeta` no longer passed
  (class still exists in instgen: delete it and `meta_from_decoded`,
  `collect_instruction_meta`). Display-equality admissions replaced by
  `instruction_semantic_key` equality. `call_facts` keyed by proof identity
  (`FunctionMetadata.call_facts: Callable[[Hashable], ...]`,
  `_call_facts_map` keyed by `entity_proof_identity`).
- `functions.py`: `compare_function_images` hands images to the verifier;
  exactness uses match keys, not display equality; `_compare_function_assembly`
  test wrapper removed.

### Derived analyses structured
- `stack_layout.py`: pairs from opcodes over row operands; modulo-stack via
  remapped structured keys; `StackLayoutResult` carried on
  `EntityCompareResult` / `ReccmpComparedEntity`; `tools/stackcmp.py` reads it.
- `inlines.py`: `Fingerprint = tuple[FingerprintRow(prefix, mnemonic,
  operands, callee)]`; register normalization on operands; effect summary on
  operands; helper calls by callee identity (`HelperCatalogEntry.identity`).
  `inline_accounting.py`: helper resolution from `("entity", orig, 0)`; the
  name/hex identity index deleted.
- `body_equivalence.py`: proofs on `FunctionImage`s via `_load_function_image`
  and semantic keys; `_code_rows`, `_code_shape`, `_alias_fingerprint`,
  caller edges all structured.
- `analysis/crt_startup.py`: operands/branch targets instead of `op_str` regex.

### Compiler/debug facts
- `cvdump/symbols.py`: `StackOrRegisterSymbol(frame_offset, register)` parsed
  at the boundary (signed); `location` string gone. Consumers: stack_layout,
  `ghidra/importer/pdb_extraction.py` (was unsigned `int(...,16)` — a bug).

## State (measured on reccmp-corpus/2026-09-26, run.sh)
- Runs: `refactor-base` (df3472c5) vs `refactor-1` (this commit).
- **Regression**: 262 exact->mismatch, 12 effective->mismatch, 49
  inconclusive->mismatch (e.g. SURRENDER 0x1000e130 `srClass::srClass`
  memory_value). Likely cause to check first: stores of vtables / addresses
  now become `("sym", Reference)` via `is_addr` on *every* row (the old gate
  skipped size<=4 rows), or resolver identities differing between sides where
  the old display compared equal. Diff one function with
  `asmcmp --verbose 0x1000e130` against master.
- **Performance**: WIZ8 470 s vs ~90 s. Suspect every-row sanitization
  (resolver calls for every `cmp imm`) and `_load_function_image` in body
  equivalence. Profile with cProfile on WIZ8.
- **Tests**: broken (collection errors). Needed: a test fixture helper
  (e.g. `tests/asm_rows.py`) that parses text once into `DecodedInstruction`
  rows/`FunctionImage` (addresses `0x1000+i`, `branch_target` from target
  indices, `control_target` for jumps, `register_access_known=False`), then
  migrate ~170 call sites (test_effective*.py, test_review_regressions,
  test_sanitize32 (rewrite onto sanitize_row), test_instgen, stack/inline
  tests, test_name_replacement (create_resolver)).

## Not started (from the plan)
- Static locals: keep `LdataEntry` parent relation in `CvdumpNode`
  (`parent_function`), delete `f"{v.name}___{sym.name}"` in
  `cvdump/analysis.py`, match by `(recomp_parent, name)` in
  `match_msvc.match_static_variables` (reject ambiguity); keep the anchor's
  variable semantic_id in `source/reader.py`.
- Source index: `SourceVariable.storage_kind` + `record_semantic_id`,
  `SourceBaseOffset.semantic_id`; make field structural fields mandatory;
  delete `strip_type_qualifiers`, `type_spelling_is_indirection`,
  `variable_type_is_indirection`; `indexer.cpp` return type from
  `function->getReturnType()`; rebuild the collector.
- `function_metadata._call_facts_of_node`: run `mangled_facts` only for fields
  still unknown.
- `triage._fact_shape`: stop regexing stringified facts; add typed facts.
- `explain.py`: remove remaining `(DATA)` / text assumptions (now uses
  identities; re-check `_fact_causes`).
- Delete `InstructionMeta`/`meta_from_decoded`/`collect_instruction_meta`;
  audit `grep -rn "\.display" reccmp` so no semantic code reads display.
- Known pre-existing: importing `reccmp.analysis.crt_startup` first triggers a
  circular import via `reccmp.compare.__init__`.
