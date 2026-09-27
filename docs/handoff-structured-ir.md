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

## Publication follow-up

The reccmp branch has not been pushed or opened as a PR. The Wizardry agent6
change `wltkuuul` still pins `reccmp` to `6c3d6f53` with `[witness]`; update
that pin only after this branch has a stable published revision, preserving
that checkout's other active changes.
