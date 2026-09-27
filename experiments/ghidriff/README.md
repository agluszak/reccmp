# ghidriff POC: reccmp-selected pairs through Ghidra's decompiler

Disposable experiment. reccmp supplies the function correspondence;
ghidriff does Ghidra import/analysis/decompile and produces the diffs.
No reccmp code is touched — if this fails, delete the directory.

## Layout

- `export_pairs.py` — turns a `reccmp-reccmp --json` report into
  `selected-pairs.json`, attaching the reccmp verdict per pair.
- `selected-pairs.json` — 50 adversarial pairs:
  10 `exact`, 18 `effective` (compiler noise), 10 `known-bad`
  (real logic divergences incl. the MGS/HP/targeting fixes), 12 `ugly`
  (switches, x87, big CFGs).
- `reccmp_pairs.py` — `ReccmpPairsDiff(GhidraDiffEngine)`: `find_matches`
  returns the supplied pairs; everything else is stock ghidriff.
  `syms_need_diff` is overridden to always return true so that even
  same-size pairs get decompiled+diffed — stock ghidriff would skip them.

## Setup

```sh
uv venv .venv
uv pip install ghidriff --python .venv/bin/python
export GHIDRA_INSTALL_DIR=/path/to/ghidra_12.1.4_PUBLIC
```

## Run

Binaries used for the corpus below:

- old (retail): `wizardry-decomp-agent6/.wiz8-work/variants/gog-base/Wiz8.exe`
  (sha256 18a74ff6…)
- new (recomp): `wizardry/repin/build/decomp/Wiz8.exe`
  — repin is one commit behind the MGS HP/targeting/pathing fix
  (498cf7dfa), so the known-bad pairs are still buggy there.

```sh
.venv/bin/python reccmp_pairs.py \
  --pairs selected-pairs.json \
  -o out \
  /path/to/Wiz8-retail.exe /path/to/Wiz8-recomp.exe

# A/B: side-by-side HTML
.venv/bin/python reccmp_pairs.py \
  --pairs selected-pairs.json \
  -o out-sxs --sxs \
  /path/to/Wiz8-retail.exe /path/to/Wiz8-recomp.exe
```

Regenerate the corpus after re-running reccmp against the same recomp
build:

```sh
reccmp-reccmp --target WIZ8 --json wiz8-report.json --json-diet   # from build/decomp
python export_pairs.py wiz8-report.json selected-pairs.json
```

## Scoring

Classify each pair by reading the decompiled diff:

- CLEAN — only irrelevant compiler noise
- ACTIONABLE — diff points at the real logic difference
- NOISY — decompiler noise dominates
- MISLEADING — suggests logic divergence where there is none
