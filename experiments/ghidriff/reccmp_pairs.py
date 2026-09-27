#!/usr/bin/env python3
"""ghidriff engine that force-feeds reccmp function pairs into the diff pipeline.

reccmp already knows the orig<->recomp function correspondence; this engine
skips ghidriff's own matching entirely and supplies those pairs directly, so
the experiment measures only Ghidra's analysis + decompile + diff quality.

Usage mirrors `python -m ghidriff` plus one extra arg:

    python reccmp_pairs.py --pairs selected-pairs.json OLD_EXE NEW_EXE [-o OUT] [--sxs]

`--pairs` JSON entries: {"orig": <va>, "recomp": <va>, "name": "...", ...}
"""

import hashlib
import json
import re
from argparse import Namespace
from pathlib import Path

from ghidriff import GhidraDiffEngine
from ghidriff.parser import get_parser


def _slug_char(c: str) -> str:
    if c.isalnum():
        return c
    return {
        "?": "qmark", "!": "bang", ".": "dot", ",": "comma", ":": "colon",
        ";": "semi", "%": "pct", "/": "slash", "\\": "bslash", "-": "dash",
        " ": "sp", "'": "apos", '"': "quot", "(": "lp", ")": "rp",
        "=": "eq", "+": "plus", "*": "star", "&": "amp", "<": "lt",
        ">": "gt", "[": "lb", "]": "rb", "{": "lc", "}": "rc", "@": "at",
        "#": "hash", "$": "dollar", "^": "caret", "|": "pipe", "~": "tilde",
        "`": "btick", "\t": "tab", "\n": "nl", "\r": "cr", "\x00": "nul",
        "_": "us",
    }.get(c, f"x{ord(c):02x}")


def string_canon(text: str, wide: bool) -> str:
    """Content-derived name for a defined string.

    Same content -> same name on both sides, so identical literals produce
    no diff and differing literals (e.g. L"?" vs L"") stay visible.
    """
    prefix = "strw" if wide else "stra"
    if not text:
        return f"{prefix}_empty"
    if all(not c.isalnum() for c in text):
        # pure punctuation: readable word map (e.g. "?" -> qmark)
        slug = "_".join(_slug_char(c) for c in text)
    else:
        slug = "_".join(
            t for t in re.split(r"[^A-Za-z0-9]+", text) if t
        ).lower()[:48]
    digest = hashlib.sha1(("w" if wide else "a").encode() + text.encode("utf-16-le" if wide else "latin-1", "replace")).hexdigest()[:8]
    return f"{prefix}_{slug}_h{digest}"


def func_canon(name: str | None, orig_va: int, used: set) -> str:
    base = re.sub(r"[^A-Za-z0-9_]", "_", name) if name else f"f_{orig_va:08x}"
    base = f"rc_{base}"
    if base in used:
        base = f"{base}_{orig_va:08x}"
    used.add(base)
    return base


class ReccmpPairsDiff(GhidraDiffEngine):
    """GhidraDiffEngine whose matches come from a reccmp pair list."""

    def __init__(self, pairs_path: Path, names_path: Path | None = None, *args, **kwargs) -> None:
        self.pairs_path = pairs_path
        self.names_path = names_path
        super().__init__(*args, **kwargs)

    def find_matches(self, p1, p2) -> list:
        """Return [[], matched, []] — the corpus is the whole universe."""
        pairs = json.loads(self.pairs_path.read_text())
        space1 = p1.getAddressFactory().getDefaultAddressSpace()
        space2 = p2.getAddressFactory().getDefaultAddressSpace()

        matched = []
        skipped = []
        for pair in pairs:
            func1 = p1.getFunctionManager().getFunctionAt(
                space1.getAddress(pair["orig"])
            )
            func2 = p2.getFunctionManager().getFunctionAt(
                space2.getAddress(pair["recomp"])
            )
            if func1 is None or func2 is None:
                skipped.append((pair["name"], pair["orig"], pair["recomp"]))
                continue
            matched.append([func1.getSymbol(), func2.getSymbol(), ["ReccmpPair"]])

        for name, orig, recomp in skipped:
            self.logger.warning(
                "no Ghidra function at pair %s orig=0x%x recomp=0x%x",
                name,
                orig,
                recomp,
            )
        self.logger.info(
            "ReccmpPairs: %d supplied, %d matched, %d skipped",
            len(pairs),
            len(matched),
            len(skipped),
        )
        return [[], matched, []]

    def syms_need_diff(self, sym, sym2, match_types, skip_types=[]) -> bool:
        """Force every supplied pair through the decompile+diff path.

        Stock ghidriff skips matches whose body size and refcount are equal;
        for this experiment we want Ghidra's decompiled output for every pair,
        because same-size pairs are exactly where subtle logic diffs hide.
        """
        return True

    def apply_canonical_names(self, old: Path, new: Path) -> None:
        """Rewrite symbols in both programs to canonical reccmp-derived names.

        Same reccmp entity -> same name on both sides:
          functions : rc_<source name>          (from the full pair report)
          strings   : stra_/strw_<content slug> (same content -> same name)
          pointers  : rcp_<target canon name>   (vtables, string tables)
          scalars   : rcd_<len>b_<content hash>
        """
        from ghidra.program.model.symbol import SourceType

        report = json.loads(self.names_path.read_text())
        corpus = json.loads(self.pairs_path.read_text())

        entries = [
            e for e in report["data"] if e.get("address") and e.get("recomp")
        ]
        entries.sort(key=lambda e: int(e["address"], 16))

        used = set()
        canon_by_orig = {
            int(e["address"], 16): func_canon(e.get("name"), int(e["address"], 16), used)
            for e in entries
        }

        p1_name = self.gen_proj_bin_name_from_path(old)
        p2_name = self.gen_proj_bin_name_from_path(new)
        p1 = self.project.openProgram("/", p1_name, False)
        p2 = self.project.openProgram("/", p2_name, False)

        for prog, addr_key, corpus_key in (
            (p1, "address", "orig"),
            (p2, "recomp", "recomp"),
        ):
            space = prog.getAddressFactory().getDefaultAddressSpace()
            fm = prog.getFunctionManager()
            st = prog.getSymbolTable()
            lst = prog.getListing()
            rm = prog.getReferenceManager()

            tx = prog.startTransaction("canonical names")
            renamed_f = renamed_d = 0
            try:
                # functions: canonical name on both sides
                for e in entries:
                    va = int(e[addr_key], 16)
                    func = fm.getFunctionAt(space.getAddress(va))
                    if func is not None:
                        canon = canon_by_orig[int(e["address"], 16)]
                        for cand in (canon, f"{canon}_{int(e['address'], 16):08x}"):
                            try:
                                func.setName(cand, SourceType.USER_DEFINED)
                                renamed_f += 1
                                break
                            except Exception:
                                continue

                # data referenced by corpus functions: collect base names,
                # then label. Identical contents share a base name; when a
                # base is needed N times, members get a stable ordinal suffix
                # (sorted by address) so both programs produce the same set.
                pending = {}
                seen = set()
                for pair in corpus:
                    func = fm.getFunctionAt(space.getAddress(pair[corpus_key]))
                    if func is None:
                        continue
                    it = func.getBody().getAddresses(True)
                    while it.hasNext():
                        a = it.next()
                        for ref in rm.getReferencesFrom(a):
                            to = ref.getToAddress()
                            if not to.isMemoryAddress() or to in seen:
                                continue
                            seen.add(to)
                            if fm.getFunctionContaining(to) is not None:
                                continue
                            canon = self._data_canon(prog, to)
                            if canon is None:
                                continue
                            pending.setdefault(canon, []).append(to)

                for canon, addrs in pending.items():
                    addrs.sort(key=lambda x: x.getOffset())
                    multi = len(addrs) > 1
                    if multi and canon.startswith("rcd_") and len(addrs) > 16:
                        # huge same-content scalar groups (e.g. NUL bytes):
                        # ordinals would misalign across programs anyway
                        continue
                    for i, to in enumerate(addrs):
                        name = f"{canon}_{i}" if multi else canon
                        if self._label(prog, to, name, SourceType):
                            renamed_d += 1
            finally:
                prog.endTransaction(tx, True)

            self.project.save(prog)
            self.logger.info(
                "canonical names in %s: %d functions, %d data refs",
                prog.name, renamed_f, renamed_d,
            )
            self.project.close(prog)

    @staticmethod
    def _scan_string(prog, addr):
        """Read a NUL-terminated printable string starting at addr.

        Returns (text, wide) or None. Ghidra does not always define a
        *string* data item at string starts (lone L'?' shows as ushort), so
        scan the raw bytes: ASCII run, else UTF-16LE-style `XX 00` run.
        """
        mem = prog.getMemory()
        buf = bytearray(128)
        try:
            got = mem.getBytes(addr, buf)
        except Exception:
            return None
        if got < 2:
            return None
        raw = bytes(buf[:got])

        def printable(b):
            return b == 0x09 or 0x20 <= b <= 0x7E

        # ASCII
        end = 0
        while end < len(raw) and printable(raw[end]):
            end += 1
        if end >= 1 and end < len(raw) and raw[end] == 0:
            # require at least one char and a terminator
            if end >= 1:
                return raw[:end].decode("latin-1"), False
        # wide: pairs of (printable, 0)
        wend = 0
        while wend + 1 < len(raw) and printable(raw[wend]) and raw[wend + 1] == 0:
            wend += 2
        if wend >= 2 and wend + 1 < len(raw) and raw[wend] == 0 and raw[wend + 1] == 0:
            return raw[:wend:2].decode("latin-1"), True
        return None

    @staticmethod
    def _data_canon(prog, addr, depth=0):
        """Canonical name for a data reference, or None to leave it."""
        fm = prog.getFunctionManager()
        lst = prog.getListing()
        st = prog.getSymbolTable()
        data = lst.getDataContaining(addr)

        # Ghidra-defined strings: trust the data item's own value
        if data is not None:
            try:
                if data.hasStringValue():
                    val = str(data.getValue())
                    dt = str(data.getDataType()).lower()
                    wide = "unicode" in dt or "utf16" in dt or "wide" in dt
                    return string_canon(val, wide)
            except Exception:
                pass
        else:
            # undefined ref target: maybe a string Ghidra never typed
            scanned = ReccmpPairsDiff._scan_string(prog, addr)
            if scanned is not None:
                return string_canon(*scanned)
            return None

        value = data.getValue()
        if data.isPointer() and value is not None:
            tgt = value
            tfunc = fm.getFunctionAt(tgt)
            if tfunc is not None:
                # funcs are renamed before the data pass, so getName() is
                # already the canonical rc_* name for paired functions
                nm = tfunc.getName()
                if nm.startswith("FUN_"):
                    return None
                return "rcp_" + nm
            # vtables / string tables: name after the pointee's own canon
            if depth < 2:
                sub = ReccmpPairsDiff._data_canon(prog, tgt, depth + 1)
                if sub is not None and not sub.startswith("rcd_"):
                    return "rcp_" + sub
            return None

        # lone wide/narrow chars typed as scalars (e.g. L'?' as ushort)
        dt2 = str(data.getDataType()).lower()
        if any(k in dt2 for k in ("char", "short", "byte", "undef")):
            scanned = ReccmpPairsDiff._scan_string(prog, addr)
            if scanned is not None:
                return string_canon(*scanned)

        try:
            b = bytes(bytearray(x & 0xFF for x in data.getBytes()))
            return f"rcd_{len(b)}b_{hashlib.sha1(b).hexdigest()[:8]}"
        except Exception:
            return None

    @staticmethod
    def _label(prog, addr, name, SourceType):
        st = prog.getSymbolTable()
        sym = st.getPrimarySymbol(addr)
        try:
            if sym is None:
                st.createLabel(addr, name, SourceType.USER_DEFINED)
            elif (
                sym.getSource() == SourceType.DEFAULT
                or sym.getName() != name
            ):
                sym.setName(name, SourceType.USER_DEFINED)
            return True
        except Exception:
            return False


def main() -> None:
    parser = get_parser()
    GhidraDiffEngine.add_ghidra_args_to_parser(parser)
    parser.add_argument(
        "--pairs", required=True, type=Path, help="reccmp pair list JSON"
    )
    parser.add_argument(
        "--names",
        type=Path,
        default=None,
        help="full reccmp report JSON — inject canonical names before diff",
    )
    args = parser.parse_args()

    output_path = Path(args.output_path)
    output_path.mkdir(exist_ok=True, parents=True)

    if args.log_path == "None":
        engine_log_path = None
    elif args.log_path == parser.get_default("log_path"):
        engine_log_path = output_path / parser.get_default("log_path")
    else:
        engine_log_path = Path(args.log_path)

    if args.project_location == parser.get_default("project_location"):
        project_path = output_path / parser.get_default("project_location")
    else:
        project_path = Path(args.project_location)

    if args.symbols_path == parser.get_default("symbols_path"):
        symbols_path = output_path / parser.get_default("symbols_path")
    else:
        symbols_path = Path(args.symbols_path)

    if args.gzfs_path == parser.get_default("gzfs_path"):
        gzfs_path = output_path / parser.get_default("gzfs_path")
    else:
        gzfs_path = Path(args.gzfs_path)

    binary_paths = args.old + [b for sub in args.new for b in sub]
    binary_paths = [Path(p) for p in binary_paths]
    missing = [p.name for p in binary_paths if not p.exists()]
    if missing:
        raise FileNotFoundError(f"Missing Bins: {' '.join(missing)}")

    project_name = f"{args.project_name}-{binary_paths[0].name}-{binary_paths[-1].name}"

    engine = ReccmpPairsDiff(
        args.pairs,
        names_path=args.names,
        args=args,
        verbose=True,
        threaded=args.threaded,
        max_ram_percent=args.max_ram_percent,
        print_jvm_flags=args.print_flags,
        jvm_args=args.jvm_args,
        force_analysis=args.force_analysis,
        force_diff=args.force_diff,
        verbose_analysis=args.va,
        no_symbols=args.no_symbols,
        engine_log_path=engine_log_path,
        engine_log_level=args.log_level,
        engine_file_log_level=args.file_log_level,
        min_func_len=args.min_func_len,
        use_calling_counts=args.use_calling_counts,
        bsim=args.bsim,
        bsim_full=args.bsim_full,
        gdts=args.gdt,
        base_address=args.base_address,
        program_options=args.program_options,
        decompiler_timeout=args.decompiler_timeout,
    )

    engine.setup_project(binary_paths, project_path, project_name, symbols_path, gzfs_path)
    engine.analyze_project()

    if args.names:
        engine.apply_canonical_names(binary_paths[0], binary_paths[-1])

    diffs = [(binary_paths[i], binary_paths[i + 1]) for i in range(len(binary_paths) - 1)]

    for old, new in diffs:
        pdiff = engine.diff_bins(old, new)

        # The corpus covers a handful of functions; drop the global
        # added/deleted symbol section (thousands of renamed symbols are
        # report noise). Keep strings: literal-content diffs are signal.
        pdiff["symbols"] = {"added": [], "deleted": []}

        engine.validate_diff_json(json.dumps(pdiff))
        diff_name = f"{old.name}-{new.name}.ghidriff"
        engine.dump_pdiff_to_path(
            diff_name,
            pdiff,
            output_path,
            side_by_side=args.side_by_side,
            max_section_funcs=args.max_section_funcs,
            md_title=args.md_title,
        )


if __name__ == "__main__":
    main()
