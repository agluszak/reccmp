"""Direct-call observations in Ghidra function bodies, for mismatch triage only.

This does not compare instructions or establish equivalence. Sequences are in
address order, not execution order; indirect calls are excluded. Ghidra owns
body boundaries; catalog maximum extents are not assumed to be function bodies.
"""

from collections import Counter
from difflib import SequenceMatcher
from typing import Any

from reccmp.compare.manifest import Manifest

from reccmp.analysis.x86 import decode_one, direct_call_target
from reccmp.types import ImageId


def direct_call_census(manifest: Manifest, programs: dict[ImageId, Any]) -> dict:
    """Read Ghidra-owned function bodies using catalog-owned identities."""
    functions = []
    identities = {image: {obj.addr(image): obj for obj in manifest.objects} for image in ImageId}
    pairs = {obj.orig_addr: obj for obj in manifest.objects}
    for alias in manifest.aliases:
        if alias.canonical_orig in pairs:
            identities[alias.image_id][alias.addr] = pairs[alias.canonical_orig]
    for entry in manifest.functions:
        if entry.recomp_addr is None:
            continue
        sides: dict[str, Any] = {}
        for image, program in programs.items():
            address = entry.orig_addr if image == ImageId.ORIG else entry.recomp_addr
            space = program.getAddressFactory().getDefaultAddressSpace()
            function = program.getFunctionManager().getFunctionAt(space.getAddress(address))
            side: dict[str, Any] = {"calls": None}
            sides[image.name.lower()] = side
            if function is None:
                side["error"] = "no-function-at-entry"
                continue
            body = function.getBody()
            side["body_ranges"] = [
                [str(r.getMinAddress()), str(r.getMaxAddress())] for r in body.getAddressRanges()
            ]
            calls = []
            for instruction in program.getListing().getInstructions(body, True):
                if not instruction.getFlowType().isCall():
                    continue
                # Ghidra may resolve an indirect call as a reference; exclude
                # it from the direct-call sequence nonetheless.
                site = int(instruction.getAddress().getOffset())
                decoded = decode_one(bytes(int(b) & 255 for b in instruction.getBytes()), site)
                target = direct_call_target(decoded) if decoded is not None else None
                if target is None:
                    continue
                obj = identities[image].get(target)
                calls.append(
                    {
                        "site": f"{site:#x}",
                        "target": f"{target:#x}",
                        "identity": f"pair:{obj.orig_addr:#x}"
                        if obj
                        else f"{image.name.lower()}:{target:#x}",
                        "paired": obj is not None,
                        "name": obj.name if obj else None,
                    }
                )
            side["calls"] = calls
        functions.append({"address": f"{entry.orig_addr:#x}", **sides})
    return {
        "manifest_sha256": manifest.digest(),
        "functions": functions,
        "scope": "direct calls in Ghidra function bodies, in address order; excludes indirect calls and tail jumps",
    }


def call_delta(original: list[dict], rebuilt: list[dict]) -> dict:
    old = [call["identity"] for call in original]
    new = [call["identity"] for call in rebuilt]
    unresolved = any(not call["paired"] for call in original + rebuilt)
    deltas = [
        {"retail": old[i:j], "rebuild": new[k:l]}
        for tag, i, j, k, l in SequenceMatcher(None, old, new, autojunk=False).get_opcodes()
        if tag != "equal"
    ]
    removed = sum(len(delta["retail"]) for delta in deltas)
    added = sum(len(delta["rebuild"]) for delta in deltas)
    if unresolved:
        category = "unresolved-target"
    elif old == new:
        category = "identical-canonical-sequence"
    elif Counter(old) == Counter(new):
        category = "same-callees-reordered"
    elif added == 0:
        category = "retail-extra-direct-calls"
    elif removed == 0:
        category = "rebuild-extra-direct-calls"
    elif removed == added == 1:
        category = "one-canonical-callee-differs"
    else:
        category = "several-canonical-callees-differ"
    return {"category": category, "deltas": deltas, "unresolved": unresolved}
