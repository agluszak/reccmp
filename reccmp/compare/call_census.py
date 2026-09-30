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


def _body_calls(program, image, address, identities, names):
    space = program.getAddressFactory().getDefaultAddressSpace()
    function = program.getFunctionManager().getFunctionAt(space.getAddress(address))
    if function is None:
        return {"calls": None, "error": "no-function-at-entry"}
    body = function.getBody()
    calls = []
    outside_fallthroughs = []
    for instruction in program.getListing().getInstructions(body, True):
        fallthrough = instruction.getFallThrough()
        if fallthrough is not None and not body.contains(fallthrough):
            outside_fallthroughs.append(
                {"site": str(instruction.getAddress()), "target": str(fallthrough)}
            )
        if not instruction.getFlowType().isCall():
            continue
        # A Ghidra reference resolving an indirect call is not a direct call.
        site = int(instruction.getAddress().getOffset())
        decoded = decode_one(bytes(int(b) & 255 for b in instruction.getBytes()), site)
        target = direct_call_target(decoded) if decoded is not None else None
        if target is None:
            continue
        obj = identities.get(target)
        calls.append(
            {
                "site": f"{site:#x}",
                "target": f"{target:#x}",
                "identity": (
                    f"pair:{obj.orig_addr:#x}"
                    if obj
                    else f"{image.name.lower()}:{target:#x}"
                ),
                "paired": obj is not None,
                "name": obj.name if obj else names.get(target),
            }
        )
    observation: dict[str, Any] = {
        "body_ranges": [
            [str(r.getMinAddress()), str(r.getMaxAddress())]
            for r in body.getAddressRanges()
        ],
        "calls": calls,
    }
    if outside_fallthroughs:
        # A shared/mid-body entry may have split the native FunctionDB body.
        # The observed prefix is not a complete authored-function sequence.
        observation.update(
            calls=None,
            partial_calls=calls,
            error="function-body-has-outside-fallthrough",
            outside_fallthroughs=outside_fallthroughs,
        )
    return observation


def direct_call_census(manifest: Manifest, programs: dict[ImageId, Any]) -> dict:
    """Read Ghidra-owned function bodies using catalog-owned identities."""
    functions = []
    identities = {
        image: {obj.addr(image): obj for obj in manifest.objects} for image in ImageId
    }
    names = {
        image: {
            obj.addr: obj.name for obj in manifest.unpaired if obj.image_id == image
        }
        for image in ImageId
    }
    pairs = {obj.orig_addr: obj for obj in manifest.objects}
    for alias in manifest.aliases:
        if alias.canonical_orig in pairs:
            identities[alias.image_id][alias.addr] = pairs[alias.canonical_orig]
    bodies: dict[ImageId, dict[int, dict[str, Any]]] = {image: {} for image in programs}
    for entry in manifest.functions:
        if entry.recomp_addr is None:
            continue
        sides = {}
        for image, program in programs.items():
            address = entry.orig_addr if image == ImageId.ORIG else entry.recomp_addr
            observation = _body_calls(
                program, image, address, identities[image], names[image]
            )
            bodies[image][address] = observation
            sides[image.name.lower()] = observation
        functions.append({"address": f"{entry.orig_addr:#x}", **sides})
    # Read only targets already observed in these callers, not a second whole-
    # binary function inventory. Unpaired template emissions can still have
    # known retail bodies; no counterpart or equivalence is inferred for them.
    targets = []
    for image, program in programs.items():
        addresses = {
            int(call["target"], 16)
            for body in bodies[image].values()
            for call in body["calls"] or []
        }
        for address in sorted(addresses):
            body = bodies[image].get(address)
            if body is None:
                body = _body_calls(
                    program, image, address, identities[image], names[image]
                )
            obj = identities[image].get(address)
            targets.append(
                {
                    "image": image.name.lower(),
                    "address": f"{address:#x}",
                    "identity": (
                        f"pair:{obj.orig_addr:#x}"
                        if obj
                        else f"{image.name.lower()}:{address:#x}"
                    ),
                    "name": obj.name if obj else names[image].get(address),
                    **body,
                }
            )
    return {
        "manifest_sha256": manifest.digest(),
        "functions": functions,
        "targets": targets,
        "scope": "direct calls in Ghidra function bodies, in address order; excludes indirect calls and tail jumps",
    }


def call_delta(original: list[dict], rebuilt: list[dict]) -> dict:
    old = [call["identity"] for call in original]
    new = [call["identity"] for call in rebuilt]
    unresolved = any(not call["paired"] for call in original + rebuilt)
    deltas = [
        {"retail": old[i:j], "rebuild": new[k:l]}
        for tag, i, j, k, l in SequenceMatcher(
            None, old, new, autojunk=False
        ).get_opcodes()
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
