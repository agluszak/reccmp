"""Part of the core analysis/comparison logic of `reccmp`.
These functions update the entity database based on analysis of the binary files.
"""

import logging
import re
import struct
from typing import Mapping
from reccmp.formats import Image, PEImage
from reccmp.formats.exceptions import (
    InvalidVirtualAddressError,
    InvalidVirtualReadError,
    InvalidStringError,
)
from reccmp.types import EntityType, ImageId
from reccmp.parser import DecompCodebase
from reccmp.parser.marker import MarkerType
from reccmp.analysis import (
    find_float_consts,
    find_import_thunks,
    find_vtordisp,
    find_eh_handlers,
    find_exception_registrations,
    is_likely_latin1,
    is_likely_widechar,
)
from reccmp.analysis.x86 import code_signature, direct_call_target, instructions
from reccmp.analysis.crt_startup import (
    detect_crt_startup_arrays,
    get_crt_function_name,
)
from .db import EntityDb, PairBasis, ReccmpEntity, entity_name_from_string
from .queries import get_floats_without_data, get_strings_without_data
from .thunk_resolve import is_plausible_vtable_target

logger = logging.getLogger(__name__)


def import_sections(db: EntityDb, image_id: ImageId, binfile: Image):
    assert image_id in (ImageId.ORIG, ImageId.RECOMP), "Invalid image id"

    for sect in binfile.sections:
        db.add_section(image_id, sect.virtual_range)


def match_entry(db: EntityDb, orig_bin: PEImage, recomp_bin: PEImage):
    # The _entry symbol is referenced in the PE header so we get this match for free.
    # AddressOfEntryPoint == 0 means "no entry point" (common for DLLs); the
    # image base itself is not a function.
    if (
        orig_bin.optional_header.address_of_entry_point == 0
        or recomp_bin.optional_header.address_of_entry_point == 0
    ):
        return

    with db.batch() as batch:
        batch.set(ImageId.RECOMP, recomp_bin.entry, type=EntityType.FUNCTION)
        batch.match(orig_bin.entry, recomp_bin.entry, basis=PairBasis.DERIVED)


def create_crt_functions(db: EntityDb, image_id: ImageId, binfile: PEImage):
    """Create entities for all functions found to be part of the CRT array.
    This includes any functions (thunks) called indirectly."""
    crt_arrays = detect_crt_startup_arrays(db, image_id, binfile)

    with db.batch() as batch:
        for array_type, array in crt_arrays.items():
            # All entities get the same base name.
            # We could use more specific names when we have more confidence in the format.
            # e.g. "atexit_setter"
            base_name = get_crt_function_name(array_type)
            for addr in [*array.functions, *array.thunks.values()]:
                batch.set(
                    image_id,
                    addr,
                    type=EntityType.FUNCTION,
                    name=base_name,
                )


def create_analysis_widechars(db: EntityDb, img_id: ImageId, binfile: PEImage):
    """Search both binaries for UTF-16LE strings at relocation targets.

    Must run before create_analysis_strings: a Latin1 scan would otherwise
    truncate wide strings at the first embedded NUL (e.g. L\"F1\" -> \"F\").

    Only accept a wide decode when it continues past the Latin1 truncation,
    so a genuine narrow \"F\" is not re-labeled as L\"F\".
    """
    with db.batch() as batch:
        last_range = range(0)
        for addr, string in binfile.iter_widechar():
            if addr in binfile.relocations:
                continue

            if addr in last_range:
                continue

            try:
                narrow = binfile.read_string(addr).decode("latin1")
            except (InvalidStringError, UnicodeDecodeError, InvalidVirtualAddressError):
                narrow = None

            # Genuine Latin1 strings re-decoded as UTF-16LE of equal length
            # (e.g. \"F\") are not wide strings.
            if narrow is not None and len(string) <= len(narrow):
                continue

            if is_likely_widechar(string) and not db.intersects(img_id, addr):
                # Size includes the 2-byte UTF-16 null terminator.
                size = 2 * len(string) + 2
                batch.set(
                    img_id,
                    addr,
                    type=EntityType.WIDECHAR,
                    name=entity_name_from_string(string, wide=True),
                    size=size,
                )
                last_range = range(addr, addr + size)


def create_analysis_strings(
    db: EntityDb, img_id: ImageId, binfile: PEImage, encoding: str = "latin1"
):
    """Search both binaries for Latin1 strings.
    We use the insert_() method so that these strings will not overwrite
    an existing entity. It's possible that some variables or pointers
    will be mistakenly identified as short strings."""
    with db.batch() as batch:
        last_range = range(0)
        for addr, string in binfile.iter_string(encoding):
            # If the address is the site of a relocation, this is a pointer, not a string.
            if addr in binfile.relocations:
                continue

            # Don't create an entity for a substring of
            # the most recently created string.
            if addr in last_range:
                continue

            if is_likely_latin1(string) and not db.intersects(img_id, addr):
                batch.set(
                    img_id,
                    addr,
                    type=EntityType.STRING,
                    name=entity_name_from_string(string),
                    size=len(string) + 1,  # including null-terminator
                )
                last_range = range(addr, addr + len(string) + 1)


def create_analysis_floats(
    db: EntityDb,
    img_id: ImageId,
    binfile: PEImage,
    write_permissions: Mapping[str, bool] | None = None,
):
    """Add floating point constants in each binary to the database.
    We are not matching anything right now because these values are not
    deduped like strings. `write_permissions` overrides the binary's own
    section permissions when deciding which data is constant."""
    with db.batch() as batch:
        for addr, size, float_value in find_float_consts(binfile, write_permissions):
            if not db.intersects(img_id, addr):
                batch.set(
                    img_id,
                    addr,
                    type=EntityType.FLOAT,
                    name=str(float_value),
                    size=size,
                )


def _seh_side_key(img_id: ImageId, field: str) -> str:
    side = "orig" if img_id == ImageId.ORIG else "recomp"
    return f"seh_{field}_{side}"


def _find_exception_owner(
    db: EntityDb, img_id: ImageId, registration_addr: int
) -> int | None:
    """Relate a registration site to a function only near its entry point."""
    owner = None
    for entity in db.all(img_id):
        addr = entity.addr(img_id)
        assert addr is not None
        if addr > registration_addr:
            break
        if entity.get("type") == EntityType.FUNCTION:
            owner = addr

    # VC5's inline form can do a few loads between entry and ``push handler``.
    if owner is None or registration_addr - owner > 32:
        return None
    return owner


def create_seh_entities(db: EntityDb, img_id: ImageId, binfile: PEImage):
    """Create SEH entities and record their structural relationships."""
    handlers = tuple(find_eh_handlers(binfile))
    registrations: dict[int, list[int]] = {}
    for registration in find_exception_registrations(binfile, handlers):
        registrations.setdefault(registration.handler_addr, []).append(
            registration.addr
        )

    with db.batch() as batch:
        for handler_addr, funcinfo in handlers:
            handler_fields = {
                _seh_side_key(img_id, "funcinfo"): funcinfo.addr,
            }
            sites = registrations.get(handler_addr, [])
            if len(sites) == 1:
                owner = _find_exception_owner(db, img_id, sites[0])
                if owner is not None:
                    handler_fields[_seh_side_key(img_id, "owner")] = owner

            # Using names derived from symbols in .cpp.s generated asm.
            batch.set(
                img_id,
                handler_addr,
                type=EntityType.LABEL,
                name="__ehhandler",
                **handler_fields,
            )
            if img_id == ImageId.ORIG:
                batch.set(
                    img_id,
                    funcinfo.addr,
                    type=EntityType.DATA,
                    name="__ehfuncinfo",
                    seh_unwinds_orig=tuple(funcinfo.unwinds),
                )
            else:
                batch.set(
                    img_id,
                    funcinfo.addr,
                    type=EntityType.DATA,
                    name="__ehfuncinfo",
                    seh_unwinds_recomp=tuple(funcinfo.unwinds),
                )

            for unwind in funcinfo.unwinds:
                if unwind.action_addr != 0:
                    batch.set(
                        img_id,
                        unwind.action_addr,
                        type=EntityType.LABEL,
                        name=f"__Unwind({unwind.target_state})",
                    )


def create_imports(db: EntityDb, image_id: ImageId, binfile: Image):
    with db.batch() as batch:
        for imp in binfile.imports:
            if imp.name:
                import_name = f"{imp.module}::{imp.name}"
            else:
                import_name = f"{imp.module}::Ordinal_{imp.ordinal}"

            batch.set(
                image_id,
                imp.addr,
                name=import_name,
                import_module=imp.module,
                import_name=imp.name or None,
                size=4,
                type=EntityType.IMPORT,
            )


def create_import_thunks(db: EntityDb, image_id: ImageId, binfile: Image):
    if not isinstance(binfile, PEImage):
        return

    function_starts = {
        addr
        for entity in db.get_all()
        if entity.get("type") == EntityType.FUNCTION
        and entity.size(image_id) == 6
        and (addr := entity.addr(image_id)) is not None
    }

    with db.batch() as batch:
        for thunk in find_import_thunks(binfile, function_starts):
            entity = db.get(image_id, thunk.addr)
            if (
                entity is not None
                and entity.orig_addr is not None
                and entity.recomp_addr is not None
                and entity.get("type") == EntityType.FUNCTION
                and db.pair_basis(entity.orig_addr) == PairBasis.ANNOTATION
            ):
                # A source-identified function remains comparable even when
                # the linker emits its entire body as a direct import jump.
                continue
            batch.set(
                image_id,
                thunk.addr,
                type=EntityType.IMPORT_THUNK,
                skip=True,
                size=thunk.size,
            )
            batch.set_ref(image_id, thunk.addr, ref=thunk.import_addr)


def create_thunks(db: EntityDb, img_id: ImageId, binfile: PEImage):
    """Create entities for any thunk functions in the image.
    These are the result of an incremental build."""
    with db.batch() as batch:
        for thunk_addr, func_addr in binfile.thunks:
            if not db.exists(img_id, thunk_addr):
                batch.set(
                    img_id,
                    thunk_addr,
                    type=EntityType.THUNK,
                    size=5,
                    skip=True,
                )
                batch.set_ref(img_id, thunk_addr, ref=func_addr)

            # We can only match two thunks if we have already matched both
            # their parent entities. There is nothing to compare because
            # they will either be equal or left unmatched. Set skip=True.


def match_exports(db: EntityDb, orig_bin: PEImage, recomp_bin: PEImage):
    # invert for name lookup
    orig_exports = {y: x for (x, y) in orig_bin.exports}

    orig_thunks = dict(orig_bin.thunks)
    recomp_thunks = dict(recomp_bin.thunks)

    with db.batch() as batch:
        for recomp_addr, export_name in recomp_bin.exports:
            orig_addr = orig_exports.get(export_name)
            if orig_addr is None:
                continue

            # Check whether either of the addresses is actually a thunk.
            # This is a quirk of the debug builds. Technically the export
            # *is* the thunk, but it's more helpful to mark the actual function.
            # It could be the case that only one side is a thunk, but we can
            # deal with that.
            if orig_addr in orig_thunks:
                orig_addr = orig_thunks[orig_addr]

            if recomp_addr in recomp_thunks:
                recomp_addr = recomp_thunks[recomp_addr]

            batch.match(orig_addr, recomp_addr, basis=PairBasis.DERIVED)


def create_analysis_vtordisps(db: EntityDb, img_id: ImageId, binfile: PEImage):
    """Creates entities for each detected vtordisp function in the image.
    The critical step is to set the 'vtordisp' attribute to True, which distinguishes
    these entities from others (i.e. thunks) that have the 'ref_' attribute set."""
    with db.batch() as batch:
        for vtor in find_vtordisp(binfile):
            batch.set(
                img_id,
                vtor.addr,
                type=EntityType.VTORDISP,
                size=vtor.size,
            )
            batch.set_ref(
                img_id, vtor.addr, displacement=vtor.displacement, ref=vtor.func_addr
            )

            # Create an entity for the referenced function, but do not overwrite an existing entity (for now).
            if not db.exists(img_id, vtor.func_addr):
                batch.set(img_id, vtor.func_addr, type=EntityType.FUNCTION)


def complete_partial_floats(db: EntityDb, image_id: ImageId, binfile: PEImage):
    """For each float entity without any data,
    read the value from the binary and set the entity name."""
    assert image_id in (ImageId.ORIG, ImageId.RECOMP), "Invalid image id"

    with db.batch() as batch:
        for addr, is_double in get_floats_without_data(db, image_id):
            try:
                if is_double:
                    (float_value,) = struct.unpack("<d", binfile.read(addr, 8))
                else:
                    (float_value,) = struct.unpack("<f", binfile.read(addr, 4))

                batch.set(image_id, addr, name=str(float_value))
            except (InvalidVirtualReadError, InvalidVirtualAddressError):
                logger.error(
                    "Failed to read %s from %s at 0x%x",
                    ("double" if is_double else "float"),
                    image_id.name.lower(),
                    addr,
                )


def complete_partial_strings(
    db: EntityDb, image_id: ImageId, binfile: PEImage, encoding: str = "latin1"
):
    """For each string/widechar entity without any data,
    read the value from the binary and set the entity name.
    If the entity has no size, read until we hit a null-terminator."""
    assert image_id in (ImageId.ORIG, ImageId.RECOMP), "Invalid image id"

    with db.batch() as batch:
        for addr, string_size, is_widechar in get_strings_without_data(db, image_id):
            try:
                if is_widechar:
                    if string_size is not None:
                        # Remove 2-byte null-terminator before decoding
                        raw = binfile.read(addr, string_size)[:-2]
                    else:
                        raw = binfile.read_widechar(addr)
                        string_size = len(raw) + 2

                    decoded_string = raw.decode("utf-16-le")
                else:
                    if string_size is not None:
                        # Remove 1-byte null-terminator before decoding
                        raw = binfile.read(addr, string_size)[:-1]
                    else:
                        raw = binfile.read_string(addr)
                        string_size = len(raw) + 1

                    decoded_string = raw.decode(encoding)

                batch.set(
                    image_id,
                    addr,
                    name=entity_name_from_string(decoded_string, is_widechar),
                    size=string_size,
                )

            except (
                InvalidVirtualReadError,
                InvalidStringError,
                InvalidVirtualAddressError,
            ):
                logger.error(
                    "Failed to read %s from %s at 0x%x",
                    ("widechar" if is_widechar else "string"),
                    image_id.name.lower(),
                    addr,
                )
            except UnicodeDecodeError:
                logger.error(
                    "Could not decode %s from %s at 0x%x",
                    ("widechar" if is_widechar else "string"),
                    image_id.name.lower(),
                    addr,
                )


def normalize_original_zero_size_data(db: EntityDb, binfile: PEImage) -> None:
    """Retype structurally proven zero-size inventory rows conservatively.

    Ghidra exports may describe interior code labels, E9 islands and named vtables as
    generic DATA. Function containment, exact opcodes and pointer-run/name agreement
    are sufficient type evidence; all other rows remain DATA for manual xref work.
    """
    functions: list[tuple[int, int]] = []
    code_ranges = [region.range for region in binfile.get_code_regions()]
    for entity in db.all(ImageId.ORIG):
        if entity.get("type") != EntityType.FUNCTION:
            continue
        addr = entity.orig_addr
        size = entity.size(ImageId.ORIG)
        if addr is not None and size is not None and size > 0:
            functions.append((addr, addr + size))
    functions.sort()

    def containing_function(addr: int) -> bool:
        for start, end in functions:
            if start >= addr:
                return False
            if addr < end:
                return True
        return False

    def in_code(addr: int) -> bool:
        return any(addr in region for region in code_ranges)

    with db.batch() as batch:
        for entity in tuple(db.unmatched(ImageId.ORIG)):
            if entity.get("type") != EntityType.DATA or entity.size(ImageId.ORIG):
                continue
            addr = entity.orig_addr
            if addr is None:
                continue
            if containing_function(addr):
                batch.set(ImageId.ORIG, addr, type=EntityType.LABEL)
                continue

            if in_code(addr):
                raw = binfile.read(addr, 5)
                if raw[0] == 0xE9:
                    target = addr + 5 + int.from_bytes(raw[1:5], "little", signed=True)
                    batch.set(
                        ImageId.ORIG,
                        addr,
                        type=EntityType.THUNK,
                        size=5,
                        skip=True,
                    )
                    batch.set_ref(ImageId.ORIG, addr, ref=target)
                continue

            name = entity.best_name() or ""
            match = re.fullmatch(r"(.+?)::(?:'vftable'|vftable)", name)
            if match is None:
                continue
            slot_count = 0
            for offset in range(0, 1024, 4):
                (target,) = struct.unpack("<I", binfile.read(addr + offset, 4))
                if not in_code(target):
                    break
                slot_count += 1
            if slot_count >= 3:
                batch.set(
                    ImageId.ORIG,
                    addr,
                    type=EntityType.VTABLE,
                    name=match.group(1),
                    size=slot_count * 4,
                    inferred_vtable=True,
                )


def _vtable_class_name(name: str | None) -> str | None:
    if not name:
        return None
    match = re.match(r"(.+?)::`vftable'", name)
    return match.group(1) if match else name


def _vtable_slot_identities(
    db: EntityDb, image_id: ImageId, binfile: PEImage, addr: int, size: int
) -> tuple[int | None, ...] | None:
    try:
        raw = binfile.read(addr, size)
    except (InvalidVirtualAddressError, InvalidVirtualReadError):
        return None
    identities: list[int | None] = []
    for (target,) in struct.iter_unpack("<I", raw):
        if target == 0:
            identities.append(None)
            continue
        canonical = db.alias_canonical_orig(image_id, target)
        if canonical is None:
            return None
        identities.append(canonical)
    return tuple(identities)


def match_inferred_vtables_by_slots(
    db: EntityDb, orig_bin: PEImage, recomp_bin: PEImage
) -> None:
    """Pair inferred retail vtables only through exact canonical slot identities."""
    recomp_by_class: dict[str, list[ReccmpEntity]] = {}
    for entity in db.unexplained(ImageId.RECOMP):
        if entity.get("type") != EntityType.VTABLE:
            continue
        class_name = _vtable_class_name(entity.best_name())
        if class_name is not None:
            recomp_by_class.setdefault(class_name, []).append(entity)

    pairs: list[tuple[int, int]] = []
    for original in db.unexplained(ImageId.ORIG):
        if not original.get("inferred_vtable"):
            continue
        orig_addr = original.orig_addr
        orig_size = original.size(ImageId.ORIG)
        class_name = _vtable_class_name(original.best_name())
        if orig_addr is None or orig_size is None or class_name is None:
            continue
        orig_slots = _vtable_slot_identities(
            db, ImageId.ORIG, orig_bin, orig_addr, orig_size
        )
        if orig_slots is None:
            continue
        equivalent: list[int] = []
        for recomp in recomp_by_class.get(class_name, []):
            recomp_addr = recomp.recomp_addr
            recomp_size = recomp.size(ImageId.RECOMP)
            if (
                recomp_addr is None
                or recomp_size != orig_size
                or _vtable_slot_identities(
                    db, ImageId.RECOMP, recomp_bin, recomp_addr, recomp_size
                )
                != orig_slots
            ):
                continue
            equivalent.append(recomp_addr)
        if len(equivalent) == 1:
            pairs.append((orig_addr, equivalent[0]))
    db.bulk_match(pairs, basis=PairBasis.DERIVED)


def _function_signature(
    binfile: PEImage, addr: int | None, size: int | None
) -> tuple | None:
    if addr is None or size is None or size <= 0:
        return None
    try:
        code = bytes(binfile.read(addr, size))
    except (InvalidVirtualAddressError, InvalidVirtualReadError):
        return None
    return code_signature(code, addr)


def classify_folded_function_aliases(
    db: EntityDb, orig_bin: PEImage, recomp_bin: PEImage
) -> None:
    """Record recompiled functions the original's identical-code folding kept
    as one body.

    The original links with ICF: functions whose code is the same are one
    body at one address. The recompiled image links without it, so a
    function the catalog cannot explain whose code is the same as exactly
    one paired function's (ILLength and PLLength) was folded into that pair
    in the original, and takes its identity. FOLDED annotations say the
    same for the cases they name.

    Folding covers only packaged functions: the original keeps several
    empty bodies apart, for one. When another function of the original
    starts with the pair's code, the folded copy could be either, so it
    keeps no identity."""
    canonical: dict[tuple, set[int]] = {}
    for pair in db.get_matches_by_type(EntityType.FUNCTION):
        signature = _function_signature(
            recomp_bin, pair.recomp_addr, pair.size(ImageId.RECOMP)
        )
        if signature:
            canonical.setdefault(signature, set()).add(pair.orig_addr)

    folded: dict[int, list[tuple[int, int]]] = {}
    for candidate in tuple(db.unexplained(ImageId.RECOMP)):
        if candidate.get("type") != EntityType.FUNCTION:
            continue
        addr = candidate.addr(ImageId.RECOMP)
        size = candidate.size(ImageId.RECOMP)
        signature = _function_signature(recomp_bin, addr, size)
        identities = canonical.get(signature, set()) if signature else set()
        if addr is not None and size is not None and len(identities) == 1:
            folded.setdefault(size, []).append((addr, next(iter(identities))))
    if not folded:
        return

    originals = [
        addr
        for entity in db.get_all()
        if entity.get("type") == EntityType.FUNCTION
        and (addr := entity.addr(ImageId.ORIG)) is not None
    ]
    for size, candidates in folded.items():
        # Original bodies at this length, compared within the original.
        bodies: dict[tuple, int] = {}
        for addr in originals:
            signature = _function_signature(orig_bin, addr, size)
            if signature:
                bodies[signature] = bodies.get(signature, 0) + 1
        for addr, orig_addr in candidates:
            signature = _function_signature(orig_bin, orig_addr, size)
            if signature and bodies.get(signature) == 1:
                db.set_alias(ImageId.RECOMP, addr, orig_addr)


def classify_synthetic_jump_aliases(
    db: EntityDb, orig_bin: PEImage, codebase: DecompCodebase
) -> None:
    """Give annotated retail jump thunks the identity of their paired target.

    A SYNTHETIC marker identifies compiler/linker emission, not a second
    authored function. Require the entire meaningful instruction to be a
    direct near jump; an ordinary source forwarding function is not an alias.
    """
    for marker in codebase.iter_name_functions():
        if marker.type != MarkerType.SYNTHETIC:
            continue
        addr = marker.offset
        candidate = db.get(ImageId.ORIG, addr)
        if (
            candidate is None
            or candidate.get("type") != EntityType.FUNCTION
            or candidate.recomp_addr is not None
        ):
            continue
        try:
            code = bytes(orig_bin.read(addr, 5))
        except (InvalidVirtualAddressError, InvalidVirtualReadError):
            continue
        if len(code) != 5 or code[0] != 0xE9:
            continue
        target = addr + 5 + struct.unpack("<i", code[1:])[0]
        paired = db.get_one_match(target)
        if paired is not None and paired.get("type") == EntityType.FUNCTION:
            db.set_alias(ImageId.ORIG, addr, target)


def _direct_calls(
    binfile: PEImage, addr: int, size: int | None
) -> tuple[int, ...] | None:
    if size is None or size <= 0 or size > 10000:
        return None
    try:
        code = bytes(binfile.read(addr, size))
    except (InvalidVirtualAddressError, InvalidVirtualReadError):
        return None
    return tuple(
        target
        for insn in instructions(code, addr)
        if (target := direct_call_target(insn)) is not None
    )


def _trailing_called_funclet(
    binfile: PEImage, addr: int, size: int | None
) -> tuple[tuple[int, ...], int] | None:
    """Find a local call target immediately following the parent's return.

    VC6 PDB function extents can include an adjacent unwind funclet. Its own
    calls belong to the funclet, not the enclosing source function.
    """
    if size is None or size <= 0 or size > 10000:
        return None
    try:
        code = bytes(binfile.read(addr, size))
    except (InvalidVirtualAddressError, InvalidVirtualReadError):
        return None
    calls: list[int] = []
    previous = None
    for insn in instructions(code, addr):
        if (
            insn.address in calls
            and previous is not None
            and previous.mnemonic.startswith("ret")
        ):
            return tuple(calls), insn.address
        if (target := direct_call_target(insn)) is not None:
            calls.append(target)
        previous = insn
    return None


def match_unpaired_direct_callees(
    db: EntityDb,
    orig_bin: PEImage,
    recomp_bin: PEImage,
    codebase: DecompCodebase | None = None,
) -> None:
    """Derive an unnamed callee from the same calls in independent paired bodies.

    Require whole direct-call sequences to align, including already paired
    callees. Two distinct paired callers must agree on one original/rebuild
    target and neither target may have another candidate. A marked synthetic
    funclet may instead use one parent when the rebuild PDB folds its body
    into that parent's extent immediately after a return. This recovers
    private library helpers absent from the rebuild PDB without pairing on
    a common short body or a rendered function name.
    """
    synthetic = (
        {
            marker.offset
            for marker in codebase.iter_name_functions()
            if marker.type == MarkerType.SYNTHETIC
        }
        if codebase is not None
        else set()
    )
    candidates = {
        entity.orig_addr
        for entity in db.unexplained(ImageId.ORIG)
        if entity.get("type") == EntityType.FUNCTION
        and entity.orig_addr is not None
        and entity.best_name() is not None
    }
    observations: dict[int, dict[int, set[int]]] = {}
    trailing_funclets: set[tuple[int, int]] = set()
    for parent in db.get_matches_by_type(EntityType.FUNCTION):
        if db.pair_basis(parent.orig_addr) != PairBasis.ANNOTATION:
            continue
        orig_calls = _direct_calls(
            orig_bin,
            parent.orig_addr,
            parent.size(ImageId.ORIG) or parent.fact(ImageId.ORIG, "orig_max_size"),
        )
        if not orig_calls or not candidates.intersection(orig_calls):
            continue
        recomp_calls = _direct_calls(
            recomp_bin, parent.recomp_addr, parent.size(ImageId.RECOMP)
        )
        if recomp_calls is None:
            continue
        if len(orig_calls) != len(recomp_calls):
            # A single marked retail funclet may be included in its rebuild
            # parent's PDB extent. Require the local target to begin directly
            # after a return and every parent call before it to align.
            if len(orig_calls) != 1 or orig_calls[0] not in synthetic:
                continue
            trailing = _trailing_called_funclet(
                recomp_bin, parent.recomp_addr, parent.size(ImageId.RECOMP)
            )
            if trailing is None or trailing[0] != (trailing[1],):
                continue
            recomp_calls = trailing[0]
            trailing_funclets.add((orig_calls[0], recomp_calls[0]))
        proposed: set[tuple[int, int]] = set()
        for orig_target, recomp_target in zip(orig_calls, recomp_calls):
            orig_identity = db.alias_canonical_orig(ImageId.ORIG, orig_target)
            recomp_identity = db.alias_canonical_orig(ImageId.RECOMP, recomp_target)
            if orig_identity is not None or recomp_identity is not None:
                if orig_identity != recomp_identity:
                    break
            elif (
                orig_target in candidates
                and db.get(ImageId.RECOMP, recomp_target) is None
                and is_plausible_vtable_target(recomp_bin, recomp_target)
            ):
                proposed.add((orig_target, recomp_target))
            else:
                break
        else:
            for orig_target, recomp_target in proposed:
                observations.setdefault(orig_target, {}).setdefault(
                    recomp_target, set()
                ).add(parent.orig_addr)

    reverse: dict[int, set[int]] = {}
    for orig_target, targets in observations.items():
        for recomp_target in targets:
            reverse.setdefault(recomp_target, set()).add(orig_target)
    with db.batch() as batch:
        for orig_target, targets in observations.items():
            if len(targets) != 1:
                continue
            recomp_target, parents = next(iter(targets.items()))
            if (
                len(parents) >= 2 or (orig_target, recomp_target) in trailing_funclets
            ) and reverse[recomp_target] == {orig_target}:
                batch.match(orig_target, recomp_target, basis=PairBasis.DERIVED)


def classify_exact_vtable_aliases(
    db: EntityDb, orig_bin: PEImage, recomp_bin: PEImage
) -> None:
    """Record exact duplicate vtable emissions against a unique canonical pair."""
    for image_id, binfile in (
        (ImageId.ORIG, orig_bin),
        (ImageId.RECOMP, recomp_bin),
    ):
        canonical: dict[tuple[str, bytes], set[int]] = {}
        for canonical_entity in db.get_matches_by_type(EntityType.VTABLE):
            addr = canonical_entity.addr(image_id)
            size = canonical_entity.size(image_id)
            name = canonical_entity.best_name()
            if addr is None or size is None or size <= 0 or name is None:
                continue
            try:
                raw = bytes(binfile.read(addr, size))
            except (InvalidVirtualAddressError, InvalidVirtualReadError):
                continue
            canonical.setdefault((name, raw), set()).add(canonical_entity.orig_addr)

        for candidate in tuple(db.unexplained(image_id)):
            if candidate.get("type") != EntityType.VTABLE:
                continue
            addr = candidate.addr(image_id)
            size = candidate.size(image_id)
            name = candidate.best_name()
            if addr is None or size is None or size <= 0 or name is None:
                continue
            try:
                raw = bytes(binfile.read(addr, size))
            except (InvalidVirtualAddressError, InvalidVirtualReadError):
                continue
            identities = canonical.get((name, raw), set())
            if len(identities) == 1:
                db.set_alias(image_id, addr, next(iter(identities)))


def classify_folded_vtable_aliases(
    db: EntityDb, orig_bin: PEImage, recomp_bin: PEImage
) -> None:
    """Link distinct rebuild tables only when all slots identify one retail table."""
    originals: dict[tuple[int, tuple[int | None, ...]], set[int]] = {}
    for entity in db.get_all():
        addr = entity.addr(ImageId.ORIG)
        size = entity.size(ImageId.ORIG)
        if entity.get("type") != EntityType.VTABLE or addr is None or size is None:
            continue
        slots = _vtable_slot_identities(db, ImageId.ORIG, orig_bin, addr, size)
        if slots is not None:
            originals.setdefault((size, slots), set()).add(addr)

    for candidate in tuple(db.unexplained(ImageId.RECOMP)):
        addr = candidate.addr(ImageId.RECOMP)
        size = candidate.size(ImageId.RECOMP)
        if candidate.get("type") != EntityType.VTABLE or addr is None or size is None:
            continue
        slots = _vtable_slot_identities(db, ImageId.RECOMP, recomp_bin, addr, size)
        if slots is None:
            continue
        identities = originals.get((size, slots), set())
        if len(identities) == 1:
            identity = next(iter(identities))
            canonical = db.get(ImageId.ORIG, identity)
            if canonical is not None and canonical.recomp_addr is not None:
                db.set_alias(ImageId.RECOMP, addr, identity)
