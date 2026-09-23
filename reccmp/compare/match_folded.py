"""Matching for MSVC-specific shapes: SEH handler blocks and FOLDED
annotations bound to an ICF-folded canonical pair."""

from reccmp.compare.db import EntityDb
from reccmp.compare.event import ReccmpEvent, ReccmpReportProtocol, reccmp_report_nop
from reccmp.compare.lines import LinesDb
from reccmp.compare.match_msvc import match_name
from reccmp.parser.codebase import DecompCodebase
from reccmp.parser.node import ParserFunction
from reccmp.types import EntityType, ImageId


def match_folded_function_aliases(
    db: EntityDb,
    codebase: DecompCodebase,
    lines_db: LinesDb,
    report: ReccmpReportProtocol = reccmp_report_nop,
    *,
    truncate: bool = False,
) -> None:
    """Bind FOLDED annotations to the canonical pair as recomp-side aliases.

    Retail ICF may keep one body at an address that several recomp symbols still
    emit under ``/OPT:NOICF``. Each FOLDED annotation names one of those recomp
    symbols; once the canonical (non-folded) annotation owns the orig address,
    the extras become ``set_alias`` edges so vtable slots and operand compare
    treat them as the shared original identity.
    """

    recomp_by_name: dict[str, list[int]] = {}
    for ent in db.get_all():
        if ent.recomp_addr is None:
            continue
        if ent.get("type") not in (EntityType.FUNCTION, None):
            continue
        name = ent.get("name")
        if not isinstance(name, str) or not name:
            continue
        key = match_name(name[:255] if truncate else name)
        recomp_by_name.setdefault(key, []).append(ent.recomp_addr)

    folded: list[ParserFunction] = [
        symbol
        for symbol in (
            *codebase.iter_line_functions(),
            *codebase.iter_name_functions(),
        )
        if symbol.is_folded
    ]
    for symbol in folded:
        if db.get_one_match(symbol.offset) is None:
            report(
                ReccmpEvent.NO_MATCH,
                symbol.offset,
                msg=(
                    f"FOLDED annotation at 0x{symbol.offset:x} has no canonical "
                    f"owner to alias ({symbol.name})"
                ),
            )
            continue

        recomp_addr: int | None = None
        if symbol.is_nameref():
            key = match_name(symbol.name[:255] if truncate else symbol.name)
            candidates = [
                addr
                for addr in recomp_by_name.get(key, [])
                if db.alias_canonical_orig(ImageId.RECOMP, addr) is None
            ]
            if len(candidates) == 1:
                recomp_addr = candidates[0]
            elif not candidates:
                report(
                    ReccmpEvent.NO_MATCH,
                    symbol.offset,
                    msg=f"Failed to alias FOLDED name '{symbol.name}' at 0x{symbol.offset:x}",
                )
                continue
            else:
                report(
                    ReccmpEvent.AMBIGUOUS_MATCH,
                    symbol.offset,
                    msg=(
                        f"Ambiguous FOLDED name '{symbol.name}' has "
                        f"{len(candidates)} recomp candidates"
                    ),
                )
                continue
        else:
            assert symbol.filename is not None
            recomp_addr = lines_db.find_function(
                symbol.filename,
                symbol.line_number,
                symbol.end_line,
                folded=True,
            )
            if recomp_addr is None:
                continue

        if not db.set_alias(ImageId.RECOMP, recomp_addr, symbol.offset):
            report(
                ReccmpEvent.NO_MATCH,
                symbol.offset,
                msg=(
                    f"Could not alias FOLDED recomp 0x{recomp_addr:x} to "
                    f"0x{symbol.offset:x} ({symbol.name})"
                ),
            )


def _seh_side_key(img_id: ImageId, field: str) -> str:
    side = "orig" if img_id == ImageId.ORIG else "recomp"
    return f"seh_{field}_{side}"


def match_seh(db: EntityDb):
    """Pair VC5 SEH metadata through an already-paired owning function.

    Generic ``__ehhandler``/``__ehfuncinfo`` names deliberately play no role.
    An owner must have exactly one structurally recovered handler on each side.
    Unwind actions are paired only when their target state is unique on both
    sides and neither action is shared by another state.
    """
    handlers_by_owner: dict[ImageId, dict[int, list[int]]] = {
        ImageId.ORIG: {},
        ImageId.RECOMP: {},
    }

    for img_id in (ImageId.ORIG, ImageId.RECOMP):
        owner_key = _seh_side_key(img_id, "owner")
        for entity in db.unmatched(img_id):
            owner = entity.get(owner_key)
            addr = entity.addr(img_id)
            if isinstance(owner, int) and isinstance(addr, int):
                handlers_by_owner[img_id].setdefault(owner, []).append(addr)

    with db.batch() as batch:
        for owner in db.get_matches_by_type(EntityType.FUNCTION):
            orig_handlers = handlers_by_owner[ImageId.ORIG].get(owner.orig_addr, [])
            recomp_handlers = handlers_by_owner[ImageId.RECOMP].get(
                owner.recomp_addr, []
            )
            if len(orig_handlers) != 1 or len(recomp_handlers) != 1:
                continue

            orig_handler_addr = orig_handlers[0]
            recomp_handler_addr = recomp_handlers[0]
            orig_handler = db.get(ImageId.ORIG, orig_handler_addr)
            recomp_handler = db.get(ImageId.RECOMP, recomp_handler_addr)
            if orig_handler is None or recomp_handler is None:
                continue

            orig_funcinfo_addr = orig_handler.get(
                _seh_side_key(ImageId.ORIG, "funcinfo")
            )
            recomp_funcinfo_addr = recomp_handler.get(
                _seh_side_key(ImageId.RECOMP, "funcinfo")
            )
            if not isinstance(orig_funcinfo_addr, int) or not isinstance(
                recomp_funcinfo_addr, int
            ):
                continue

            orig_funcinfo = db.get(ImageId.ORIG, orig_funcinfo_addr)
            recomp_funcinfo = db.get(ImageId.RECOMP, recomp_funcinfo_addr)
            if orig_funcinfo is None or recomp_funcinfo is None:
                continue

            batch.match(orig_handler_addr, recomp_handler_addr)
            batch.match(orig_funcinfo_addr, recomp_funcinfo_addr)

            orig_unwinds = orig_funcinfo.get(_seh_side_key(ImageId.ORIG, "unwinds"), ())
            recomp_unwinds = recomp_funcinfo.get(
                _seh_side_key(ImageId.RECOMP, "unwinds"), ()
            )
            orig_by_state: dict[int, list[int]] = {}
            recomp_by_state: dict[int, list[int]] = {}
            for unwind in orig_unwinds:
                if unwind.action_addr:
                    orig_by_state.setdefault(unwind.target_state, []).append(
                        unwind.action_addr
                    )
            for unwind in recomp_unwinds:
                if unwind.action_addr:
                    recomp_by_state.setdefault(unwind.target_state, []).append(
                        unwind.action_addr
                    )

            orig_actions = [
                addr for addresses in orig_by_state.values() for addr in addresses
            ]
            recomp_actions = [
                addr for addresses in recomp_by_state.values() for addr in addresses
            ]
            for state in orig_by_state.keys() & recomp_by_state.keys():
                orig_addrs = orig_by_state[state]
                recomp_addrs = recomp_by_state[state]
                if len(orig_addrs) != 1 or len(recomp_addrs) != 1:
                    continue

                orig_action = orig_addrs[0]
                recomp_action = recomp_addrs[0]
                if (
                    orig_actions.count(orig_action) != 1
                    or recomp_actions.count(recomp_action) != 1
                ):
                    continue
                if (
                    db.get(ImageId.ORIG, orig_action) is not None
                    and db.get(ImageId.RECOMP, recomp_action) is not None
                ):
                    batch.match(orig_action, recomp_action)
