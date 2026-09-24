"""Symbolic execution of one instruction on one side."""

from __future__ import annotations

from reccmp.compare.asm.model import (
    REGISTERS,
    Instruction,
    Reject,
    operand_display,
    operand_identity,
)
from reccmp.compare.asm.verifier.addresses import (
    Value,
    flatten_mem,
    stack_rooted,
    unwind_spadd,
)
from reccmp.compare.asm.verifier.state import (
    CARRY_BINOPS,
    CC_CANON,
    COMMUTATIVE_BINOPS,
    JCC_MNEMONICS,
    ORDERED_BINOPS,
    STRING_OPS,
    WIDTHS,
    X87_CONSTANTS,
    X87_UNARY,
    ZERO_FLAGS,
    Context,
    SideState,
    register_arguments,
    X87Stack,
    commutative_result,
    memory_load_tag,
    vsort,
)

# ---------------------------------------------------------------------------
# Symbolic execution of one side


def mem_address(
    state: SideState, op, escape: bool = False, write: bool = False
) -> Value:
    # pylint: disable=too-many-boolean-expressions
    _, size, seg, reg_terms, disp, syms = op
    disp_key: Value | int = disp
    if (
        state.rename_slots
        and not seg
        and not syms
        and disp < 0
        and any(reg == "ebp" for reg, _ in reg_terms)
        and stack_rooted(state.regs["bp"])
    ):
        if not escape and reg_terms == [("ebp", 1)]:
            # A plain frame-local slot: alpha-renamable across the sides.
            disp_key = state.slot_ref(disp, size, write)
        else:
            # The slot's address escapes (lea) or the access is indexed
            # (a local array): renaming frame slots is no longer safe.
            state.slots_escaped = True
    pairs = [(state.read_reg(reg), scale) for reg, scale in reg_terms]
    if isinstance(disp_key, int):
        # Fold constant stack-pointer adjustments into the displacement so
        # that e.g. [esp + 8] before a push and [esp + 0xc] after it denote
        # the same address. Skipped for alpha-renamed frame slots, whose
        # key is the slot id rather than an offset.
        folded = []
        for value, scale in pairs:
            base, offset = unwind_spadd(value)
            if offset:
                disp_key += scale * offset
                value = base
            folded.append((value, scale))
        pairs = folded
    terms = tuple(sorted(pairs, key=repr))
    if escape and any(stack_rooted(value) for value, _ in terms):
        # A stack address escapes into a register: pointers derived from it
        # could reach frame slots or saved registers on the stack.
        state.slots_escaped = True
    return ("mem", seg, terms, disp_key, syms)


def read_operand(state: SideState, ctx: Context, op) -> Value:
    kind = op[0]
    if kind == "reg":
        return state.read_reg(op[1])
    if kind == "imm":
        return ("imm", op[1])
    if kind == "sym":
        return ("sym", operand_identity(op[1]))
    if kind == "st":
        return state.x87.read(op[1])
    if kind == "mem":
        address = mem_address(state, op)
        width = WIDTHS.get(op[1])
        if ctx.trace is not None:
            ctx.trace.append(("r", address, width, False))
        tag = memory_load_tag(ctx, address, width, False)
        state.load_log.add((address, tag))
        return ("load", address, op[1], tag)
    raise Reject


def _canonical_address_value(address: Value) -> Value:
    """Return the arithmetic value computed by a simple LEA.

    A compiler may spell pointer addition either as ``lea [value + offset]``
    or as an integer ``add`` on the loaded pointer.  Keep symbolic/global and
    stack addresses as address expressions, but canonicalize ordinary
    scale-one pointer arithmetic through the same associative-add builder used
    by ADD itself.
    """
    mem = flatten_mem(address)
    _, seg, terms, disp, syms = mem
    if seg or syms or not isinstance(disp, int):
        return ("addr", mem)
    if any(scale != 1 or stack_rooted(value) for value, scale in terms):
        return ("addr", mem)

    values = [value for value, _ in terms]
    if disp or not values:
        values.append(("imm", disp))
    if len(values) == 1:
        return values[0]

    result = values[0]
    for value in values[1:]:
        result = commutative_result("add", result, value)
    return result


def receiver_equivalence_class(receiver: Value, ctx: Context) -> Value:
    """Canonical receiver identity independent of reload generation tags."""
    seen: set[int] = set()
    while (
        isinstance(receiver, tuple)
        and len(receiver) == 4
        and receiver[0] == "load"
        and id(receiver) not in seen
    ):
        seen.add(id(receiver))
        address, size = receiver[1], WIDTHS.get(receiver[2])
        forwarded = ctx.receiver_values.get((address, size))
        if forwarded is None:
            return ("receiver_load", address, receiver[2])
        store_tag, stored_value = forwarded
        load_tag = receiver[3]
        event_tags = [tag for tag, _ in ctx.mem_events]
        if store_tag == load_tag:
            receiver = stored_value
        elif store_tag in event_tags and load_tag in event_tags:
            if event_tags.index(store_tag) < event_tags.index(load_tag):
                receiver = stored_value
            else:
                return ("receiver_load", address, receiver[2])
        else:
            return ("receiver_load", address, receiver[2])
    return receiver


def _canonical_virtual_target(target: Value, ctx: Context) -> Value | None:
    """Recognize ``load(load(receiver) + slot)`` as a virtual call target.

    The physical register carrying the vtable is intentionally absent from
    the result.  Existing symbolic-value and CFG-join equivalence then makes
    register allocation, moved equivalent loads, and equivalent phi inputs
    transparent while retaining the receiver and slot as the call identity.
    """
    if not (isinstance(target, tuple) and len(target) == 4 and target[0] == "load"):
        return None

    call_mem = flatten_mem(target[1])
    _, call_seg, call_terms, slot, call_syms = call_mem
    if call_seg or call_syms or not isinstance(slot, int) or len(call_terms) != 1:
        return None
    vtable, scale = call_terms[0]
    if scale != 1 or not (
        isinstance(vtable, tuple) and len(vtable) == 4 and vtable[0] == "load"
    ):
        return None

    receiver_mem = flatten_mem(vtable[1])
    _, receiver_seg, receiver_terms, receiver_disp, receiver_syms = receiver_mem
    if (
        receiver_seg
        or receiver_syms
        or receiver_disp != 0
        or len(receiver_terms) != 1
        or receiver_terms[0][1] != 1
    ):
        return None
    receiver = receiver_equivalence_class(receiver_terms[0][0], ctx)
    return ("vcall", receiver, slot)


def write_operand(state: SideState, ctx: Context, op, value: Value, obs: list) -> None:
    kind = op[0]
    if kind == "reg":
        state.write_reg(op[1], value)
    elif kind == "mem":
        address = mem_address(state, op, write=True)
        if ctx.trace is not None:
            ctx.trace.append(("w", address, WIDTHS.get(op[1]), False))
        obs.append(("store", address, op[1], value))
    elif kind == "st":
        state.x87.write(op[1], value)
    else:
        raise Reject


def _mul_registers(op) -> tuple[str, str]:
    """Accumulator/high register pair for single-operand mul/imul/div/idiv,
    depending on the operand width."""
    width = op[1] if op[0] == "mem" else REGISTERS[op[1]][1]
    if width in ("byte", "l8", "h8"):
        return "al", "ah"
    if width in ("word", "r16"):
        return "ax", "dx"
    return "eax", "edx"


def esp_add(value: Value, delta: int) -> Value:
    if isinstance(value, tuple) and value[0] == "spadd":
        base, offset = value[1], value[2] + delta
        return base if offset == 0 else ("spadd", base, offset)
    return ("spadd", value, delta)


# Condition codes whose outcome depends on the carry flag.
CF_CONDITIONS = frozenset({"b", "ae", "a", "be"})


def canon_condition(cc: str, state: SideState) -> Value:
    """Canonical predicate for a condition code applied to a flag state, so
    that `cmp a, b` + jg equals `cmp b, a` + jl."""
    flags = state.flags
    entry = CC_CANON.get(cc)
    if entry is not None and isinstance(flags, tuple) and flags[0] == "cmp":
        pred, swap = entry
        a, b = flags[1], flags[2]
        width = flags[3] if len(flags) > 3 else None
        base: tuple
        if pred in ("eq", "ne"):
            base = (pred, vsort(a, b))
        else:
            base = (pred, b, a) if swap else (pred, a, b)
        return base if width is None else (*base, width)
    if (
        cc in ("o", "no")
        and isinstance(flags, tuple)
        and flags[0] == "sahf"
        and len(flags) > 2
    ):
        # OF is preserved across SAHF; observe the prior flag producer.
        return ("cc", cc, flags[2])
    if cc in CF_CONDITIONS:
        # The carry flag may have a different (older) producer than the
        # rest of the flags.
        return ("cc", cc, flags, state.carry)
    return ("cc", cc, flags)


_PART_BYTES = {"l8": 1, "h8": 1, "r8": 1, "r8h": 1, "r16": 2, "r32": 4}


def _compare_width(op_a, op_b) -> int | str:
    """Byte width of a CMP/TEST, used so signed/unsigned outcomes stay distinct."""
    for op in (op_a, op_b):
        if not isinstance(op, tuple) or not op:
            continue
        if op[0] == "reg":
            part = REGISTERS.get(op[1], (None, None))[1]
            if part in _PART_BYTES:
                return _PART_BYTES[part]
        if op[0] == "mem":
            size = op[1]
            if size in WIDTHS:
                return WIDTHS[size]
            if isinstance(size, str) and size.startswith("size"):
                try:
                    return int(size[4:])
                except ValueError:
                    return size
    return "unk"


def _branch_obs_dest(ins: Instruction) -> object:
    """Proof identity of a direct transfer; never a relative displacement."""
    if ins.control_target is not None:
        return ins.control_target
    if ins.raw_operands:
        return ins.raw_operands[0]
    return None


def execute(
    state: SideState, ctx: Context, idx: int, ins: Instruction, obs: list
) -> None:
    # pylint: disable=too-many-locals
    # pylint: disable=too-many-branches,too-many-statements
    mnemonic = ins.mnemonic
    ops = ins.operands
    value: Value

    if mnemonic == "mov" and len(ops) == 2:
        write_operand(state, ctx, ops[0], read_operand(state, ctx, ops[1]), obs)
    elif mnemonic in ("movsx", "movzx") and len(ops) == 2:
        src = ops[1]
        width = src[1] if src[0] == "mem" else REGISTERS[src[1]][1]
        value = (mnemonic, width, read_operand(state, ctx, src))
        write_operand(state, ctx, ops[0], value, obs)
    elif mnemonic == "lea" and len(ops) == 2 and ops[1][0] == "mem":
        address = mem_address(state, ops[1], escape=True)
        write_operand(state, ctx, ops[0], _canonical_address_value(address), obs)
    elif mnemonic == "xchg" and len(ops) == 2:
        a = read_operand(state, ctx, ops[0])
        b = read_operand(state, ctx, ops[1])
        write_operand(state, ctx, ops[0], b, obs)
        write_operand(state, ctx, ops[1], a, obs)
    elif mnemonic in COMMUTATIVE_BINOPS and len(ops) == 2:
        a = read_operand(state, ctx, ops[0])
        b = read_operand(state, ctx, ops[1])
        pair = vsort(a, b)
        if mnemonic == "xor" and a == b:
            value = ("imm", 0)
        elif mnemonic in ("and", "or") and a == b:
            # and/or of a value with itself leaves it unchanged.
            value = a
        else:
            value = commutative_result(mnemonic, a, b)
        write_operand(state, ctx, ops[0], value, obs)
        if mnemonic == "xor" and a == b:
            # Zero idiom: the flags are those of comparing zero with zero.
            state.flags = ZERO_FLAGS
        elif mnemonic in ("and", "or") and a == b:
            # SF/ZF/PF reflect the value; CF and OF are cleared: exactly
            # the flag state of `cmp value, 0`.
            width = _compare_width(ops[0], ops[0])
            state.flags = ("cmp", a, ("imm", 0), width)
        else:
            state.flags = ("flags", mnemonic, *pair)
        # and/or/xor clear CF; add/imul produce a carry-out.
        if mnemonic in ("and", "or", "xor"):
            state.carry = ("cf0",)
        else:
            state.carry = ("carry", mnemonic, *pair)
    elif mnemonic == "imul" and len(ops) == 3:
        value = (
            "imul3",
            read_operand(state, ctx, ops[1]),
            read_operand(state, ctx, ops[2]),
        )
        write_operand(state, ctx, ops[0], value, obs)
        state.flags = ("flags", *value)
        state.carry = ("carry", *value)
    elif mnemonic in ORDERED_BINOPS and len(ops) == 2:
        a = read_operand(state, ctx, ops[0])
        b = read_operand(state, ctx, ops[1])
        value = ("imm", 0) if (mnemonic == "sub" and a == b) else (mnemonic, a, b)
        write_operand(state, ctx, ops[0], value, obs)
        if mnemonic == "sub" and a == b:
            # Zero idiom: same flag state as xor r, r.
            state.flags = ZERO_FLAGS
            state.carry = ("cf0",)
        else:
            state.flags = ("flags", mnemonic, a, b)
            # The borrow out of sub is the unsigned comparison of its operands.
            state.carry = (
                ("lt_u", a, b) if mnemonic == "sub" else ("carry", mnemonic, a, b)
            )
    elif mnemonic in CARRY_BINOPS and len(ops) == 2:
        a = read_operand(state, ctx, ops[0])
        b = read_operand(state, ctx, ops[1])
        value = (mnemonic, a, b, state.carry)
        write_operand(state, ctx, ops[0], value, obs)
        state.flags = ("flags", *value)
        state.carry = ("carry", *value)
    elif mnemonic in ("inc", "dec", "neg", "not") and len(ops) == 1:
        value = (mnemonic, read_operand(state, ctx, ops[0]))
        write_operand(state, ctx, ops[0], value, obs)
        # inc/dec rewrite the flags but preserve CF; not touches nothing.
        if mnemonic != "not":
            state.flags = ("flags", *value)
        if mnemonic == "neg":
            state.carry = ("carry", *value)
    elif mnemonic == "cmp" and len(ops) == 2:
        a = read_operand(state, ctx, ops[0])
        b = read_operand(state, ctx, ops[1])
        width = _compare_width(ops[0], ops[1])
        if a == b:
            state.flags = ZERO_FLAGS
        else:
            state.flags = ("cmp", a, b, width)
        # `x < x` and `x < 0` (unsigned) are always false.
        if b in (a, ("imm", 0)):
            state.carry = ("cf0",)
        else:
            state.carry = ("lt_u", a, b, width)
    elif mnemonic == "test" and len(ops) == 2:
        a = read_operand(state, ctx, ops[0])
        b = read_operand(state, ctx, ops[1])
        width = _compare_width(ops[0], ops[1])
        if a == b:
            # `test r, r` sets SF/ZF/PF from the value and clears CF/OF:
            # exactly the flag state of `cmp r, 0`.
            state.flags = ("cmp", a, ("imm", 0), width)
        else:
            state.flags = ("test", *vsort(a, b), width)
        state.carry = ("cf0",)
    elif mnemonic in ("mul", "imul") and len(ops) == 1:
        acc, hi = _mul_registers(ops[0])
        pair = vsort(state.read_reg(acc), read_operand(state, ctx, ops[0]))
        state.write_reg(acc, (mnemonic, "lo", *pair))
        state.write_reg(hi, (mnemonic, "hi", *pair))
        state.flags = ("flags", mnemonic, *pair)
        state.carry = ("carry", mnemonic, *pair)
    elif mnemonic in ("div", "idiv") and len(ops) == 1:
        acc, hi = _mul_registers(ops[0])
        divisor = read_operand(state, ctx, ops[0])
        dividend = (state.read_reg(hi), state.read_reg(acc))
        state.write_reg(acc, (mnemonic, "quot", dividend, divisor))
        state.write_reg(hi, (mnemonic, "rem", dividend, divisor))
        state.flags = ("undef_flags", idx)
        state.carry = ("undef_cf", idx)
    elif mnemonic == "cdq":
        state.write_reg("edx", ("cdq", state.read_reg("eax")))
    elif mnemonic == "cwde":
        state.write_reg("eax", ("cwde", state.read_reg("ax")))
    elif mnemonic == "sahf":
        # SAHF loads SF/ZF/AF/PF/CF from AH; OF is preserved.
        state.flags = ("sahf", state.read_reg("ah"), state.flags)
        state.carry = ("sahf_cf", state.read_reg("ah"))
    elif mnemonic == "push" and len(ops) == 1:
        value = read_operand(state, ctx, ops[0])
        new_esp = esp_add(state.read_reg("esp"), -4)
        state.write_reg("esp", new_esp)
        if ctx.trace is not None:
            ctx.trace.append(("w", new_esp, 4, "push"))
        obs.append(("store", new_esp, "stack", value))
    elif mnemonic == "pop" and len(ops) == 1:
        esp = state.read_reg("esp")
        if ctx.trace is not None:
            ctx.trace.append(("r", esp, 4, "pop"))
        write_operand(
            state,
            ctx,
            ops[0],
            ("load", esp, "stack", memory_load_tag(ctx, esp, 4, "pop")),
            obs,
        )
        state.write_reg("esp", esp_add(esp, 4))
    elif mnemonic == "leave":
        ebp = state.read_reg("ebp")
        if ctx.trace is not None:
            ctx.trace.append(("r", ebp, 4, "pop"))
        state.write_reg(
            "ebp", ("load", ebp, "stack", memory_load_tag(ctx, ebp, 4, "pop"))
        )
        state.write_reg("esp", esp_add(ebp, 4))
    elif mnemonic == "call" and len(ops) == 1:
        # The callee may take arguments in ecx (thiscall) or ecx+edx
        # (fastcall). When per-callsite convention data from the PDB is
        # available and says a register is unused, its (dead) value need
        # not match; otherwise it must match exactly.
        facts = None
        if ctx.metadata is not None and ctx.metadata.call_facts is not None:
            if ops[0][0] == "sym":
                facts = ctx.metadata.call_facts(operand_display(ops[0][1]))
        ecx_argument, edx_argument = register_arguments(facts)
        target = read_operand(state, ctx, ops[0])
        virtual_target = _canonical_virtual_target(target, ctx)
        entry = ["call", virtual_target or target]
        # A known virtual target is not proof that arguments agree. Always
        # observe the actual this/receiver (ecx). Include edx only when the
        # ABI says it is an argument — never merely because the call looked
        # virtual (edx often holds the vtable pointer, not an argument).
        if virtual_target is not None:
            entry.append(receiver_equivalence_class(state.read_reg("ecx"), ctx))
            if facts is not None and facts.uses_edx:
                entry.append(state.read_reg("edx"))
        else:
            if ecx_argument:
                entry.append(state.read_reg("ecx"))
            if edx_argument:
                entry.append(state.read_reg("edx"))
        obs.append(tuple(entry))
        incoming_esp = state.read_reg("esp")
        for reg in ("eax", "ecx", "edx"):
            state.write_reg(reg, ("callret", idx, reg))
        # Preserve dependence on incoming SP; unknown cleanup must not
        # erase a pre-call stack discrepancy.
        state.write_reg("esp", ("callesp", idx, incoming_esp))
        state.flags = ("callflags", idx)
        state.carry = ("callcf", idx)
        state.x87 = X87Stack(epoch=idx + 1)
    elif mnemonic == "ret":
        obs.append(("retstack", ins.raw_operands, state.x87.state_key()[1:]))
        # Externally observable machine state at return must match exactly:
        # the callee-saved registers, the stack pointer, and the return
        # value as determined by the function's return kind. Without
        # return-type metadata from the PDB, eax must match exactly.
        obs.append(
            (
                "retsaved",
                tuple(state.regs[f] for f in ("b", "si", "di", "bp", "sp")),
            )
        )
        kind = ctx.metadata.return_kind if ctx.metadata is not None else "unknown"
        if kind == "void":
            pass
        elif kind == "float":
            obs.append(("retfpu", state.x87.read(0)))
        elif kind == "i8":
            obs.append(("retval", state.read_reg("al")))
        elif kind == "i16":
            obs.append(("retval", state.read_reg("ax")))
        elif kind == "i64":
            obs.append(("retval", state.read_reg("eax"), state.read_reg("edx")))
        elif state.x87.known:
            # x87 return value: st(0) must match; eax is scratch.
            obs.append(("retfpu", state.x87.known[0]))
        else:
            obs.append(("retval", state.read_reg("eax")))
    elif mnemonic in JCC_MNEMONICS and len(ops) == 1:
        pred = canon_condition(JCC_MNEMONICS[mnemonic], state)
        obs.append(("branch", pred, _branch_obs_dest(ins)))
    elif mnemonic == "jmp" and len(ops) == 1:
        if ops[0][0] == "mem":
            obs.append(("jmpind", read_operand(state, ctx, ops[0])))
        else:
            obs.append(("jmp", _branch_obs_dest(ins)))
    elif mnemonic in ("loop", "loope", "loopne", "jcxz", "jecxz") and len(ops) == 1:
        obs.append(
            (mnemonic, state.read_reg("ecx"), state.flags, _branch_obs_dest(ins))
        )
        if mnemonic.startswith("loop"):
            state.write_reg("ecx", ("loopdec", state.read_reg("ecx")))
    elif mnemonic.startswith("set") and mnemonic[3:] in CC_CANON and len(ops) == 1:
        pred = canon_condition(mnemonic[3:], state)
        write_operand(state, ctx, ops[0], ("setcc", pred), obs)
    elif mnemonic.startswith("cmov") and mnemonic[4:] in JCC_MNEMONICS.values():
        pred = canon_condition(mnemonic[4:], state)
        value = (
            "cmov",
            pred,
            read_operand(state, ctx, ops[0]),
            read_operand(state, ctx, ops[1]),
        )
        write_operand(state, ctx, ops[0], value, obs)
    elif mnemonic in STRING_OPS:
        reads, writes, _writes_memory = STRING_OPS[mnemonic]
        key = (mnemonic, ins.prefix)
        observed = [key]
        for family in reads.split():
            observed.append(state.regs[family])
        if ins.prefix:
            observed.append(state.regs["c"])
        obs.append(tuple(observed))
        for family in writes.split():
            state.regs[family] = ("strres", idx, family)
        if ins.prefix:
            state.regs["c"] = ("strres", idx, "c")
        if mnemonic.startswith(("scas", "cmps")):
            state.flags = ("strflags", idx)
            state.carry = ("strcf", idx)

    elif mnemonic in ("nop", "int3"):
        pass
    elif mnemonic.startswith("f"):
        execute_x87(state, ctx, ins, obs)
    else:
        raise Reject


def execute_x87(state: SideState, ctx: Context, ins: Instruction, obs: list) -> None:
    # pylint: disable=too-many-branches,too-many-statements
    mnemonic = ins.mnemonic
    ops = ins.operands
    x87 = state.x87

    if mnemonic in ("fld", "fild") and len(ops) == 1:
        x87.push(read_operand(state, ctx, ops[0]))
    elif mnemonic in X87_CONSTANTS and not ops:
        x87.push(("fconst", mnemonic))
    elif mnemonic in ("fst", "fstp", "fist", "fistp") and len(ops) == 1:
        value = x87.read(0)
        if mnemonic.startswith("fist"):
            value = ("fist", value)
        write_operand(state, ctx, ops[0], value, obs)
        if mnemonic.endswith("p"):
            x87.pop()
    elif mnemonic in ("fadd", "fmul", "faddp", "fmulp", "fiadd", "fimul"):
        op = "f" + ("add" if "add" in mnemonic else "mul")
        if mnemonic in ("faddp", "fmulp"):
            dest = ops[0][1] if ops else 1
            value = (op, *vsort(x87.read(dest), x87.read(0)))
            x87.write(dest, value)
            x87.pop()
        elif len(ops) == 2 and ops[0] == ("st", 0):
            x87.write(0, (op, *vsort(x87.read(0), x87.read(ops[1][1]))))
        elif len(ops) == 2 and ops[1] == ("st", 0):
            dest = ops[0][1]
            x87.write(dest, (op, *vsort(x87.read(dest), x87.read(0))))
        elif len(ops) == 1:
            x87.write(0, (op, *vsort(x87.read(0), read_operand(state, ctx, ops[0]))))
        else:
            raise Reject
    elif (
        mnemonic in ("fsub", "fsubr", "fdiv", "fdivr", "fisub", "fidiv")
        and len(ops) == 1
    ):
        op = "fsub" if "sub" in mnemonic else "fdiv"
        other = read_operand(state, ctx, ops[0])
        if mnemonic.endswith("r"):
            x87.write(0, (op, other, x87.read(0)))
        else:
            x87.write(0, (op, x87.read(0), other))
    elif mnemonic in ("fsubp", "fsubrp", "fdivp", "fdivrp"):
        op = "fsub" if "sub" in mnemonic else "fdiv"
        dest = ops[0][1] if ops else 1
        if "r" in mnemonic[4:]:
            value = (op, x87.read(0), x87.read(dest))
        else:
            value = (op, x87.read(dest), x87.read(0))
        x87.write(dest, value)
        x87.pop()
    elif mnemonic in X87_UNARY and not ops:
        x87.write(0, (mnemonic, x87.read(0)))
    elif mnemonic == "fxch":
        i = ops[0][1] if ops else 1
        a, b = x87.read(0), x87.read(i)
        x87.write(0, b)
        x87.write(i, a)
    elif mnemonic in ("fcom", "fcomp", "fucom", "fucomp", "ficom", "ficomp"):
        other = read_operand(state, ctx, ops[0]) if ops else x87.read(1)
        state.fpu_flags = ("fcom", x87.read(0), other)
        if mnemonic.endswith("p"):
            x87.pop()
    elif mnemonic in ("fcompp", "fucompp"):
        state.fpu_flags = ("fcom", x87.read(0), x87.read(1))
        x87.pop()
        x87.pop()
    elif mnemonic == "ftst":
        state.fpu_flags = ("fcom", x87.read(0), ("imm", 0))
    elif mnemonic == "fnstsw" and ops == (("reg", "ax"),):
        state.write_reg("ax", ("fsw", state.fpu_flags))
    elif mnemonic == "fnstcw" and len(ops) == 1:
        write_operand(state, ctx, ops[0], ("fcw",), obs)
    elif mnemonic == "fldcw" and len(ops) == 1:
        # Loading the control word affects rounding of subsequent operations;
        # the loaded value flows in via a checked channel only if it differs.
        obs.append(("fldcw", read_operand(state, ctx, ops[0])))
    elif mnemonic in ("fprem", "fscale"):
        x87.write(0, (mnemonic, x87.read(0), x87.read(1)))
    elif mnemonic in ("fpatan", "fyl2x"):
        value = (mnemonic, x87.read(0), x87.read(1))
        x87.pop()
        x87.write(0, value)
    else:
        raise Reject
