"""Symbolic execution of one instruction on one side."""

from __future__ import annotations

from collections.abc import Hashable

from reccmp.compare.asm.ir import DecodedInstruction
from reccmp.compare.asm.model import REGISTERS, Reject
from reccmp.compare.asm.operand import Imm, Mem, Operand, Reg, ScaledReg, St, Sym
from reccmp.compare.asm.verifier.addresses import (
    AddressTerm,
    AddressValue,
    CallResult,
    Compare,
    CompareKind,
    ConditionCode,
    Constant,
    DivideResult,
    Extend,
    ExtendKind,
    Load,
    MemoryAddress,
    MultiplyResult,
    Operation,
    OperationKind,
    ProductPart,
    Select,
    SetCondition,
    StackOffset,
    StringResult,
    SymbolValue,
    UnaryOperation,
    Value,
    flatten_mem,
    stack_rooted,
    unwind_spadd,
)
from reccmp.compare.asm.verifier.frame import (
    frame_offset,
    maybe_frame_pointer,
    read_slot,
    write_slot,
)
from reccmp.compare.asm.verifier.state import (
    Branch,
    Call,
    CARRY_BINOPS,
    CC_CANON,
    COMMUTATIVE_BINOPS,
    IndirectJump,
    JCC_MNEMONICS,
    Jump,
    LoadControlWord,
    Loop,
    LoopKind,
    Observation,
    ORDERED_BINOPS,
    ReturnFpu,
    ReturnSaved,
    ReturnStack,
    ReturnValue,
    Store,
    STRING_OPS,
    StringOperation,
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
    state: SideState, op: Mem, escape: bool = False, write: bool = False
) -> Value:
    # pylint: disable=too-many-boolean-expressions
    disp_key: Value | int = op.displacement
    if (
        state.rename_slots
        and not op.segment
        and not op.symbols
        and op.displacement < 0
        and any(term.register == "ebp" for term in op.terms)
        and stack_rooted(state.regs["bp"])
    ):
        if not escape and op.terms == (ScaledReg("ebp", 1),):
            # A plain frame-local slot: alpha-renamable across the sides.
            disp_key = state.slot_ref(op.displacement, op.size, write)
        else:
            # The slot's address escapes (lea) or the access is indexed
            # (a local array): renaming frame slots is no longer safe.
            state.slots_escaped = True
    pairs = [
        AddressTerm(state.read_reg(term.register), term.scale) for term in op.terms
    ]
    if isinstance(disp_key, int):
        # Fold constant stack-pointer adjustments into the displacement so
        # that e.g. [esp + 8] before a push and [esp + 0xc] after it denote
        # the same address. Skipped for alpha-renamed frame slots, whose
        # key is the slot id rather than an offset.
        folded = []
        for term in pairs:
            base, offset = unwind_spadd(term.value)
            if offset:
                disp_key += term.scale * offset
            folded.append(AddressTerm(base, term.scale))
        pairs = folded
    terms = tuple(sorted(pairs, key=repr))
    if escape and any(stack_rooted(term.value) for term in terms):
        # A stack address escapes into a register: pointers derived from it
        # could reach frame slots or saved registers on the stack.
        state.slots_escaped = True
    return MemoryAddress(op.segment, terms, disp_key, op.symbols)


def read_operand(state: SideState, ctx: Context, op: Operand) -> Value:
    # pylint: disable=too-many-return-statements
    match op:
        case Reg(name) if name in REGISTERS:
            return state.read_reg(name)
        case Reg(name):
            # A register outside the model (a segment register) reads as a
            # fixed symbol of the function.
            return SymbolValue(name)
        case Imm(value):
            return Constant(value)
        case Sym(ref):
            return SymbolValue(ref.identity)
        case St(index):
            return state.x87.read(index)
        case Mem(size=size):
            address = mem_address(state, op)
            width = WIDTHS.get(size)
            offset = frame_offset(state, ctx, address, width, slot=False)
            if offset is not None:
                assert width is not None
                return read_slot(state, offset, width)
            if ctx.trace is not None:
                ctx.trace.append(("r", address, width, False))
            tag = memory_load_tag(ctx, address, width, False)
            state.load_log.add((address, tag))
            return Load(address, size, tag)
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
    if mem.segment or mem.symbols or not isinstance(mem.displacement, int):
        return AddressValue(mem)
    if any(term.scale != 1 or stack_rooted(term.value) for term in mem.terms):
        return AddressValue(mem)

    values = [term.value for term in mem.terms]
    if mem.displacement or not values:
        values.append(Constant(mem.displacement))
    if len(values) == 1:
        return values[0]

    result = values[0]
    for value in values[1:]:
        result = commutative_result("add", result, value)
    return result


def receiver_equivalence_class(receiver: Value, ctx: Context) -> Value:
    """Canonical receiver identity independent of reload generation tags."""
    seen: set[int] = set()
    while isinstance(receiver, Load) and id(receiver) not in seen:
        seen.add(id(receiver))
        address, size = receiver.address, WIDTHS.get(receiver.width)
        forwarded = ctx.receiver_values.get((address, size))
        if forwarded is None:
            return ("receiver_load", address, receiver.width)
        store_tag, stored_value = forwarded
        load_tag = receiver.generation
        event_tags = [tag for tag, _ in ctx.mem_events]
        if store_tag == load_tag:
            receiver = stored_value
        elif store_tag in event_tags and load_tag in event_tags:
            if event_tags.index(store_tag) < event_tags.index(load_tag):
                receiver = stored_value
            else:
                return ("receiver_load", address, receiver.width)
        else:
            return ("receiver_load", address, receiver.width)
    return receiver


def _canonical_virtual_target(target: Value, ctx: Context) -> Value | None:
    """Recognize ``load(load(receiver) + slot)`` as a virtual call target.

    The physical register carrying the vtable is intentionally absent from
    the result.  Existing symbolic-value and CFG-join equivalence then makes
    register allocation, moved equivalent loads, and equivalent phi inputs
    transparent while retaining the receiver and slot as the call identity.
    """
    if not isinstance(target, Load):
        return None

    call_mem = flatten_mem(target.address)
    if (
        call_mem.segment
        or call_mem.symbols
        or not isinstance(call_mem.displacement, int)
        or len(call_mem.terms) != 1
    ):
        return None
    vtable = call_mem.terms[0]
    if vtable.scale != 1 or not isinstance(vtable.value, Load):
        return None

    receiver_mem = flatten_mem(vtable.value.address)
    if (
        receiver_mem.segment
        or receiver_mem.symbols
        or receiver_mem.displacement != 0
        or len(receiver_mem.terms) != 1
        or receiver_mem.terms[0].scale != 1
    ):
        return None
    receiver = receiver_equivalence_class(receiver_mem.terms[0].value, ctx)
    return ("vcall", receiver, call_mem.displacement)


def write_operand(
    state: SideState,
    ctx: Context,
    op: Operand,
    value: Value,
    obs: list[Observation],
) -> None:
    match op:
        case Reg(name):
            state.write_reg(name, value)
        case Mem(size=size):
            address = mem_address(state, op, write=True)
            width = WIDTHS.get(size)
            offset = frame_offset(state, ctx, address, width, slot=False)
            if offset is not None:
                assert width is not None
                write_slot(state, offset, width, value)
                return
            if ctx.trace is not None:
                ctx.trace.append(("w", address, width, False))
            obs.append(Store(address, size, value))
        case St(index):
            state.x87.write(index, value)
        case _:
            raise Reject


def _operand_width(op: Operand) -> str:
    """A memory operand's size keyword, or a register's part."""
    match op:
        case Mem(size=size):
            return size
        case Reg(name):
            return REGISTERS[name][1]
    raise Reject


def _mul_registers(op: Operand) -> tuple[str, str]:
    """Accumulator/high register pair for single-operand mul/imul/div/idiv,
    depending on the operand width."""
    width = _operand_width(op)
    if width in ("byte", "l8", "h8"):
        return "al", "ah"
    if width in ("word", "r16"):
        return "ax", "dx"
    return "eax", "edx"


def _st_index(op: Operand) -> int:
    match op:
        case St(index):
            return index
    raise Reject


def esp_add(value: Value, delta: int) -> Value:
    if isinstance(value, StackOffset):
        offset = value.offset + delta
        return value.base if offset == 0 else StackOffset(value.base, offset)
    return StackOffset(value, delta)


# Condition codes whose outcome depends on the carry flag.
CF_CONDITIONS = frozenset({"b", "ae", "a", "be"})


def _import_call(target: Value) -> Value:
    """The callee of a call through an import slot: `call [__imp_X]` loads
    the slot, `call thunk` runs the thunk's `jmp [__imp_X]`, which loads it
    at the same point; both run X with the same return address. Only the
    call target is unified: the thunk's address and the slot's content are
    different pointer values."""
    if isinstance(target, SymbolValue) and isinstance(target.identity, tuple):
        if target.identity[:1] == ("jmp_through",):
            return ("call_through", target.identity[1])
    if isinstance(target, Load) and target.width == "dword":
        slot = _absolute_symbol(target.address)
        if slot is not None:
            return ("call_through", slot)
    return target


def _absolute_symbol(address: Value) -> Hashable | None:
    """The identity of `[symbol]`: an address with no registers, no
    displacement and one symbol."""
    if isinstance(address, SymbolValue):
        return address.identity
    if not isinstance(address, MemoryAddress):
        return None
    if (
        address.terms
        or address.displacement != 0
        or len(address.symbols) != 1
        or address.symbols[0].sign != 1
    ):
        return None
    return address.symbols[0].ref.identity


def _constant_order(pred: CompareKind, a: Value, b: Value, width) -> Compare:
    """One spelling of an order against a constant: `x < c` is `x <= c-1`
    and `c < x` is `c+1 <= x`, except at the ends of the range, and the
    constant is written in the comparison's signedness. A compiler picks
    either spelling (`cmp eax, 0x41; jb` / `cmp eax, 0x40; jbe`)."""
    if not isinstance(width, int):
        return Compare(pred, a, b)  # the range, and so the rewrite, is unknown
    bits = 8 * width
    mask = (1 << bits) - 1
    signed = pred in (CompareKind.LT_S, CompareKind.LE_S)

    def value(constant: int) -> int:
        constant &= mask
        if signed and constant >> (bits - 1):
            constant -= 1 << bits
        return constant

    low, high = (-(1 << (bits - 1)), (1 << (bits - 1)) - 1) if signed else (0, mask)
    strict = pred in (CompareKind.LT_U, CompareKind.LT_S)
    kind = CompareKind.LE_S if signed else CompareKind.LE_U
    if isinstance(b, Constant):
        constant = value(b.value)
        if strict and constant > low:
            return Compare(kind, a, Constant(constant - 1), width)
        return Compare(pred, a, Constant(constant), width)
    if isinstance(a, Constant):
        constant = value(a.value)
        if strict and constant < high:
            return Compare(kind, Constant(constant + 1), b, width)
        return Compare(pred, Constant(constant), b, width)
    return Compare(pred, a, b, width)


def canon_condition(cc: str, state: SideState) -> Value:
    """Canonical predicate for a condition code applied to a flag state, so
    that `cmp a, b` + jg equals `cmp b, a` + jl."""
    flags = state.flags
    entry = CC_CANON.get(cc)
    if (
        entry is not None
        and isinstance(flags, tuple)
        and flags[:1] == ("cmp",)
        and len(flags) >= 3
    ):
        pred, swap = entry
        a, b = flags[1], flags[2]
        width = flags[3] if len(flags) > 3 else None
        if pred in (CompareKind.EQ, CompareKind.NE):
            left, right = vsort(a, b)
            return Compare(pred, left, right, width)
        if swap:
            return _constant_order(pred=pred, a=b, b=a, width=width)
        return _constant_order(pred=pred, a=a, b=b, width=width)
    if (
        cc in ("o", "no")
        and isinstance(flags, tuple)
        and flags[0] == "sahf"
        and len(flags) > 2
    ):
        # OF is preserved across SAHF; observe the prior flag producer.
        return ConditionCode(cc, flags[2])
    if cc in CF_CONDITIONS:
        # The carry flag may have a different (older) producer than the
        # rest of the flags.
        return ConditionCode(cc, flags, state.carry)
    return ConditionCode(cc, flags)


_PART_BYTES = {"l8": 1, "h8": 1, "r8": 1, "r8h": 1, "r16": 2, "r32": 4}


def _compare_width(op_a: Operand, op_b: Operand) -> int | str:
    """Byte width of a CMP/TEST, used so signed/unsigned outcomes stay distinct."""
    for op in (op_a, op_b):
        match op:
            case Reg(name) if REGISTERS.get(name, (None, None))[1] in _PART_BYTES:
                return _PART_BYTES[REGISTERS[name][1]]
            case Mem(size=size) if size in WIDTHS:
                return WIDTHS[size]
            case Mem(size=size) if size.startswith("size"):
                try:
                    return int(size[4:])
                except ValueError:
                    return size
    return "unk"


def _branch_obs_dest(ins: DecodedInstruction) -> Hashable | None:
    """Proof identity of a direct transfer; never a relative displacement."""
    return ins.control_target


def execute(
    state: SideState,
    ctx: Context,
    idx: int,
    ins: DecodedInstruction,
    obs: list[Observation],
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
        value = Extend(
            ExtendKind(mnemonic), _operand_width(src), read_operand(state, ctx, src)
        )
        write_operand(state, ctx, ops[0], value, obs)
    elif mnemonic == "lea" and len(ops) == 2 and isinstance(ops[1], Mem):
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
            value = Constant(0)
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
            state.flags = ("cmp", a, Constant(0), width)
        else:
            state.flags = ("flags", mnemonic, *pair)
        # and/or/xor clear CF; add/imul produce a carry-out.
        if mnemonic in ("and", "or", "xor"):
            state.carry = ("cf0",)
        else:
            state.carry = ("carry", mnemonic, *pair)
    elif mnemonic == "imul" and len(ops) == 3:
        value = Operation(
            OperationKind.IMUL3,
            (read_operand(state, ctx, ops[1]), read_operand(state, ctx, ops[2])),
        )
        write_operand(state, ctx, ops[0], value, obs)
        state.flags = ("flags", value.kind.value, *value.operands)
        state.carry = ("carry", value.kind.value, *value.operands)
    elif mnemonic in ORDERED_BINOPS and len(ops) == 2:
        a = read_operand(state, ctx, ops[0])
        b = read_operand(state, ctx, ops[1])
        value = (
            Constant(0)
            if mnemonic == "sub" and a == b
            else Operation(OperationKind(mnemonic), (a, b))
        )
        write_operand(state, ctx, ops[0], value, obs)
        if mnemonic == "sub" and a == b:
            # Zero idiom: same flag state as xor r, r.
            state.flags = ZERO_FLAGS
            state.carry = ("cf0",)
        else:
            state.flags = ("flags", mnemonic, a, b)
            # The borrow out of sub is the unsigned comparison of its operands.
            state.carry = (
                Compare(CompareKind.LT_U, a, b)
                if mnemonic == "sub"
                else ("carry", mnemonic, a, b)
            )
    elif mnemonic in CARRY_BINOPS and len(ops) == 2:
        a = read_operand(state, ctx, ops[0])
        b = read_operand(state, ctx, ops[1])
        value = Operation(OperationKind(mnemonic), (a, b, state.carry))
        write_operand(state, ctx, ops[0], value, obs)
        state.flags = ("flags", value.kind.value, *value.operands)
        state.carry = ("carry", value.kind.value, *value.operands)
    elif mnemonic in ("inc", "dec", "neg", "not") and len(ops) == 1:
        value = UnaryOperation(
            OperationKind(mnemonic), read_operand(state, ctx, ops[0])
        )
        write_operand(state, ctx, ops[0], value, obs)
        # inc/dec rewrite the flags but preserve CF; not touches nothing.
        if mnemonic != "not":
            state.flags = ("flags", value.kind.value, value.operand)
        if mnemonic == "neg":
            state.carry = ("carry", value.kind.value, value.operand)
    elif mnemonic == "cmp" and len(ops) == 2:
        a = read_operand(state, ctx, ops[0])
        b = read_operand(state, ctx, ops[1])
        width = _compare_width(ops[0], ops[1])
        if a == b:
            state.flags = ZERO_FLAGS
        else:
            state.flags = ("cmp", a, b, width)
        # `x < x` and `x < 0` (unsigned) are always false.
        if b in (a, Constant(0)):
            state.carry = ("cf0",)
        else:
            state.carry = Compare(CompareKind.LT_U, a, b, width)
    elif mnemonic == "test" and len(ops) == 2:
        a = read_operand(state, ctx, ops[0])
        b = read_operand(state, ctx, ops[1])
        width = _compare_width(ops[0], ops[1])
        if a == b:
            # `test r, r` sets SF/ZF/PF from the value and clears CF/OF:
            # exactly the flag state of `cmp r, 0`.
            state.flags = ("cmp", a, Constant(0), width)
        else:
            state.flags = ("test", *vsort(a, b), width)
        state.carry = ("cf0",)
    elif mnemonic in ("mul", "imul") and len(ops) == 1:
        acc, hi = _mul_registers(ops[0])
        pair = vsort(state.read_reg(acc), read_operand(state, ctx, ops[0]))
        signed = mnemonic == "imul"
        state.write_reg(acc, MultiplyResult(signed, ProductPart.LOW, pair))
        state.write_reg(hi, MultiplyResult(signed, ProductPart.HIGH, pair))
        state.flags = ("flags", mnemonic, *pair)
        state.carry = ("carry", mnemonic, *pair)
    elif mnemonic in ("div", "idiv") and len(ops) == 1:
        acc, hi = _mul_registers(ops[0])
        divisor = read_operand(state, ctx, ops[0])
        signed = mnemonic == "idiv"
        state.write_reg(
            acc,
            DivideResult(
                signed,
                ProductPart.QUOTIENT,
                state.read_reg(hi),
                state.read_reg(acc),
                divisor,
            ),
        )
        state.write_reg(
            hi,
            DivideResult(
                signed,
                ProductPart.REMAINDER,
                state.read_reg(hi),
                state.read_reg(acc),
                divisor,
            ),
        )
        state.flags = ("undef_flags", idx)
        state.carry = ("undef_cf", idx)
    elif mnemonic == "cdq":
        state.write_reg("edx", UnaryOperation(OperationKind.CDQ, state.read_reg("eax")))
    elif mnemonic == "cwde":
        state.write_reg("eax", UnaryOperation(OperationKind.CWDE, state.read_reg("ax")))
    elif mnemonic == "sahf":
        # SAHF loads SF/ZF/AF/PF/CF from AH; OF is preserved.
        state.flags = ("sahf", state.read_reg("ah"), state.flags)
        state.carry = ("sahf_cf", state.read_reg("ah"))
    elif mnemonic == "push" and len(ops) == 1:
        value = read_operand(state, ctx, ops[0])
        new_esp = esp_add(state.read_reg("esp"), -4)
        state.write_reg("esp", new_esp)
        offset = frame_offset(state, ctx, new_esp, 4, slot=True)
        if offset is not None:
            write_slot(state, offset, 4, value)
        else:
            if ctx.trace is not None:
                ctx.trace.append(("w", new_esp, 4, "push"))
            obs.append(Store(new_esp, "stack", value))
    elif mnemonic == "pop" and len(ops) == 1:
        esp = state.read_reg("esp")
        offset = frame_offset(state, ctx, esp, 4, slot=True)
        if offset is not None:
            popped = read_slot(state, offset, 4)
        else:
            if ctx.trace is not None:
                ctx.trace.append(("r", esp, 4, "pop"))
            popped = Load(esp, "stack", memory_load_tag(ctx, esp, 4, "pop"))
        write_operand(state, ctx, ops[0], popped, obs)
        state.write_reg("esp", esp_add(esp, 4))
    elif mnemonic == "leave":
        ebp = state.read_reg("ebp")
        offset = frame_offset(state, ctx, ebp, 4, slot=True)
        if offset is not None:
            state.write_reg("ebp", read_slot(state, offset, 4))
        else:
            if ctx.trace is not None:
                ctx.trace.append(("r", ebp, 4, "pop"))
            state.write_reg(
                "ebp", Load(ebp, "stack", memory_load_tag(ctx, ebp, 4, "pop"))
            )
        state.write_reg("esp", esp_add(ebp, 4))
    elif mnemonic == "call" and len(ops) == 1:
        # The callee may take arguments in ecx (thiscall) or ecx+edx
        # (fastcall). When per-callsite convention data from the PDB is
        # available and says a register is unused, its (dead) value need
        # not match; otherwise it must match exactly.
        facts = None
        if ctx.metadata is not None and ctx.metadata.call_facts is not None:
            if isinstance(ops[0], Sym):
                facts = ctx.metadata.call_facts(ops[0].ref.identity)
        ecx_argument, edx_argument = register_arguments(facts)
        target = _import_call(read_operand(state, ctx, ops[0]))
        virtual_target = _canonical_virtual_target(target, ctx)
        arguments = []
        # A known virtual target is not proof that arguments agree. Always
        # observe the actual this/receiver (ecx). Include edx only when the
        # ABI says it is an argument — never merely because the call looked
        # virtual (edx often holds the vtable pointer, not an argument).
        if virtual_target is not None:
            arguments.append(receiver_equivalence_class(state.read_reg("ecx"), ctx))
            if facts is not None and facts.uses_edx:
                arguments.append(state.read_reg("edx"))
        else:
            if ecx_argument:
                arguments.append(state.read_reg("ecx"))
            if edx_argument:
                arguments.append(state.read_reg("edx"))
        obs.append(Call(virtual_target or target, tuple(arguments)))
        incoming_esp = state.read_reg("esp")
        for reg in ("eax", "ecx", "edx"):
            state.write_reg(reg, CallResult(idx, reg))
        # Preserve dependence on incoming SP; unknown cleanup must not
        # erase a pre-call stack discrepancy.
        state.write_reg("esp", ("callesp", idx, incoming_esp))
        state.flags = ("callflags", idx)
        state.carry = ("callcf", idx)
        state.x87 = X87Stack(epoch=idx + 1)
    elif mnemonic == "ret":
        obs.append(ReturnStack(ins.operands, state.x87.state_key()[1:]))
        # Externally observable machine state at return must match exactly:
        # the callee-saved registers, the stack pointer, and the return
        # value as determined by the function's return kind. Without
        # return-type metadata from the PDB, eax must match exactly.
        obs.append(
            ReturnSaved(tuple(state.regs[f] for f in ("b", "si", "di", "bp", "sp")))
        )
        kind = ctx.metadata.return_kind if ctx.metadata is not None else "unknown"
        if kind == "void":
            pass
        elif kind == "float":
            obs.append(ReturnFpu(state.x87.read(0)))
        elif kind == "i8":
            obs.append(ReturnValue((state.read_reg("al"),)))
        elif kind == "i16":
            obs.append(ReturnValue((state.read_reg("ax"),)))
        elif kind == "i64":
            obs.append(ReturnValue((state.read_reg("eax"), state.read_reg("edx"))))
        elif state.x87.known:
            # x87 return value: st(0) must match; eax is scratch.
            obs.append(ReturnFpu(state.x87.known[0]))
        else:
            obs.append(ReturnValue((state.read_reg("eax"),)))
    elif mnemonic in JCC_MNEMONICS and len(ops) == 1:
        pred = canon_condition(JCC_MNEMONICS[mnemonic], state)
        obs.append(Branch(pred, _branch_obs_dest(ins)))
    elif mnemonic == "jmp" and len(ops) == 1:
        if isinstance(ops[0], Mem):
            obs.append(IndirectJump(read_operand(state, ctx, ops[0])))
        else:
            obs.append(Jump(_branch_obs_dest(ins)))
    elif mnemonic in ("loop", "loope", "loopne", "jcxz", "jecxz") and len(ops) == 1:
        obs.append(
            Loop(
                LoopKind(mnemonic),
                state.read_reg("ecx"),
                state.flags,
                _branch_obs_dest(ins),
            )
        )
        if mnemonic.startswith("loop"):
            state.write_reg(
                "ecx",
                UnaryOperation(OperationKind.LOOP_DECREMENT, state.read_reg("ecx")),
            )
    elif mnemonic.startswith("set") and mnemonic[3:] in CC_CANON and len(ops) == 1:
        pred = canon_condition(mnemonic[3:], state)
        write_operand(state, ctx, ops[0], SetCondition(pred), obs)
    elif mnemonic.startswith("cmov") and mnemonic[4:] in JCC_MNEMONICS.values():
        pred = canon_condition(mnemonic[4:], state)
        value = Select(
            pred,
            read_operand(state, ctx, ops[0]),
            read_operand(state, ctx, ops[1]),
        )
        write_operand(state, ctx, ops[0], value, obs)
    elif mnemonic in STRING_OPS:
        reads, writes, _writes_memory = STRING_OPS[mnemonic]
        if state.frame and any(
            maybe_frame_pointer(state.regs[family]) for family in ("si", "di")
        ):
            raise Reject  # it may read or write a promoted slot
        observed = [state.regs[family] for family in reads.split()]
        if ins.prefix:
            observed.append(state.regs["c"])
        obs.append(
            StringOperation(mnemonic, ins.prefix, tuple(observed), _writes_memory)
        )
        for family in writes.split():
            state.regs[family] = StringResult(idx, family)
        if ins.prefix:
            state.regs["c"] = StringResult(idx, "c")
        if mnemonic.startswith(("scas", "cmps")):
            state.flags = ("strflags", idx)
            state.carry = ("strcf", idx)

    elif mnemonic in ("nop", "int3"):
        pass
    elif mnemonic.startswith("f"):
        execute_x87(state, ctx, ins, obs)
    else:
        raise Reject


def execute_x87(
    state: SideState,
    ctx: Context,
    ins: DecodedInstruction,
    obs: list[Observation],
) -> None:
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
            dest = _st_index(ops[0]) if ops else 1
            value = (op, *vsort(x87.read(dest), x87.read(0)))
            x87.write(dest, value)
            x87.pop()
        elif len(ops) == 2 and ops[0] == St(0):
            x87.write(0, (op, *vsort(x87.read(0), x87.read(_st_index(ops[1])))))
        elif len(ops) == 2 and ops[1] == St(0):
            dest = _st_index(ops[0])
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
        dest = _st_index(ops[0]) if ops else 1
        if "r" in mnemonic[4:]:
            value = (op, x87.read(0), x87.read(dest))
        else:
            value = (op, x87.read(dest), x87.read(0))
        x87.write(dest, value)
        x87.pop()
    elif mnemonic in X87_UNARY and not ops:
        x87.write(0, (mnemonic, x87.read(0)))
    elif mnemonic == "fxch":
        i = _st_index(ops[0]) if ops else 1
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
        state.fpu_flags = ("fcom", x87.read(0), Constant(0))
    elif mnemonic == "fnstsw" and ops == (Reg("ax"),):
        state.write_reg("ax", ("fsw", state.fpu_flags))
    elif mnemonic == "fnstcw" and len(ops) == 1:
        write_operand(state, ctx, ops[0], ("fcw",), obs)
    elif mnemonic == "fldcw" and len(ops) == 1:
        # Loading the control word affects rounding of subsequent operations;
        # the loaded value flows in via a checked channel only if it differs.
        obs.append(LoadControlWord(read_operand(state, ctx, ops[0])))
    elif mnemonic in ("fprem", "fscale"):
        x87.write(0, (mnemonic, x87.read(0), x87.read(1)))
    elif mnemonic in ("fpatan", "fyl2x"):
        value = (mnemonic, x87.read(0), x87.read(1))
        x87.pop()
        x87.write(0, value)
    else:
        raise Reject
