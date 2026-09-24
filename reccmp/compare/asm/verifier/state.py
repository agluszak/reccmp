"""Per-side symbolic machine state, ABI metadata and the shared memory model."""

from __future__ import annotations

from dataclasses import (
    dataclass,
    field,
)
from typing import Callable

from reccmp.call_facts import CallFacts

from reccmp.compare.asm.model import (
    REGISTERS,
    Reject,
)
from reccmp.compare.asm.verifier.addresses import (
    Value,
    abs_stack_offset,
    flatten_mem,
    mem_disjoint,
    stack_rooted,
    unwind_spadd,
)
from reccmp.compare.diagnosis import AnalysisRecorder

FAMILIES = ("a", "b", "c", "d", "si", "di", "bp", "sp")

COMMUTATIVE_BINOPS = {"add", "and", "or", "xor", "imul"}
ASSOCIATIVE_COMMUTATIVE_BINOPS = {"add"}
ORDERED_BINOPS = {"sub", "shl", "shr", "sar", "rol", "ror"}
CARRY_BINOPS = {"adc", "sbb"}

# jcc/setcc condition codes with their operand-swap counterpart for
# a `cmp`-produced flag state. eq/ne are symmetric under a swap.
CC_CANON = {
    "e": ("eq", False),
    "ne": ("ne", False),
    "l": ("lt_s", False),
    "g": ("lt_s", True),
    "le": ("le_s", False),
    "ge": ("le_s", True),
    "b": ("lt_u", False),
    "a": ("lt_u", True),
    "be": ("le_u", False),
    "ae": ("le_u", True),
}

# Canonical flag state of every zero idiom (`xor r, r`, `sub r, r`,
# `cmp r, r`): the result is zero, CF and OF are cleared.
ZERO_FLAGS = ("cmp", ("imm", 0), ("imm", 0))

X87_CONSTANTS = {"fld1", "fldz", "fldpi", "fldl2e", "fldl2t", "fldlg2", "fldln2"}
X87_UNARY = {"fchs", "fabs", "fsqrt", "frndint", "fcos", "fsin", "ftan", "f2xm1"}


def vsort(a: Value, b: Value) -> tuple[Value, Value]:
    """Canonical order for the operands of a commutative operation."""
    return (a, b) if repr(a) <= repr(b) else (b, a)


def commutative_result(mnemonic: str, a: Value, b: Value) -> Value:
    """Canonicalize a commutative integer result.

    Integer addition is associative modulo the destination width, so flatten
    nested additions before sorting their leaves.  Flags and carry are kept as
    binary expressions by execute(): reassociation can change the flags from
    the final physical add even when the destination value is equal.
    """
    if mnemonic not in ASSOCIATIVE_COMMUTATIVE_BINOPS:
        return (mnemonic, *vsort(a, b))

    terms: list[Value] = []
    pending = [a, b]
    while pending:
        value = pending.pop()
        if isinstance(value, tuple) and value and value[0] == mnemonic:
            pending.extend(value[1:])
        else:
            terms.append(value)
    return (mnemonic, *sorted(terms, key=repr))


@dataclass
class X87Stack:
    # known[0] is st(0). Slots below the known region belong to the caller
    # (or to a callee's float return) and are addressed by (epoch, index).
    known: list[Value] = field(default_factory=list)
    deep_pops: int = 0
    epoch: int = 0

    def read(self, i: int) -> Value:
        if i < len(self.known):
            return self.known[i]
        return ("fdeep", self.epoch, self.deep_pops + i - len(self.known))

    def push(self, value: Value) -> None:
        # The physical x87 stack has 8 slots; deeper is an overflow.
        if len(self.known) >= 8:
            raise Reject
        self.known.insert(0, value)

    def pop(self) -> None:
        if self.known:
            self.known.pop(0)
        else:
            self.deep_pops += 1

    def write(self, i: int, value: Value) -> None:
        if i < len(self.known):
            self.known[i] = value
        else:
            # Writing into the unknown region cannot be modeled.
            raise Reject

    def state_key(self) -> tuple:
        return (tuple(self.known), self.deep_pops, self.epoch)


@dataclass
class SideState:
    # pylint: disable=too-many-instance-attributes
    regs: dict[str, Value] = field(
        default_factory=lambda: {f: ("init", f) for f in FAMILIES}
    )
    # Every ordinary memory read performed by this side, as (address value,
    # memory generation). Used to discharge the trap-parity obligation of a
    # one-sided load on the other side: an extra explicit load is only
    # harmless when the other side provably reads the same address at the
    # same memory generation (e.g. folded into another instruction).
    load_log: set = field(default_factory=set)
    flags: Value = ("init", "flags")
    fpu_flags: Value = ("init", "fpuflags")
    # The carry flag is tracked separately from the other integer flags:
    # inc/dec preserve CF while rewriting the rest, so a single combined
    # flag value would let e.g. `cmp a, b; inc ecx; adc ...` erase a
    # CF difference introduced by swapped cmp operands.
    carry: Value = ("init", "carry")
    x87: X87Stack = field(default_factory=X87Stack)
    # Frame-slot alpha-renaming: negative ebp displacements are replaced by
    # slot ids assigned in first-use order, so the two sides may lay out
    # their locals differently. Validated by _slots_consistent at the end.
    slot_map: dict[int, int | None] = field(default_factory=dict)
    slot_accesses: list[tuple[int, int | None]] = field(default_factory=list)
    slots_escaped: bool = False
    rename_slots: bool = True

    def slot_ref(self, disp: int, size: str, write: bool) -> Value | int:
        """Canonical key for a frame-local access. A slot becomes renamable
        only when its first access is a write (a proper lifetime start);
        a slot that is read first would let two different uninitialized
        locals appear equal, so it keeps its raw displacement."""
        self.slot_accesses.append((disp, WIDTHS.get(size)))
        if disp not in self.slot_map:
            if write:
                self.slot_map[disp] = sum(
                    1 for v in self.slot_map.values() if v is not None
                )
            else:
                self.slot_map[disp] = None
        slot = self.slot_map[disp]
        return disp if slot is None else ("slot", slot)

    def read_reg(self, name: str) -> Value:
        family, part = REGISTERS[name]
        value = self.regs[family]
        if part == "r32":
            return value
        # Reading back the part that was just inserted yields that value.
        if isinstance(value, tuple) and value and value[0] == "ins_" + part:
            return value[2]
        return (part, value)

    def write_reg(self, name: str, value: Value) -> None:
        family, part = REGISTERS[name]
        if part == "r32":
            self.regs[family] = value
            return
        old = self.regs[family]
        # Overwriting the same part again: the previous insertion is dead.
        if isinstance(old, tuple) and old and old[0] == "ins_" + part:
            old = old[1]
        self.regs[family] = ("ins_" + part, old, value)


WIDTHS = {"byte": 1, "word": 2, "dword": 4, "qword": 8, "tbyte": 10}


@dataclass(frozen=True)
class FunctionMetadata:
    """Optional PDB-derived facts that widen what the verifier can prove.

    return_kind: how the compared function returns its result —
    "void" (eax is dead at ret), "i8"/"i16" (only al/ax matter),
    "i32", "i64" (edx:eax), "float" (st0), or "unknown" (exact eax).

    call_facts: resolves a sanitized call-target name to what is known about
    calling it; an unknown register usage means ecx/edx are compared."""

    return_kind: str = "unknown"
    call_facts: Callable[[str], CallFacts | None] | None = None
    # Accept observed values z3 proves equal (project `verifier` config).
    algebraic_identities: bool = True


def register_arguments(facts: CallFacts | None) -> tuple[bool, bool]:
    """Whether ecx and edx must be treated as arguments of a call."""
    if facts is None:
        return (True, True)
    return (facts.uses_ecx is not False, facts.uses_edx is not False)


@dataclass
class Context:
    # pylint: disable=too-many-instance-attributes
    # Memory summary tag, shared by both sides: the tag of the most recent
    # store/clobber event, or the scope's initial value. Used as a block's
    # outgoing memory state and as the base tag for loads that no recorded
    # store can alias.
    gen: int | Value = 0
    # Committed memory events, newest last: (tag, access) for a store with
    # a known (address value, width, stack kind), or (tag, None) for a
    # clobber-all (call, string write, resync). A load is tagged by the
    # newest event that may alias it, so independent loads keep their tag
    # across unrelated stores.
    mem_events: list[tuple] = field(default_factory=list)
    # Values written to exact addresses by already matched stores. Virtual
    # receiver canonicalization uses this narrow forwarding table so a
    # receiver reloaded from its stable slot is equivalent to the saved value.
    # Unknown calls do not erase the identity of that slot; an explicit later
    # store to the same address replaces it.
    receiver_values: dict[tuple[Value, int | None], tuple[Value, Value]] = field(
        default_factory=dict
    )
    # Whether a pointer into this function's own frame may have escaped
    # (stored to memory or passed to a callee). Until then, memory below
    # the entry stack pointer is private scratch that no incoming pointer
    # can alias.
    stack_escaped: bool = False
    # Every symbolic node (including subexpressions) of every expression
    # that was proven equal across the two sides. Divergent register values
    # are only excused when they appear in this set (they were consumed by
    # something matched). matched_ids memoizes the DAG walk by identity.
    matched_nodes: set[Value] = field(default_factory=set)
    matched_ids: set[int] = field(default_factory=set)
    # Memoization for _tree_size, keyed by object identity. The keepalive
    # list pins the measured tuples so their ids cannot be recycled.
    size_cache: dict[int, int] = field(default_factory=dict)
    keepalive: list = field(default_factory=list)
    # When not None, every memory access performed by execute() is recorded
    # here as ("r"|"w", address value, width, is_stack_slot).
    trace: list | None = None
    # Pending callee-save register substitutions: mutable records
    # [orig family, recomp family, stack slot address, still_valid].
    # A potentially aliasing write to the slot clears still_valid.
    save_stack: list[list] = field(default_factory=list)
    # Trap-parity obligations from one-sided memory reads: (other side's
    # state, address value, memory generation). Discharged at the end of
    # the verification scope against the other side's load_log.
    load_obligations: list[tuple] = field(default_factory=list)
    # Live one-sided spills: [side state, entry-sp offset, value, event
    # tag]. Must be empty at every call: a pushed value still live at a
    # call would be an argument the other side never passed.
    scratch_pushes: list[list] = field(default_factory=list)
    # Which acceptance features fired, for debug/audit logging.
    categories: set[str] = field(default_factory=set)
    # PDB-derived return-type and callee-convention facts, if available.
    metadata: FunctionMetadata | None = None
    # Structured evidence sink for the current verifier strategy.
    recorder: AnalysisRecorder | None = None

    def __post_init__(self) -> None:
        # The memory tag of the scope's entry state: loads that precede
        # every recorded store (or that no recorded store aliases) carry it.
        self.initial_gen = self.gen

    def add_matched(self, value) -> None:
        stack = [value]
        while stack:
            node = stack.pop()
            if not isinstance(node, tuple):
                continue
            key = id(node)
            if key in self.matched_ids:
                continue
            self.matched_ids.add(key)
            self.keepalive.append(node)
            self.matched_nodes.add(node)
            stack.extend(node)


# A load only needs the newest may-aliasing store. Scanning the whole event
# log per load would be quadratic on huge straight-line functions; past the
# cap, the newest unresolved tag is a sound (merely coarser) answer.
_ALIAS_SCAN_LIMIT = 128


def memory_load_tag(ctx: Context, address: Value, width, stack) -> int | Value:
    """Tag identifying which memory state a load reads: the tag of the
    newest committed store that may alias it (or of any clobber), else the
    scope's initial memory tag. Two loads of the same address with the same
    tag are the same read."""
    access = (address, width, stack)
    events = ctx.mem_events
    scanned = 0
    for index in range(len(events) - 1, -1, -1):
        tag, store = events[index]
        scanned += 1
        if scanned > _ALIAS_SCAN_LIMIT:
            return tag
        if store is None or _store_may_alias_load(store, access, ctx.stack_escaped):
            return tag
    return ctx.initial_gen


def frame_pointer_value(value: Value) -> bool:
    """Does this value hold a pointer into the current function's own
    frame (strictly below the entry stack pointer)? Such a value reaching
    memory or a callee makes the frame externally reachable."""
    if not isinstance(value, tuple) or not value:
        return False
    if value[0] == "addr":
        resolved = abs_stack_offset(value[1], False)
        if resolved is None:
            # An escaping address we cannot resolve: assume the worst
            # when it is stack-rooted at all.
            return any(stack_rooted(term) for term, _ in value[1][2])
        root, offset = resolved
        return root == ("init", "sp") and offset < 0
    if value[0] == "spadd":
        root, offset = unwind_spadd(value)
        return root == ("init", "sp") and offset < 0
    return False


def _store_may_alias_load(store: tuple, load: tuple, stack_escaped: bool) -> bool:
    """May this committed store affect this load? Refines mem_disjoint
    with an ABI fact: while no frame pointer has escaped, memory strictly
    below the entry stack pointer is the function's private scratch, which
    no incoming (unknown) pointer can alias."""
    if mem_disjoint(store, load):
        return False
    if not stack_escaped:
        for scratch, other in ((store, load), (load, store)):
            resolved = abs_stack_offset(scratch[0], scratch[2])
            if (
                resolved is not None
                and resolved[0] == ("init", "sp")
                and resolved[1] < 0
                and not other[2]
            ):
                other_mem = flatten_mem(other[0])
                if not any(stack_rooted(term) for term, _ in other_mem[2]):
                    return False
    return True


def commit_clobber(ctx: Context, marker) -> None:
    """Record a write to unknown locations: every later load re-reads."""
    tag = ("mem", marker, "clobber")
    ctx.mem_events.append((tag, None))
    ctx.gen = tag


def commit_memory(ctx: Context, obs: list, marker) -> None:
    """Commit the memory effects of one verified instruction pair. Deferred
    until after both sides executed so that loads within the pair observe
    the same pre-instruction memory."""
    for k, entry in enumerate(obs):
        kind = entry[0]
        if kind == "store":
            _, address, size, value = entry
            width = 4 if size == "stack" else WIDTHS.get(size)
            stack = "push" if size == "stack" else False
            tag = ("mem", marker, k)
            ctx.mem_events.append((tag, (address, width, stack)))
            ctx.receiver_values[(address, width)] = (tag, value)
            ctx.gen = tag
            if frame_pointer_value(value):
                ctx.stack_escaped = True
        elif kind == "call":
            if ctx.scratch_pushes:
                # A one-sided spill still on the stack at a call would be
                # an extra argument: not provably equivalent.
                raise Reject
            for argument in entry[2:]:
                if frame_pointer_value(argument):
                    ctx.stack_escaped = True
            commit_clobber(ctx, (marker, k))
        elif isinstance(kind, tuple):
            # String instruction: (mnemonic, prefix). Writers clobber; the
            # data they copy was already committed by its original store.
            if STRING_OPS.get(kind[0], ("", "", False))[2]:
                commit_clobber(ctx, (marker, k))


# Symbolic values are DAGs (a register value can feed several later values),
# but comparison and repr expand them to trees. Reject a function once any
# tracked value grows beyond this tree size: pathological shapes like a long
# `add eax, eax` doubling chain would otherwise take exponential time.
VALUE_SIZE_LIMIT = 50_000


def _tree_size(value, ctx: Context) -> int:
    if not isinstance(value, tuple):
        return 1
    key = id(value)
    cached = ctx.size_cache.get(key)
    if cached is not None:
        return cached
    size = 1 + sum(_tree_size(child, ctx) for child in value)
    ctx.size_cache[key] = size
    ctx.keepalive.append(value)
    return size


def guard_state_size(state: SideState, ctx: Context) -> None:
    for value in (*state.regs.values(), state.flags, state.carry, state.fpu_flags):
        if _tree_size(value, ctx) > VALUE_SIZE_LIMIT:
            raise Reject
    for value in state.x87.known:
        if _tree_size(value, ctx) > VALUE_SIZE_LIMIT:
            raise Reject


JCC_MNEMONICS = {
    "ja": "a",
    "jae": "ae",
    "jb": "b",
    "jbe": "be",
    "je": "e",
    "jne": "ne",
    "jg": "g",
    "jge": "ge",
    "jl": "l",
    "jle": "le",
    "js": "s",
    "jns": "ns",
    "jo": "o",
    "jno": "no",
    "jp": "p",
    "jnp": "np",
}


STRING_OPS = {
    # mnemonic: (registers read, registers written, writes memory)
    **{f"movs{s}": ("si di", "si di", True) for s in "bwd"},
    **{f"stos{s}": ("a di", "di", True) for s in "bwd"},
    **{f"lods{s}": ("a si", "a si", False) for s in "bwd"},
    **{f"scas{s}": ("a di", "di", False) for s in "bwd"},
    **{f"cmps{s}": ("si di", "si di", False) for s in "bwd"},
}


# Observable tags of instructions that may transfer control locally.
CONTROL_TAGS = frozenset(
    {"branch", "jmp", "jmpind", "loop", "loope", "loopne", "jcxz", "jecxz"}
)


def clone_state(state: SideState) -> SideState:
    clone = SideState(rename_slots=state.rename_slots)
    clone.regs = dict(state.regs)
    clone.flags = state.flags
    clone.carry = state.carry
    clone.fpu_flags = state.fpu_flags
    clone.x87 = X87Stack(
        known=list(state.x87.known),
        deep_pops=state.x87.deep_pops,
        epoch=state.x87.epoch,
    )
    clone.slot_map = dict(state.slot_map)
    clone.slot_accesses = list(state.slot_accesses)
    clone.slots_escaped = state.slots_escaped
    clone.load_log = set(state.load_log)
    return clone
