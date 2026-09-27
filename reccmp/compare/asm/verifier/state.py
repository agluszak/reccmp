"""Per-side symbolic machine state, ABI metadata and the shared memory model."""

from __future__ import annotations

from dataclasses import dataclass, field
from collections.abc import Hashable
from enum import Enum
from typing import TYPE_CHECKING, Callable, TypeAlias

from reccmp.call_facts import CallFacts

from reccmp.compare.asm.model import (
    REGISTERS,
    Reject,
)
from reccmp.compare.asm.operand import Operand
from reccmp.compare.asm.verifier.addresses import (
    AddressValue,
    CallResult,
    CfgMemoryInit,
    CompareFlags,
    CompareKind,
    Constant,
    DeepFloat,
    Extract,
    Init,
    Insert,
    MemoryClobber,
    MemoryGeneration,
    MemoryStore,
    Operation,
    OperationKind,
    Phi,
    Resync,
    RegisterPart,
    Slot,
    StackOffset,
    StringResult,
    Value,
    X87EpochJoin,
    abs_stack_offset,
    constant_offset,
    flatten_mem,
    is_value,
    mem_disjoint,
    value_children,
    stack_rooted,
    unwind_spadd,
)
from reccmp.compare.diagnosis import AnalysisRecorder, EffectiveReason
from reccmp.types import ImageId

if TYPE_CHECKING:
    from reccmp.compare.callee_cleanup import CallStackEffect

FAMILIES = ("a", "b", "c", "d", "si", "di", "bp", "sp")


def is_scratch(value: object) -> bool:
    """Whether ``value`` has no unobserved content. Left in a caller-saved
    register while the other side holds something else, such a value marks
    the register dead."""
    return isinstance(value, (Init, CallResult, StringResult, Resync)) or (
        isinstance(value, Phi) and value.settled
    )


COMMUTATIVE_BINOPS = {"add", "and", "or", "xor", "imul"}
ASSOCIATIVE_COMMUTATIVE_BINOPS = {"add"}
ORDERED_BINOPS = {"sub", "shl", "shr", "sar", "rol", "ror"}
CARRY_BINOPS = {"adc", "sbb"}

# jcc/setcc condition codes with their operand-swap counterpart for
# a `cmp`-produced flag state. eq/ne are symmetric under a swap.
CC_CANON = {
    "e": (CompareKind.EQ, False),
    "ne": (CompareKind.NE, False),
    "l": (CompareKind.LT_S, False),
    "g": (CompareKind.LT_S, True),
    "le": (CompareKind.LE_S, False),
    "ge": (CompareKind.LE_S, True),
    "b": (CompareKind.LT_U, False),
    "a": (CompareKind.LT_U, True),
    "be": (CompareKind.LE_U, False),
    "ae": (CompareKind.LE_U, True),
}

# Canonical flag state of every zero idiom (`xor r, r`, `sub r, r`,
# `cmp r, r`): the result is zero, CF and OF are cleared.
ZERO_FLAGS = CompareFlags(Constant(0), Constant(0))

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
    kind = OperationKind(mnemonic)
    if mnemonic not in ASSOCIATIVE_COMMUTATIVE_BINOPS:
        return Operation(kind, vsort(a, b))

    terms: list[Value] = []
    pending = [a, b]
    while pending:
        value = pending.pop()
        if isinstance(value, Operation) and value.kind is kind:
            pending.extend(value.operands)
        else:
            terms.append(value)
    return Operation(kind, tuple(sorted(terms, key=repr)))


@dataclass
class X87Stack:
    # known[0] is st(0). Slots below the known region belong to the caller
    # (or to a callee's float return) and are addressed by (epoch, index).
    known: list[Value] = field(default_factory=list)
    deep_pops: int = 0
    epoch: int | X87EpochJoin = 0

    def read(self, i: int) -> Value:
        if i < len(self.known):
            return self.known[i]
        return DeepFloat(self.epoch, self.deep_pops + i - len(self.known))

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
        default_factory=lambda: {family: Init(family) for family in FAMILIES}
    )
    # Every ordinary memory read performed by this side, as (address value,
    # memory generation). Used to discharge the trap-parity obligation of a
    # one-sided load on the other side: an extra explicit load is only
    # harmless when the other side provably reads the same address at the
    # same memory generation (e.g. folded into another instruction).
    load_log: set = field(default_factory=set)
    flags: Value = Init("flags")
    fpu_flags: Value = Init("fpuflags")
    # The carry flag is tracked separately from the other integer flags:
    # inc/dec preserve CF while rewriting the rest, so a single combined
    # flag value would let e.g. `cmp a, b; inc ecx; adc ...` erase a
    # CF difference introduced by swapped cmp operands.
    carry: Value = Init("carry")
    x87: X87Stack = field(default_factory=X87Stack)
    # Frame-slot alpha-renaming: negative ebp displacements are replaced by
    # slot ids assigned in first-use order, so the two sides may lay out
    # their locals differently. Validated by _slots_consistent at the end.
    slot_map: dict[int, int | None] = field(default_factory=dict)
    slot_accesses: list[tuple[int, int | None]] = field(default_factory=list)
    slots_escaped: bool = False
    rename_slots: bool = True
    # Promotion mode (see verifier.frame): this side's private frame, as
    # entry-sp offset -> (width, value), the value None where the slot can
    # no longer be read. None when promotion is off.
    frame: dict[int, tuple[int, Value | None]] | None = None

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
        return disp if slot is None else Slot(slot)

    def read_reg(self, name: str) -> Value:
        family, part = REGISTERS[name]
        value = self.regs[family]
        if part == "r32":
            return value
        # Reading back the part that was just inserted yields that value.
        register_part = RegisterPart(part)
        if isinstance(value, Insert) and value.part is register_part:
            return value.new
        return Extract(register_part, value)

    def write_reg(self, name: str, value: Value) -> None:
        family, part = REGISTERS[name]
        if part == "r32":
            if self.frame is not None and family in ("sp", "bp"):
                # Promotion places slots by offset from the entry stack
                # pointer: keep stack pointers in one form.
                root, offset = constant_offset(value)
                if root == Init("sp"):
                    value = StackOffset(root, offset) if offset else root
            self.regs[family] = value
            return
        old = self.regs[family]
        register_part = RegisterPart(part)
        # Overwriting the same part again: the previous insertion is dead.
        if isinstance(old, Insert) and old.part is register_part:
            old = old.old
        self.regs[family] = Insert(register_part, old, value)


WIDTHS = {"byte": 1, "word": 2, "dword": 4, "qword": 8, "tbyte": 10}


@dataclass(frozen=True)
class FunctionMetadata:
    """Optional PDB-derived facts that widen what the verifier can prove.

    return_kind: how the compared function returns its result —
    "void" (eax is dead at ret), "i8"/"i16" (only al/ax matter),
    "i32", "i64" (edx:eax), "float" (st0), or "unknown" (exact eax).

    call_facts: what is known about calling a callee, by its proof identity
    (the call operand's Reference.identity); an unknown register usage
    means ecx/edx are compared."""

    return_kind: str = "unknown"
    call_facts: Callable[[Hashable], CallFacts | None] | None = None
    # Accept observed values z3 proves equal (project `verifier` config).
    algebraic_identities: bool = True
    # The stack effect of the call instruction at an address, from each
    # side's own binary (orig, recomp): see callee_cleanup.StaticCode.
    stack_effects: (
        tuple[
            Callable[[int], CallStackEffect | None],
            Callable[[int], CallStackEffect | None],
        ]
        | None
    ) = None


def register_arguments(facts: CallFacts | None) -> tuple[bool, bool]:
    """Whether ecx and edx must be treated as arguments of a call."""
    if facts is None:
        return (True, True)
    return (facts.uses_ecx is not False, facts.uses_edx is not False)


class LoopKind(Enum):
    LOOP = "loop"
    LOOPE = "loope"
    LOOPNE = "loopne"
    JCXZ = "jcxz"
    JECXZ = "jecxz"


@dataclass(frozen=True, slots=True)
class Store:
    address: Value
    size: str
    value: Value


@dataclass(frozen=True, slots=True)
class Call:
    target: Value
    arguments: tuple[Value, ...]


@dataclass(frozen=True, slots=True)
class LocalDestination:
    key: Hashable


@dataclass(frozen=True, slots=True)
class ExternalDestination:
    identity: Hashable


@dataclass(frozen=True, slots=True)
class UnresolvedDestination:
    address: int


Destination: TypeAlias = LocalDestination | ExternalDestination | UnresolvedDestination


@dataclass(frozen=True, slots=True)
class SwitchTarget:
    """Canonical target of an indirect jump through a recognized switch table."""


@dataclass(frozen=True, slots=True)
class Branch:
    predicate: Value
    destination: Destination | None


@dataclass(frozen=True, slots=True)
class IndirectJump:
    target: Value | SwitchTarget
    selector: tuple[Value, ...] = ()


@dataclass(frozen=True, slots=True)
class Jump:
    destination: Destination | None


@dataclass(frozen=True, slots=True)
class Loop:
    kind: LoopKind
    counter: Value
    flags: Value
    destination: Destination | None


@dataclass(frozen=True, slots=True)
class ReturnValue:
    values: tuple[Value, ...]


@dataclass(frozen=True, slots=True)
class ReturnFpu:
    value: Value


@dataclass(frozen=True, slots=True)
class ReturnStack:
    operands: tuple[Operand, ...]
    x87_state: tuple


@dataclass(frozen=True, slots=True)
class ReturnSaved:
    values: tuple[Value, ...]


@dataclass(frozen=True, slots=True)
class FrameArguments:
    values: tuple[tuple[int, int, Value], ...] | None


@dataclass(frozen=True, slots=True)
class StringOperation:
    mnemonic: str
    prefix: str
    values: tuple[Value, ...]
    writes_memory: bool


@dataclass(frozen=True, slots=True)
class LoadControlWord:
    value: Value


Observation: TypeAlias = (
    Store
    | Call
    | Branch
    | IndirectJump
    | Jump
    | Loop
    | ReturnValue
    | ReturnFpu
    | ReturnStack
    | ReturnSaved
    | FrameArguments
    | StringOperation
    | LoadControlWord
)
ControlObservation: TypeAlias = Branch | IndirectJump | Jump | Loop


def is_control_observation(observation: Observation) -> bool:
    return isinstance(observation, (Branch, IndirectJump, Jump, Loop))


def is_conditional_observation(observation: Observation) -> bool:
    return isinstance(observation, (Branch, Loop))


def observation_values(observation: Observation) -> tuple[Value, ...]:
    values: tuple[Value, ...] = ()
    match observation:
        case Store(address, _, value):
            values = (address, value)
        case Call(target, arguments):
            values = (target, *arguments)
        case Branch(predicate):
            values = (predicate,)
        case IndirectJump(target, selector):
            values = (
                selector if isinstance(target, SwitchTarget) else (target, *selector)
            )
        case Loop(_, counter, flags):
            values = (counter, flags)
        case ReturnValue(found) | ReturnSaved(found):
            values = found
        case ReturnFpu(value) | LoadControlWord(value):
            values = (value,)
        case FrameArguments(values=arguments) if arguments is not None:
            values = tuple(value for _, _, value in arguments)
        case StringOperation(values=found):
            values = found
    return values


@dataclass(slots=True)
class CalleeSaveSubstitution:
    orig_family: str
    recomp_family: str
    address: Value
    valid: bool = True


@dataclass(frozen=True, slots=True)
class LoadObligation:
    side: ImageId
    address: Value
    generation: MemoryGeneration


@dataclass(frozen=True, slots=True)
class ScratchPush:
    side: ImageId
    offset: int
    value: Value
    tag: MemoryGeneration


@dataclass
class Context:
    # pylint: disable=too-many-instance-attributes
    # Memory summary tag, shared by both sides: the tag of the most recent
    # store/clobber event, or the scope's initial value. Used as a block's
    # outgoing memory state and as the base tag for loads that no recorded
    # store can alias.
    gen: MemoryGeneration = field(default_factory=CfgMemoryInit)
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
    receiver_values: dict[tuple[Value, int | None], tuple[MemoryGeneration, Value]] = (
        field(default_factory=dict)
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
    # Pending callee-save register substitutions. A potentially aliasing
    # write to the slot marks the substitution invalid.
    save_stack: list[CalleeSaveSubstitution] = field(default_factory=list)
    # Trap-parity obligations from one-sided memory reads. Discharged at the
    # end of the verification scope against the other side's load_log.
    load_obligations: list[LoadObligation] = field(default_factory=list)
    # Live one-sided spills. Must be empty at every call: a pushed value still
    # live at a call would be an argument the other side never passed.
    scratch_pushes: list[ScratchPush] = field(default_factory=list)
    # Which acceptance features fired, for debug/audit logging.
    categories: set[EffectiveReason] = field(default_factory=set)
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
            if not is_value(node):
                continue
            key = id(node)
            if key in self.matched_ids:
                continue
            self.matched_ids.add(key)
            self.keepalive.append(node)
            self.matched_nodes.add(node)
            stack.extend(value_children(node))


# A load only needs the newest may-aliasing store. Scanning the whole event
# log per load would be quadratic on huge straight-line functions; past the
# cap, the newest unresolved tag is a sound (merely coarser) answer.
_ALIAS_SCAN_LIMIT = 128


def memory_load_tag(ctx: Context, address: Value, width, stack) -> MemoryGeneration:
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
    if isinstance(value, AddressValue):
        resolved = abs_stack_offset(value.address, False)
        if resolved is None:
            # An escaping address we cannot resolve: assume the worst
            # when it is stack-rooted at all.
            return any(
                stack_rooted(term.value) for term in flatten_mem(value.address).terms
            )
        root, offset = resolved
        return root == Init("sp") and offset < 0
    if isinstance(value, StackOffset):
        root, offset = unwind_spadd(value)
        return root == Init("sp") and offset < 0
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
                and resolved[0] == Init("sp")
                and resolved[1] < 0
                and not other[2]
            ):
                other_mem = flatten_mem(other[0])
                if not any(stack_rooted(term.value) for term in other_mem.terms):
                    return False
    return True


def commit_clobber(ctx: Context, tag: MemoryClobber) -> None:
    """Record a write to unknown locations: every later load re-reads."""
    ctx.mem_events.append((tag, None))
    ctx.gen = tag


def commit_memory(ctx: Context, obs: list[Observation], marker) -> None:
    """Commit the memory effects of one verified instruction pair. Deferred
    until after both sides executed so that loads within the pair observe
    the same pre-instruction memory."""
    for k, entry in enumerate(obs):
        match entry:
            case Store(address, size, value):
                width = 4 if size == "stack" else WIDTHS.get(size)
                stack = "push" if size == "stack" else False
                tag = MemoryStore(marker, k)
                ctx.mem_events.append((tag, (address, width, stack)))
                ctx.receiver_values[(address, width)] = (tag, value)
                ctx.gen = tag
                if frame_pointer_value(value):
                    ctx.stack_escaped = True
            case Call(arguments=arguments):
                if ctx.scratch_pushes:
                    # A one-sided spill still on the stack at a call would be
                    # an extra argument: not provably equivalent.
                    raise Reject
                for argument in arguments:
                    if frame_pointer_value(argument):
                        ctx.stack_escaped = True
                commit_clobber(ctx, MemoryClobber(marker, k))
            case StringOperation(writes_memory=True):
                # Writers clobber; the data they copy was already committed
                # by its original store.
                commit_clobber(ctx, MemoryClobber(marker, k))


# Symbolic values are DAGs (a register value can feed several later values),
# but comparison and repr expand them to trees. Reject a function once any
# tracked value grows beyond this tree size: pathological shapes like a long
# `add eax, eax` doubling chain would otherwise take exponential time.
VALUE_SIZE_LIMIT = 50_000


def _tree_size(value, ctx: Context) -> int:
    if not is_value(value):
        return 1
    key = id(value)
    cached = ctx.size_cache.get(key)
    if cached is not None:
        return cached
    size = 1 + sum(_tree_size(child, ctx) for child in value_children(value))
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
    clone.frame = dict(state.frame) if state.frame is not None else None
    clone.load_log = set(state.load_log)
    return clone
