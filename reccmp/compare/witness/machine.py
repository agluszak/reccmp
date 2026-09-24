"""Emulate one function of one binary under the witness execution model.

The model, identical for the original and the recompiled side:

- The whole image is mapped at its preferred base, so code and initialized
  data read as they do in the file.
- Any other address is backed on first touch by a page whose contents are a
  deterministic function of the seed and the page address. Reads never fault.
- The function starts with the same registers and stack arguments on both
  sides and returns into a sentinel address.
- Calls are not executed. Each call is recorded as an event and returns a
  value that depends only on the seed and the call's position, with no
  effect on memory. The callee's stack cleanup comes from its ``ret N``
  when the callee is in the image, or from the caller's own cleanup.
  Otherwise the run stops at that call ("truncated").
"""

from __future__ import annotations

import collections
import hashlib
import struct
from dataclasses import dataclass, field
from typing import Callable

from capstone import (  # type: ignore[import-untyped]
    CS_ARCH_X86,
    CS_MODE_32,
    Cs,
    CsInsn,
)
from capstone.x86 import X86_OP_IMM, X86_OP_MEM, X86_OP_REG  # type: ignore
from unicorn import (  # type: ignore[import-untyped]
    UC_ARCH_X86,
    UC_HOOK_BLOCK,
    UC_HOOK_CODE,
    UC_HOOK_MEM_READ,
    UC_HOOK_MEM_UNMAPPED,
    UC_HOOK_MEM_WRITE,
    UC_MODE_32,
    Uc,
    UcError,
)
from unicorn.x86_const import (  # type: ignore[import-untyped]
    UC_X86_REG_EAX,
    UC_X86_REG_EBP,
    UC_X86_REG_EBX,
    UC_X86_REG_ECX,
    UC_X86_REG_EDI,
    UC_X86_REG_EDX,
    UC_X86_REG_EFLAGS,
    UC_X86_REG_EIP,
    UC_X86_REG_ESI,
    UC_X86_REG_ESP,
)

from reccmp.formats import Image
from reccmp.formats.image import ImageSectionFlags

PAGE = 0x1000
STACK_BASE = 0x7F000000
STACK_SIZE = 0x100000
STACK_TOP = STACK_BASE + STACK_SIZE - 0x1000
RETURN_SENTINEL = 0x7FFE0000
# The modelled heap is a set of objects: HEAP_OBJECTS windows of OBJECT_SIZE
# bytes, OBJECT_STRIDE apart. Pointer inputs point at object bases.
HEAP_BASE = 0x20000000
HEAP_OBJECTS = 256
OBJECT_STRIDE = 0x10000
OBJECT_SIZE = 0x4000
# Imports are bound to addresses derived from their names, so the same
# import has the same address in both binaries.
IMPORT_BASE = 0x60000000
IMPORT_SPAN = 0x01000000
STACK_ARG_DWORDS = 16
INSTRUCTION_LIMIT = 200_000
# Bounds for one run; exceeding either ends it without a verdict.
LAZY_PAGE_LIMIT = 2048
# Values this close to the image but outside it come from indexing an image
# object out of bounds; their meaning depends on the binary's layout.
NEAR_IMAGE = 0x01000000
TIME_LIMIT_US = 2_000_000

_GP = {
    "eax": UC_X86_REG_EAX,
    "ebx": UC_X86_REG_EBX,
    "ecx": UC_X86_REG_ECX,
    "edx": UC_X86_REG_EDX,
    "esi": UC_X86_REG_ESI,
    "edi": UC_X86_REG_EDI,
    "ebp": UC_X86_REG_EBP,
}


def _prng(seed: int, *key: int) -> int:
    digest = hashlib.blake2b(
        struct.pack(f"<{len(key) + 1}Q", seed, *key), digest_size=4
    ).digest()
    return int.from_bytes(digest, "little")


def import_address(module: str, name: str) -> int:
    digest = hashlib.blake2b(f"{module.lower()}!{name}".encode(), digest_size=4)
    return IMPORT_BASE + (
        int.from_bytes(digest.digest(), "little") % IMPORT_SPAN & ~0xF
    )


def model_dword(seed: int, *key: int, pool: tuple[int, ...] = ()) -> int:
    """An input value: small integers, 16-bit values and pointers into the
    heap, so loops terminate and dereferences land on shared memory. With a
    pool (one constant of the compared code and its neighbours), half the
    values come from it, so comparisons against it see both outcomes."""
    r = _prng(seed, *key)
    if pool and r >> 31:
        return pool[(r >> 8) % len(pool)]
    kind = r & 3
    if kind == 0:
        return (r >> 8) & 0xF
    if kind == 1:
        return HEAP_BASE + ((r >> 4) % HEAP_OBJECTS) * OBJECT_STRIDE
    if kind == 2:
        return (r >> 8) & 0xFF
    # Wide enough to vary arithmetic, and a dereference still lands in the
    # low region both sides address identically.
    return (r >> 8) & 0xFFFF


def page_contents(seed: int, page: int, pool: tuple[int, ...] = ()) -> bytes:
    if page < LOW_PAGES:
        # Small integers used as pointers land here; like a null page it
        # reads as zeros, so further dereferences stay in shared memory.
        return bytes(PAGE)
    return b"".join(
        model_dword(seed, page, i, pool=pool).to_bytes(4, "little")
        for i in range(PAGE // 4)
    )


@dataclass(frozen=True)
class CallEvent:
    """One call the function made, in the side's own address space."""

    call_site: int
    target: int | None
    target_slot: int | None  # memory operand address for indirect calls
    ecx: int
    edx: int
    stack_args: tuple[int, ...]


@dataclass
class Trace:
    """Everything the model lets the outside world see about one run."""

    # pylint: disable=too-many-instance-attributes

    # return, truncated, limit, fault, foreign_access, uninitialized_read,
    # pointer_bytes_read, nondeterministic, bad_stack
    end: str
    calls: list[CallEvent] = field(default_factory=list)
    # (address, size) of every store outside the stack.
    writes: dict[int, int] = field(default_factory=dict)
    # Calls whose stack cleanup was assumed from the pushes before them.
    assumed_cleanup: int = 0
    # Instructions of the compared function that ran.
    executed: set[int] = field(default_factory=set)
    # For each byte outside the stack: the (address, size, value was an image
    # address, calls made before it) of the store that wrote it last.
    last_writer: dict[int, tuple[int, int, bool, int]] = field(default_factory=dict)
    # Data reads inside the image; each must hit a known object for the run
    # to be layout-independent.
    image_reads: set[int] = field(default_factory=set)
    eax: int = 0
    edx: int = 0
    esp_after_return: int = 0
    detail: str = ""


@dataclass(frozen=True)
class RunInput:
    seed: int
    registers: dict[str, int]
    stack_args: tuple[int, ...]
    # Constants the modelled memory and call results may draw from.
    pool: tuple[int, ...] = ()

    @classmethod
    def from_seed(cls, seed: int, pool: tuple[int, ...] = ()) -> "RunInput":
        regs = {name: model_dword(seed, 1, i, pool=pool) for i, name in enumerate(_GP)}
        # ``this`` for thiscall members: always a heap object.
        regs["ecx"] = HEAP_BASE + (seed % HEAP_OBJECTS) * OBJECT_STRIDE
        # Arguments are often object pointers: make half of them heap objects.
        args = tuple(
            (
                HEAP_BASE + (_prng(seed, 6, i) % HEAP_OBJECTS) * OBJECT_STRIDE
                if _prng(seed, 7, i) & 1
                else model_dword(seed, 2, i, pool=pool)
            )
            for i in range(STACK_ARG_DWORDS)
        )
        return cls(seed, regs, args, pool)


class SideMachine:
    """A Unicorn instance holding one binary, reusable across runs."""

    # pylint: disable=too-many-instance-attributes

    def __init__(self, image: Image, import_cleanup: dict[str, int] | None = None):
        self.image = image
        # Argument bytes popped by imported callees, by import name.
        self.import_cleanup = import_cleanup or {}
        self.uc = Uc(UC_ARCH_X86, UC_MODE_32)
        self.cs = Cs(CS_ARCH_X86, CS_MODE_32)
        self.cs.detail = True
        self._image_pages: dict[int, bytes] = {}
        lo = min(s.virtual_address for s in image.sections) & ~(PAGE - 1)
        hi = max(s.virtual_address + s.extent for s in image.sections)
        hi = (hi + PAGE - 1) & ~(PAGE - 1)
        self.image_range = range(lo, hi)
        self.uc.mem_map(lo, hi - lo)
        for section in image.sections:
            data = bytes(section.view[: section.size_of_raw_data])
            self.uc.mem_write(section.virtual_address, data)
        self.imports: dict[int, str] = {}
        self.import_slots: set[int] = set()
        for imp in getattr(image, "get_imports", lambda: ())():
            name = imp.name or f"#{imp.ordinal}"
            fake = import_address(imp.module, name)
            self.imports[fake] = f"{imp.module}!{name}"
            self.import_slots.update(range(imp.addr, imp.addr + 4))
            self.uc.mem_write(imp.addr, fake.to_bytes(4, "little"))
        self._pristine = bytes(self.uc.mem_read(lo, hi - lo))
        self.code_ranges = tuple(
            section.virtual_range
            for section in image.sections
            if ImageSectionFlags.EXECUTE in section.flags
        )
        self.uc.mem_map(STACK_BASE, STACK_SIZE)
        self.uc.mem_map(RETURN_SENTINEL, PAGE)
        self._lazy_pages: set[int] = set()
        self._dirty_image_pages: set[int] = set()
        self._insn_cache: dict[int, CsInsn | None] = {}
        self._pop_cache: dict[int, int | None] = {}
        self._seed = 0
        self._pool: tuple[int, ...] = ()
        # First address the last run touched outside all shared memory.
        self.foreign_access: int | None = None

    # -- decoding --------------------------------------------------------

    def insn_at(self, addr: int) -> CsInsn | None:
        if addr not in self._insn_cache:
            try:
                code = bytes(self.uc.mem_read(addr, 16))
            except UcError:
                code = b""
            self._insn_cache[addr] = next(self.cs.disasm(code, addr, 1), None)
        return self._insn_cache[addr]

    def callee_pop_bytes(self, target: int) -> int | None:
        """``ret N`` of a callee in the image, following ``jmp`` thunks."""
        if target in self._pop_cache:
            return self._pop_cache[target]
        result = None
        addr, seen = target, 0
        while addr in self.image_range and seen < 4000:
            insn = self.insn_at(addr)
            if insn is None:
                break
            seen += 1
            if insn.mnemonic == "ret":
                result = insn.operands[0].imm if insn.operands else 0
                break
            if insn.mnemonic == "jmp" and seen == 1:
                op = insn.operands[0]
                if op.type == X86_OP_IMM:
                    addr = op.imm
                    continue
                break
            addr += insn.size
        self._pop_cache[target] = result
        return result

    def _pushed_bytes(self, recent: collections.deque[int]) -> int:
        """Bytes pushed on the executed path right before the current call."""
        total = 0
        history = list(recent)[:-1]  # the call itself is last
        for addr in reversed(history):
            insn = self.insn_at(addr)
            if insn is None:
                break
            if insn.mnemonic == "push":
                total += 4
                continue
            written = insn.op_str.split(",", 1)[0].strip()
            if insn.mnemonic in _STACK_BARRIERS or written in ("esp", "ebp"):
                break
        return total

    def _caller_cleanup(self, after_call: int) -> int | None:
        """Bytes the caller removes right after the call (cdecl)."""
        insn = self.insn_at(after_call)
        if insn is None:
            return None
        if (
            insn.mnemonic == "add"
            and len(insn.operands) == 2
            and insn.operands[0].type == X86_OP_REG
            and insn.reg_name(insn.operands[0].reg) == "esp"
            and insn.operands[1].type == X86_OP_IMM
        ):
            return insn.operands[1].imm
        return None

    # -- memory model ----------------------------------------------------

    def _reset(self, seed: int, pool: tuple[int, ...] = ()) -> None:
        for page in self._lazy_pages:
            self.uc.mem_unmap(page, PAGE)
        self._lazy_pages.clear()
        lo = self.image_range.start
        for page in self._dirty_image_pages:
            off = page - lo
            self.uc.mem_write(page, self._pristine[off : off + PAGE])
        self._dirty_image_pages.clear()
        self.uc.mem_write(STACK_BASE, b"\0" * STACK_SIZE)
        self._seed = seed
        self._pool = pool
        self.foreign_access = None

    def _on_unmapped(self, uc, _access, addr, size, _value, _data) -> bool:
        first = addr & ~(PAGE - 1)
        last = (addr + max(size, 1) - 1) & ~(PAGE - 1)
        for page in range(first, last + PAGE, PAGE):
            if page in self._lazy_pages:
                continue
            if len(self._lazy_pages) >= LAZY_PAGE_LIMIT:
                return False
            if not _shared_region(page):
                # An address outside everything both sides share may have
                # been computed from an image address, so its contents would
                # depend on each binary's layout. Leave it unmapped: the run
                # faults and gives no verdict.
                self.foreign_access = addr
                return False
            try:
                uc.mem_map(page, PAGE)
            except UcError:
                return False
            uc.mem_write(page, page_contents(self._seed, page, self._pool))
            self._lazy_pages.add(page)
        return True

    def code_constants(self, func_range: range) -> set[int]:
        """Immediates in a function body that are not addresses."""
        found: set[int] = set()
        addr = func_range.start
        while addr < func_range.stop:
            insn = self.insn_at(addr)
            if insn is None:
                break
            for op in insn.operands:
                if op.type == X86_OP_IMM and insn.mnemonic not in _BRANCHES:
                    value = op.imm & 0xFFFFFFFF
                    if value not in self.image_range:
                        found.add(value)
            addr += insn.size
        return found

    def resolve_code(self, target: int) -> int:
        """Follow ``jmp rel32`` and ``jmp [slot]`` thunks from ``target``."""
        for _ in range(4):
            if target not in self.image_range:
                break
            insn = self.insn_at(target)
            if insn is None or insn.mnemonic != "jmp" or not insn.operands:
                break
            op = insn.operands[0]
            if op.type == X86_OP_IMM:
                target = op.imm
            elif op.type == X86_OP_MEM and not op.mem.base and not op.mem.index:
                target = self.read(op.mem.disp & 0xFFFFFFFF, 4)
            else:
                break
        return target

    def in_code(self, addr: int) -> bool:
        return any(addr in r for r in self.code_ranges)

    def in_stack(self, addr: int) -> bool:
        return STACK_BASE <= addr < STACK_BASE + STACK_SIZE

    # -- running ---------------------------------------------------------

    def run(
        self,
        func_range: range,
        run_input: RunInput,
        call_result: Callable[[int], tuple[int, int]],
    ) -> Trace:
        # pylint: disable=too-many-locals,too-many-statements
        uc = self.uc
        self._reset(run_input.seed, run_input.pool)
        trace = Trace(end="return")
        for name, value in run_input.registers.items():
            uc.reg_write(_GP[name], value)
        uc.reg_write(UC_X86_REG_EFLAGS, 0x202)
        esp = STACK_TOP
        uc.mem_write(esp, RETURN_SENTINEL.to_bytes(4, "little"))
        for i, value in enumerate(run_input.stack_args):
            uc.mem_write(esp + 4 + 4 * i, value.to_bytes(4, "little"))
        uc.reg_write(UC_X86_REG_ESP, esp)
        lo = self.image_range.start

        written_stack: set[int] = set()
        recent: collections.deque[int] = collections.deque(maxlen=64)
        uninitialized_read: list[int] = []

        def on_write(_uc, _access, addr, size, value, _data):
            if self.in_stack(addr):
                written_stack.update(range(addr, addr + size))
                return
            trace.writes[addr] = max(size, trace.writes.get(addr, 0))
            writer = (
                addr,
                size,
                (value & 0xFFFFFFFF) in self.image_range,
                len(trace.calls),
            )
            for byte in range(addr, addr + size):
                trace.last_writer[byte] = writer
            if addr in self.image_range:
                self._dirty_image_pages.add(lo + ((addr - lo) & ~(PAGE - 1)))

        def on_read(_uc, _access, addr, _size, _value, _data):
            trace.image_reads.add(addr)

        pointer_bytes_read: list[int] = []

        def on_heap_read(_uc, _access, addr, size, _value, _data):
            # Reading part of a stored pointer (or bytes of several stores
            # that include a pointer) yields a layout-dependent value.
            if pointer_bytes_read or addr in self.image_range or self.in_stack(addr):
                return
            writers = {
                w[:3] if w else None
                for w in (trace.last_writer.get(b) for b in range(addr, addr + size))
            }
            if writers in ({(addr, size, True)}, {(addr, size, False)}):
                return
            if any(w is not None and w[2] for w in writers):
                pointer_bytes_read.append(addr)

        def on_stack_read(_uc, _access, addr, size, _value, _data):
            # A local read before anything wrote it holds stale data whose
            # position depends on each side's frame layout.
            if not uninitialized_read and any(
                byte not in written_stack for byte in range(addr, addr + size)
            ):
                uninitialized_read.append(addr)

        def stop(reason: str, detail: str = "") -> None:
            trace.end, trace.detail = reason, detail
            uc.emu_stop()

        def on_code(_uc, addr, _size, _data):
            trace.executed.add(addr)
            recent.append(addr)
            insn = self.insn_at(addr)
            if insn is not None and insn.mnemonic in NONDETERMINISTIC:
                stop("nondeterministic", insn.mnemonic)
                return
            if insn is None or insn.mnemonic != "call":
                return
            op = insn.operands[0]
            target: int | None = None
            slot: int | None = None
            if op.type == X86_OP_IMM:
                target = op.imm
            elif op.type == X86_OP_REG:
                target = uc.reg_read(_reg_id(insn.reg_name(op.reg)))
            elif op.type == X86_OP_MEM:
                slot = _effective_address(uc, insn, op)
                target = int.from_bytes(uc.mem_read(slot, 4), "little")
            after = addr + insn.size
            pop = None
            if target is not None and target in self.image_range:
                pop = self.callee_pop_bytes(self.resolve_code(target))
            if pop is None and target in self.imports:
                pop = self.import_cleanup.get(self.imports[target].split("!", 1)[1])
            if pop is None and self._caller_cleanup(after) is not None:
                pop = 0
            if pop is None:
                # Unknown callee: assume it pops what was pushed just before
                # the call. A wrong guess leaves esp wrong at return in
                # frame-pointer-less code, which gives no verdict.
                pop = self._pushed_bytes(recent)
                trace.assumed_cleanup += 1
            cur_esp = uc.reg_read(UC_X86_REG_ESP)
            arg_bytes = pop if pop else (self._caller_cleanup(after) or 0)
            args = tuple(
                int.from_bytes(uc.mem_read(cur_esp + 4 * i, 4), "little")
                for i in range(arg_bytes // 4)
            )
            trace.calls.append(
                CallEvent(
                    addr,
                    target,
                    slot,
                    uc.reg_read(UC_X86_REG_ECX),
                    uc.reg_read(UC_X86_REG_EDX),
                    args,
                )
            )
            if pop is None:
                stop("truncated", f"unknown stack cleanup at {addr:#x}")
                return
            index = len(trace.calls) - 1
            # Out-parameters: the callee writes a seeded dword through each
            # argument pointing into the caller's frame, independent of where
            # each side placed the local.
            for arg_index, value in enumerate(args):
                if STACK_BASE <= value < STACK_TOP and value >= cur_esp:
                    fill = model_dword(self._seed, 5, index, arg_index, pool=self._pool)
                    uc.mem_write(value, fill.to_bytes(4, "little"))
                    written_stack.update(range(value, value + 4))
            eax, edx = call_result(index)
            uc.reg_write(UC_X86_REG_EAX, eax)
            uc.reg_write(UC_X86_REG_EDX, edx)
            uc.reg_write(UC_X86_REG_ESP, cur_esp + pop)
            uc.reg_write(UC_X86_REG_EIP, after)

        def on_block(_uc, addr, _size, _data):
            if addr in func_range or addr == RETURN_SENTINEL:
                return
            # Control left the function other than through ret: a tail call
            # or a jump into shared code. Recorded as a call (the target may
            # overwrite anything stored so far) and the run ends there.
            trace.calls.append(
                CallEvent(
                    addr,
                    addr,
                    None,
                    uc.reg_read(UC_X86_REG_ECX),
                    uc.reg_read(UC_X86_REG_EDX),
                    (),
                )
            )
            stop("truncated", f"left function to {addr:#x}")

        hooks = [
            uc.hook_add(UC_HOOK_MEM_UNMAPPED, self._on_unmapped),
            uc.hook_add(UC_HOOK_MEM_WRITE, on_write),
            uc.hook_add(
                UC_HOOK_MEM_READ, on_stack_read, begin=STACK_BASE, end=STACK_TOP - 1
            ),
            uc.hook_add(UC_HOOK_MEM_READ, on_heap_read),
            uc.hook_add(
                UC_HOOK_MEM_READ,
                on_read,
                begin=self.image_range.start,
                end=self.image_range.stop - 1,
            ),
            uc.hook_add(
                UC_HOOK_CODE, on_code, begin=func_range.start, end=func_range.stop - 1
            ),
            uc.hook_add(UC_HOOK_BLOCK, on_block),
        ]
        try:
            uc.emu_start(
                func_range.start,
                RETURN_SENTINEL,
                timeout=TIME_LIMIT_US,
                count=INSTRUCTION_LIMIT,
            )
        except UcError as ex:
            trace.end, trace.detail = "fault", str(ex)
        finally:
            for hook in hooks:
                uc.hook_del(hook)
        if self.foreign_access is not None and trace.end == "fault":
            trace.end = "foreign_access"
        if uninitialized_read and trace.end in ("return", "truncated"):
            trace.end = "uninitialized_read"
        if pointer_bytes_read and trace.end in ("return", "truncated"):
            trace.end = "pointer_bytes_read"
        if trace.end == "return":
            if uc.reg_read(UC_X86_REG_EIP) != RETURN_SENTINEL:
                trace.end = "limit"
            else:
                trace.eax = uc.reg_read(UC_X86_REG_EAX)
                trace.edx = uc.reg_read(UC_X86_REG_EDX)
                trace.esp_after_return = uc.reg_read(UC_X86_REG_ESP)
                if not STACK_TOP + 4 <= trace.esp_after_return <= STACK_TOP + 0x104:
                    trace.end = "bad_stack"
        return trace

    def read(self, addr: int, size: int) -> int:
        """Read after a run; untouched addresses get their modelled contents."""
        try:
            data = self.uc.mem_read(addr, size)
        except UcError:
            self._on_unmapped(self.uc, None, addr, size, 0, None)
            data = self.uc.mem_read(addr, size)
        return int.from_bytes(data, "little")


# Small integers used as pointers (including null-based arithmetic).
LOW_PAGES = 0x100000
# Instructions that end the run of argument pushes before a call.
_STACK_BARRIERS = frozenset(
    {"call", "pop", "leave", "ret", "pushal", "popal", "pushfd", "popfd", "enter"}
)
# fs:[0], the SEH exception list head: fs has base 0 in the emulator, so SEH
# frame registration writes here. It is bookkeeping, not program output.
SEH_CHAIN = range(0, 4)
# Instructions whose result depends on the host, not on the inputs.
NONDETERMINISTIC = frozenset(
    {"rdtsc", "rdtscp", "cpuid", "rdrand", "rdseed", "in", "out", "insb", "outsb"}
)

_BRANCHES = frozenset({"call", "jmp", "ret", "loop", "jecxz", "jcxz"}) | frozenset(
    f"j{cc}" for cc in "o no b ae e ne be a s ns p np l ge le g".split()
)


def _shared_region(addr: int) -> bool:
    """Memory both sides reach through the same pointer values: the low
    region (small integers used as pointers), heap objects and bound
    imports. Addresses computed from image addresses rarely land here."""
    if addr < LOW_PAGES or IMPORT_BASE <= addr < IMPORT_BASE + IMPORT_SPAN:
        return True
    offset = addr - HEAP_BASE
    return (
        0 <= offset < HEAP_OBJECTS * OBJECT_STRIDE
        and offset % OBJECT_STRIDE < OBJECT_SIZE
    )


_REG_IDS = {
    **_GP,
    "esp": UC_X86_REG_ESP,
}


def _reg_id(name: str) -> int:
    return _REG_IDS[name]


def _effective_address(uc: Uc, insn: CsInsn, op) -> int:
    mem = op.mem
    addr = mem.disp
    if mem.base:
        addr += uc.reg_read(_reg_id(insn.reg_name(mem.base)))
    if mem.index:
        addr += uc.reg_read(_reg_id(insn.reg_name(mem.index))) * mem.scale
    return addr & 0xFFFFFFFF
