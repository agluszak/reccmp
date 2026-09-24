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
import functools
import hashlib
import struct
from dataclasses import dataclass, field
from typing import Mapping

from capstone import CS_GRP_CALL, CsError, CsInsn  # type: ignore[import-untyped]
from capstone.x86 import (  # type: ignore
    X86_INS_ADD,
    X86_INS_CPUID,
    X86_INS_ENTER,
    X86_INS_IN,
    X86_INS_INSB,
    X86_INS_JMP,
    X86_INS_OUT,
    X86_INS_OUTSB,
    X86_INS_PUSH,
    X86_INS_RDRAND,
    X86_INS_RDSEED,
    X86_INS_RDTSC,
    X86_INS_RDTSCP,
    X86_INS_RET,
    X86_OP_IMM,
    X86_OP_MEM,
    X86_OP_REG,
    X86_REG_BP,
    X86_REG_EAX,
    X86_REG_EBP,
    X86_REG_EBX,
    X86_REG_ECX,
    X86_REG_EDI,
    X86_REG_EDX,
    X86_REG_ESI,
    X86_REG_ESP,
    X86_REG_SP,
)
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

from reccmp.compare.asm.decode import decode_one, direct_branch_target
from reccmp.call_facts import CallFacts
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
# Imports are bound to addresses from one registry shared by both machines,
# so the same import has the same address in both binaries.
IMPORT_BASE = 0x60000000
IMPORT_SPAN = 0x01000000
IMPORT_STRIDE = 0x10
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


def import_key(module: str, name: str) -> str:
    """One import's identity: module names are case-insensitive."""
    return f"{module.lower()}!{name}"


def _image_imports(image: Image) -> list[tuple[str, int]]:
    """(key, import table slot) of each import of an image."""
    return [
        (import_key(imp.module, imp.name or f"#{imp.ordinal}"), imp.addr)
        for imp in getattr(image, "get_imports", lambda: ())()
    ]


def import_registry(*images: Image) -> dict[str, int]:
    """A distinct modelled address for every import of the given images."""
    keys = sorted({key for image in images for key, _ in _image_imports(image)})
    if len(keys) * IMPORT_STRIDE > IMPORT_SPAN:
        raise ValueError("too many imports for the modelled import region")
    return {key: IMPORT_BASE + IMPORT_STRIDE * index for index, key in enumerate(keys)}


def _model_value(r: int, pool: tuple[int, ...]) -> int:
    """An input value from 32 random bits: small integers, 16-bit values and
    pointers into the heap, so loops terminate and dereferences land on
    shared memory. With a pool (one constant of the compared code and its
    neighbours), half the values come from it, so comparisons against it see
    both outcomes."""
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


def model_dword(seed: int, *key: int, pool: tuple[int, ...] = ()) -> int:
    return _model_value(_prng(seed, *key), pool)


# Both machines ask for the same pages in turn; the second gets them free.
@functools.lru_cache(maxsize=1024)
def page_contents(seed: int, page: int, pool: tuple[int, ...] = ()) -> bytes:
    if page < LOW_PAGES:
        # Small integers used as pointers land here; like a null page it
        # reads as zeros, so further dereferences stay in shared memory.
        return bytes(PAGE)
    words = hashlib.shake_128(struct.pack("<QQ", seed, page)).digest(PAGE)
    return struct.pack(
        f"<{PAGE // 4}I",
        *(_model_value(r, pool) for r in struct.unpack(f"<{PAGE // 4}I", words)),
    )


def call_outputs(seed: int, index: int, pool: tuple[int, ...]) -> tuple[int, ...]:
    """eax, ecx, edx and eflags after the index-th call. The callee may
    clobber every caller-saved register and the flags; both sides see the
    same values, so nothing the function does with them afterwards can
    differ because of the model."""
    eax, ecx, edx = (model_dword(seed, 3, index, reg, pool=pool) for reg in range(3))
    # CF PF AF ZF SF OF from the seed; IF and the reserved bit set.
    eflags = 0x202 | (_prng(seed, 4, index) & 0x8D5)
    return eax, ecx, edx, eflags


def pages_touched(addr: int, size: int) -> range:
    """Every page an access of ``size`` bytes at ``addr`` touches."""
    first = addr & ~(PAGE - 1)
    last = (addr + max(size, 1) - 1) & ~(PAGE - 1)
    return range(first, last + PAGE, PAGE)


@dataclass(frozen=True)
class AccessSpan:
    """The bytes one memory access touched."""

    address: int
    size: int

    @property
    def last(self) -> int:
        return self.address + max(self.size, 1) - 1

    def within(self, region: range) -> bool:
        return self.address in region and self.last in region


@dataclass(frozen=True)
class CallEvent:
    """One call the function made, in the side's own address space."""

    call_site: int
    target: int | None
    target_slot: int | None  # memory operand address for indirect calls
    ecx: int
    edx: int
    stack_args: tuple[int, ...]
    # The callee's stack cleanup was guessed from the pushes before the
    # call: esp, and everything computed from it, may be wrong afterwards.
    assumed_cleanup: bool = False


@dataclass
class Trace:
    """Everything the model lets the outside world see about one run."""

    # pylint: disable=too-many-instance-attributes

    # return, truncated, limit, fault, foreign_access, uninitialized_read,
    # pointer_bytes_read, nondeterministic, bad_stack, code_write
    end: str
    calls: list[CallEvent] = field(default_factory=list)
    # Widest store outside the stack at each address.
    writes: dict[int, int] = field(default_factory=dict)
    # Instructions of the compared function that ran.
    executed: set[int] = field(default_factory=set)
    # For each byte outside the stack: the (address, size, value was an image
    # address, calls made before it) of the store that wrote it last.
    last_writer: dict[int, tuple[int, int, bool, int]] = field(default_factory=dict)
    # Data reads inside the image; each must lie within one known object for
    # the run to be layout-independent.
    image_reads: set[AccessSpan] = field(default_factory=set)
    eax: int = 0
    edx: int = 0
    esp_after_return: int = 0
    detail: str = ""

    def mixes_pointer_bytes(self, addr: int, size: int) -> bool:
        """Bytes of a stored pointer, not read back as that same store, have
        a layout-dependent value."""
        writers = {
            w[:3] if w else None
            for w in (self.last_writer.get(b) for b in range(addr, addr + size))
        }
        if writers in ({(addr, size, True)}, {(addr, size, False)}):
            return False
        return any(w is not None and w[2] for w in writers)


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

    def __init__(
        self,
        image: Image,
        imports: Mapping[str, int] | None = None,
        import_facts: Mapping[str, CallFacts] | None = None,
    ):
        """``imports`` maps import keys to modelled addresses and must be
        the same registry for both machines of a comparison; by default it
        covers this image only. ``import_facts`` gives call facts by import
        name."""
        self.image = image
        self.import_facts = import_facts or {}
        self.uc = Uc(UC_ARCH_X86, UC_MODE_32)
        lo = min(s.virtual_address for s in image.sections) & ~(PAGE - 1)
        hi = max(s.virtual_address + s.extent for s in image.sections)
        hi = (hi + PAGE - 1) & ~(PAGE - 1)
        self.image_range = range(lo, hi)
        self.uc.mem_map(lo, hi - lo)
        for section in image.sections:
            data = bytes(section.view[: section.size_of_raw_data])
            self.uc.mem_write(section.virtual_address, data)
        registry = imports if imports is not None else import_registry(image)
        # Modelled import address -> import key, and each import table slot.
        self.imports: dict[int, str] = {}
        self.import_slots: list[range] = []
        for key, slot in _image_imports(image):
            self.imports[registry[key]] = key
            self.import_slots.append(range(slot, slot + 4))
            self.uc.mem_write(slot, registry[key].to_bytes(4, "little"))
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
        # One flag per stack byte: nonzero once the current run wrote it.
        self._written_stack = bytearray(STACK_SIZE)
        # Decoding reads the pristine image, which runs never change, so
        # results are cached per machine.
        self.insn_at = functools.cache(self._decode)
        self.callee_pop_bytes = functools.cache(self._callee_pop_bytes)
        self._seed = 0
        self._pool: tuple[int, ...] = ()
        # First address the last run touched outside all shared memory.
        self.foreign_access: int | None = None

    # -- decoding --------------------------------------------------------

    def _decode(self, addr: int) -> CsInsn | None:
        if addr not in self.image_range:
            return None
        offset = addr - self.image_range.start
        return decode_one(self._pristine[offset : offset + 16], addr)

    def _callee_pop_bytes(self, target: int) -> int | None:
        """``ret N`` of a callee in the image, following ``jmp`` thunks."""
        addr, seen = target, 0
        while addr in self.image_range and seen < 4000:
            insn = self.insn_at(addr)
            if insn is None:
                break
            seen += 1
            if insn.id == X86_INS_RET:
                return insn.operands[0].imm if insn.operands else 0
            if insn.id == X86_INS_JMP and seen == 1:
                jump = direct_branch_target(insn)
                if jump is None:
                    break
                addr = jump
                continue
            addr += insn.size
        return None

    def _pushed_bytes(self, recent: collections.deque[int]) -> int:
        """Bytes pushed on the executed path right before the current call:
        the pushes back to the first earlier instruction that changes esp or
        ebp any other way."""
        total = 0
        history = list(recent)[:-1]  # the call itself is last
        for addr in reversed(history):
            insn = self.insn_at(addr)
            if insn is None:
                break
            if insn.id == X86_INS_PUSH:
                total += insn.operands[0].size if insn.operands else 4
                continue
            if insn.id == X86_INS_ENTER:  # Capstone reports no register writes
                break
            try:
                written = set(insn.regs_access()[1])
            except CsError:
                break
            if written & _FRAME_REGISTERS:
                break
        return total

    def _caller_cleanup(self, after_call: int) -> int | None:
        """Bytes the caller removes right after the call (cdecl)."""
        insn = self.insn_at(after_call)
        if (
            insn is not None
            and insn.id == X86_INS_ADD
            and _is_reg(insn, 0, X86_REG_ESP)
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
        # Every stack store of a run is flagged, so only the span between the
        # first and last flagged byte can differ from the zeroed stack. The
        # argument frame above it is rewritten by each run.
        first = self._written_stack.find(1)
        if first >= 0:
            zeros = bytes(self._written_stack.rfind(1) + 1 - first)
            self.uc.mem_write(STACK_BASE + first, zeros)
            self._written_stack[first : first + len(zeros)] = zeros
        self._seed = seed
        self._pool = pool
        self.foreign_access = None

    def _on_unmapped(self, uc, _access, addr, size, _value, _data) -> bool:
        for page in pages_touched(addr, size):
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

    def resolve_code(self, target: int) -> int:
        """Follow ``jmp rel32`` and ``jmp [slot]`` thunks from ``target``."""
        for _ in range(4):
            if target not in self.image_range:
                break
            insn = self.insn_at(target)
            if insn is None or insn.id != X86_INS_JMP or not insn.operands:
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

    def in_import_slot(self, span: AccessSpan) -> bool:
        return any(span.within(slot) for slot in self.import_slots)

    def stack_cleanup(self, target: int | None, after_call: int) -> int | None:
        """Argument bytes the callee removes, from evidence about this
        binary: the callee's own ``ret N`` (through ``jmp`` thunks), the
        import's call facts, or the caller removing them itself."""
        if target is not None:
            resolved = self.resolve_code(target)
            if resolved in self.image_range:
                pop = self.callee_pop_bytes(resolved)
                if pop is not None:
                    return pop
            if resolved in self.imports:
                name = self.imports[resolved].split("!", 1)[1]
                facts = self.import_facts.get(name)
                if facts is not None and facts.stack_cleanup is not None:
                    return facts.stack_cleanup
        if self._caller_cleanup(after_call) is not None:
            return 0
        return None

    def in_stack(self, addr: int) -> bool:
        return STACK_BASE <= addr < STACK_BASE + STACK_SIZE

    # -- running ---------------------------------------------------------

    def run(self, func_range: range, run_input: RunInput) -> Trace:
        # pylint: disable=too-many-locals,too-many-statements
        uc = self.uc
        self._reset(run_input.seed, run_input.pool)
        trace = Trace(end="return")
        for name, value in run_input.registers.items():
            uc.reg_write(_GP[name], value)
        uc.reg_write(UC_X86_REG_EFLAGS, 0x202)
        esp = STACK_TOP
        frame = (RETURN_SENTINEL, *run_input.stack_args)
        uc.mem_write(esp, struct.pack(f"<{len(frame)}I", *frame))
        uc.reg_write(UC_X86_REG_ESP, esp)
        written_stack = self._written_stack

        def mark_stack_written(addr: int, size: int) -> None:
            offset = addr - STACK_BASE
            written_stack[offset : offset + size] = b"\1" * size

        recent: collections.deque[int] = collections.deque(maxlen=64)
        uninitialized_read: list[int] = []

        def on_write(_uc, _access, addr, size, value, _data):
            if self.in_stack(addr):
                mark_stack_written(addr, size)
                return
            span = AccessSpan(addr, size)
            if any(
                span.address <= code.stop - 1 and code.start <= span.last
                for code in self.code_ranges
            ):
                # Decoding reads the pristine image; code that changes
                # itself is not modelled.
                stop("code_write", f"{addr:#x}")
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
            self._dirty_image_pages.update(
                page for page in pages_touched(addr, size) if page in self.image_range
            )

        def on_read(_uc, _access, addr, size, _value, _data):
            trace.image_reads.add(AccessSpan(addr, size))

        pointer_bytes_read: list[int] = []

        def on_heap_read(_uc, _access, addr, size, _value, _data):
            # Reading part of a stored pointer (or bytes of several stores
            # that include a pointer) yields a layout-dependent value.
            if pointer_bytes_read or addr in self.image_range or self.in_stack(addr):
                return
            if trace.mixes_pointer_bytes(addr, size):
                pointer_bytes_read.append(addr)

        def on_stack_read(_uc, _access, addr, size, _value, _data):
            # A local read before anything wrote it holds stale data whose
            # position depends on each side's frame layout.
            offset = addr - STACK_BASE
            if not uninitialized_read and 0 in written_stack[offset : offset + size]:
                uninitialized_read.append(addr)

        def stop(reason: str, detail: str = "") -> None:
            trace.end, trace.detail = reason, detail
            uc.emu_stop()

        def on_code(_uc, addr, _size, _data):
            trace.executed.add(addr)
            recent.append(addr)
            insn = self.insn_at(addr)
            if insn is not None and insn.id in NONDETERMINISTIC:
                stop("nondeterministic", insn.mnemonic)
                return
            if insn is None or not insn.group(CS_GRP_CALL):
                return
            op = insn.operands[0]
            target: int | None = direct_branch_target(insn)
            slot: int | None = None
            if op.type == X86_OP_REG:
                target = uc.reg_read(_UC_REGS[op.reg])
            elif op.type == X86_OP_MEM:
                slot = _effective_address(uc, op)
                target = int.from_bytes(uc.mem_read(slot, 4), "little")
            after = addr + insn.size
            pop = self.stack_cleanup(target, after)
            assumed = pop is None
            if pop is None:
                # Unknown callee: assume it pops what was pushed just before
                # the call, and keep running for coverage. The comparison
                # gives no verdict on anything after such a call.
                pop = self._pushed_bytes(recent)
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
                    assumed,
                )
            )
            index = len(trace.calls) - 1
            # Out-parameters: the callee writes a seeded dword through each
            # argument pointing into the caller's frame, independent of where
            # each side placed the local.
            for arg_index, value in enumerate(args):
                if STACK_BASE <= value < STACK_TOP and value >= cur_esp:
                    fill = model_dword(self._seed, 5, index, arg_index, pool=self._pool)
                    uc.mem_write(value, fill.to_bytes(4, "little"))
                    mark_stack_written(value, 4)
            eax, ecx, edx, eflags = call_outputs(self._seed, index, self._pool)
            uc.reg_write(UC_X86_REG_EAX, eax)
            uc.reg_write(UC_X86_REG_ECX, ecx)
            uc.reg_write(UC_X86_REG_EDX, edx)
            uc.reg_write(UC_X86_REG_EFLAGS, eflags)
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
# An instruction writing one of these ends the run of argument pushes.
_FRAME_REGISTERS = frozenset({X86_REG_ESP, X86_REG_SP, X86_REG_EBP, X86_REG_BP})
# fs:[0], the SEH exception list head: fs has base 0 in the emulator, so SEH
# frame registration writes here. It is bookkeeping, not program output.
SEH_CHAIN = range(0, 4)
# Instructions whose result depends on the host, not on the inputs.
NONDETERMINISTIC = frozenset(
    {
        X86_INS_RDTSC,
        X86_INS_RDTSCP,
        X86_INS_CPUID,
        X86_INS_RDRAND,
        X86_INS_RDSEED,
        X86_INS_IN,
        X86_INS_OUT,
        X86_INS_INSB,
        X86_INS_OUTSB,
    }
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


# Capstone register -> Unicorn register, for operands evaluated at run time.
_UC_REGS = {
    X86_REG_EAX: UC_X86_REG_EAX,
    X86_REG_EBX: UC_X86_REG_EBX,
    X86_REG_ECX: UC_X86_REG_ECX,
    X86_REG_EDX: UC_X86_REG_EDX,
    X86_REG_ESI: UC_X86_REG_ESI,
    X86_REG_EDI: UC_X86_REG_EDI,
    X86_REG_EBP: UC_X86_REG_EBP,
    X86_REG_ESP: UC_X86_REG_ESP,
}


def _is_reg(insn: CsInsn, index: int, *regs: int) -> bool:
    """Whether operand ``index`` of ``insn`` is one of the registers ``regs``."""
    operands = insn.operands
    return (
        len(operands) > index
        and operands[index].type == X86_OP_REG
        and operands[index].reg in regs
    )


def _effective_address(uc: Uc, op) -> int:
    mem = op.mem
    addr = mem.disp
    if mem.base:
        addr += uc.reg_read(_UC_REGS[mem.base])
    if mem.index:
        addr += uc.reg_read(_UC_REGS[mem.index]) * mem.scale
    return addr & 0xFFFFFFFF
