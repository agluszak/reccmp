"""What a comparison run says about each requested function.

An empty decompiled diff is not an equivalence proof, and a changed
decompilation is not a demonstrated bug. The outcomes only say whether the
analyzed output shows a difference, and whether the analysis happened.
"""

import difflib
import enum
from collections import Counter
from dataclasses import dataclass, replace
from ghidriff.code import body_change_kind, decompilation_parts

from reccmp.compare.manifest import FunctionEntry
from reccmp.types import ImageId
from .build_context import is_source_path


class Outcome(enum.Enum):
    # Code or referenced data differs; review the evidence.
    DIFFERENCES = "differences"
    # The requested analysis completed without a visible difference.
    NO_DIFFERENCES = "no-differences"
    # Correspondence is missing or ambiguous.
    UNPAIRED = "unpaired"
    # The requested comparison was not completed.
    ANALYSIS_FAILED = "analysis-failed"


class FailureKind(enum.Enum):
    # Ghidra has no function at the entry and could not create one.
    NO_FUNCTION = "no-function"
    # The entry lies inside a function Ghidra starts elsewhere.
    ENTRY_CONFLICT = "entry-conflict"
    # The decompiler returned an error instead of code.
    DECOMPILE_ERROR = "decompile-error"
    # The differ produced no decompilation for this side.
    NOT_DECOMPILED = "not-decompiled"
    SWITCH_ANALYSIS = "switch-analysis"


@dataclass(frozen=True)
class AnalysisFailure:
    kind: FailureKind
    image: ImageId
    # Address of the conflicting function, for ENTRY_CONFLICT.
    other_function: int | None = None
    message: str | None = None


@dataclass(frozen=True)
class StringValue:
    text: str


@dataclass(frozen=True)
class ObjectOffset:
    """A location inside a paired catalog object, named by its identity."""

    orig_addr: int
    name: str
    offset: int


@dataclass(frozen=True)
class PointerValue:
    # None when the pointee is neither a named object nor a string.
    target: ObjectOffset | StringValue | None


@dataclass(frozen=True)
class RawBytes:
    """Uninterpreted bytes. ``relocated`` means the range holds an address,
    whose value differs between binaries by construction. Without a catalog
    extent, the length is only as far as Ghidra's data typing reaches on
    that side, so it may differ from the other side's."""

    data: bytes
    relocated: bool
    extent_known: bool


@dataclass(frozen=True)
class Uninitialized:
    """Memory with no initial contents (``.bss``)."""


@dataclass(frozen=True)
class UnknownExtent:
    """Neither the catalog nor Ghidra says how large the location is, so
    there are no contents to show or compare."""


@dataclass(frozen=True)
class PastEnd:
    """The address just past a paired object, compared with as a bound. What
    starts there is whatever the linker placed next, not something the code
    refers to."""


Contents = (
    StringValue | PointerValue | RawBytes | Uninitialized | UnknownExtent | PastEnd
)


def contents_comparable(contents: Contents) -> bool:
    """Whether equal or unequal contents mean anything across binaries."""
    match contents:
        case (
            PointerValue(target=None)
            | RawBytes(relocated=True)
            | UnknownExtent()
            | PastEnd()
        ):
            return False
    return True


@dataclass(frozen=True)
class DataReference:
    """One data location a function refers to."""

    address: int
    # The paired object the location belongs to; None when the catalog has
    # no correspondence for it.
    object: ObjectOffset | None
    contents: Contents


class DataFindingKind(enum.Enum):
    # A paired object both sides refer to has different contents.
    OBJECT_CONTENTS = "object-contents"
    # The rest of the data the two sides refer to holds different contents.
    REFERENCED_CONTENTS = "referenced-contents"


@dataclass(frozen=True)
class DataFinding:
    kind: DataFindingKind
    # Set for OBJECT_CONTENTS.
    object: ObjectOffset | None
    orig: tuple[Contents, ...]
    recomp: tuple[Contents, ...]


def compare_references(
    orig: tuple[DataReference, ...], recomp: tuple[DataReference, ...]
) -> tuple[DataFinding, ...]:
    """Compare the contents of the data both sides refer to.

    A paired object both sides refer to is compared by identity. References
    to different paired objects are visible by name in the code; their contents
    do not describe corresponding data. Only unidentified references are
    compared as a multiset, to flag different literals without claiming which
    location corresponds to which.
    """
    findings: list[DataFinding] = []

    def by_identity(refs: tuple[DataReference, ...]) -> dict[ObjectOffset, Contents]:
        return {ref.object: ref.contents for ref in refs if ref.object is not None}

    orig_objects = by_identity(orig)
    recomp_objects = by_identity(recomp)
    shared = orig_objects.keys() & recomp_objects.keys()
    for key in sorted(shared, key=lambda k: (k.orig_addr, k.offset)):
        left, right = orig_objects[key], recomp_objects[key]
        if contents_comparable(left) and contents_comparable(right) and left != right:
            findings.append(
                DataFinding(DataFindingKind.OBJECT_CONTENTS, key, (left,), (right,))
            )

    def rest(refs: tuple[DataReference, ...]) -> Counter[Contents]:
        return Counter(
            ref.contents
            for ref in refs
            if ref.object is None and contents_comparable(ref.contents)
        )

    orig_rest = rest(orig)
    recomp_rest = rest(recomp)
    orig_only = list((orig_rest - recomp_rest).elements())
    recomp_only = list((recomp_rest - orig_rest).elements())
    for left in tuple(orig_only):
        for i, right in enumerate(recomp_only):
            if consistent(left, right):
                orig_only.remove(left)
                del recomp_only[i]
                break
    # A contents difference needs contents on both sides. A reference with
    # nothing comparable on the other side is either missing there, which the
    # code shows, or of unknown contents, which says nothing.
    if orig_only and recomp_only:
        findings.append(
            DataFinding(
                DataFindingKind.REFERENCED_CONTENTS,
                None,
                tuple(orig_only),
                tuple(recomp_only),
            )
        )
    return tuple(findings)


def _string_encodings(text: str) -> tuple[bytes, ...]:
    """The terminated byte forms a string can take in the binary."""
    return (
        text.encode("latin1", errors="replace") + b"\0",
        text.encode("utf-16-le") + b"\0\0",
    )


def _zero_filled(contents: Contents) -> bool:
    """No initial value: its length is where the region's extent ended,
    not something the program states."""
    match contents:
        case Uninitialized():
            return True
        case RawBytes(data, relocated=False):
            return not any(data)
    return False


def consistent(left: Contents, right: Contents) -> bool:
    """Whether two renderings can describe the same contents.

    The same bytes may reach the comparison as a string on one side and as
    raw bytes on the other, or as raw bytes of different lengths, when only
    Ghidra's data typing gave them an extent. Consistent renderings are not
    a difference; everything else is."""
    if _zero_filled(left) and _zero_filled(right):
        return True
    match left, right:
        case (StringValue(a), StringValue(b)) if is_source_path(a) and is_source_path(
            b
        ):
            return True
        case (StringValue(text), RawBytes(data, extent_known=known)) | (
            RawBytes(data, extent_known=known),
            StringValue(text),
        ):
            return any(
                data.startswith(encoded) or (not known and encoded.startswith(data))
                for encoded in _string_encodings(text)
            )
        case (RawBytes(a, extent_known=False), RawBytes(b)) | (
            RawBytes(a),
            RawBytes(b, extent_known=False),
        ):
            return a.startswith(b) or b.startswith(a)
    return left == right


@dataclass(frozen=True)
class InlineCode:
    orig: list[str] | None
    recomp: list[str] | None
    callees: tuple[int, ...]
    failures: tuple[AnalysisFailure, ...] = ()


@dataclass(frozen=True)
class InlineDiff:
    diff: tuple[str, ...] | None
    callees: tuple[int, ...]


@dataclass(frozen=True)
class FunctionResult:
    entry: FunctionEntry
    outcome: Outcome
    normal_diff: tuple[str, ...] = ()
    data_findings: tuple[DataFinding, ...] = ()
    failures: tuple[AnalysisFailure, ...] = ()
    unidentified_references: int = 0
    inline: InlineDiff | None = None
    signature_diff: tuple[str, ...] = ()
    code_change_kind: str | None = None

    @property
    def code_diff(self) -> tuple[str, ...]:
        return (self.inline.diff or ()) if self.inline is not None else self.normal_diff

    @property
    def inline_normalized_diff(self) -> tuple[str, ...] | None:
        return self.inline.diff if self.inline is not None else None

    @property
    def inline_callees(self) -> tuple[int, ...]:
        return self.inline.callees if self.inline is not None else ()


def code_diff(orig: list[str], recomp: list[str], name: str) -> tuple[str, ...]:
    return tuple(
        difflib.unified_diff(
            orig, recomp, fromfile=f"orig/{name}", tofile=f"recomp/{name}"
        )
    )


def classify(
    entry: FunctionEntry,
    *,
    failures: tuple[AnalysisFailure, ...],
    orig_code: list[str] | None,
    recomp_code: list[str] | None,
    orig_refs: tuple[DataReference, ...],
    recomp_refs: tuple[DataReference, ...],
) -> FunctionResult:
    """The outcome for one requested function from its analyzed output.

    ``*_code`` are normalized decompilation lines, or None when the side was
    not decompiled."""
    if entry.recomp_addr is None:
        return FunctionResult(entry, Outcome.UNPAIRED)

    if not failures:
        failures = tuple(
            AnalysisFailure(FailureKind.NOT_DECOMPILED, image)
            for image, code in (
                (ImageId.ORIG, orig_code),
                (ImageId.RECOMP, recomp_code),
            )
            if code is None
        )
    if failures:
        return FunctionResult(entry, Outcome.ANALYSIS_FAILED, failures=failures)
    assert orig_code is not None and recomp_code is not None

    orig_signature, orig_body = decompilation_parts(orig_code)
    recomp_signature, recomp_body = decompilation_parts(recomp_code)
    diff = code_diff(orig_body, recomp_body, entry.name)
    findings = compare_references(orig_refs, recomp_refs)
    unidentified = sum(1 for ref in (*orig_refs, *recomp_refs) if ref.object is None)
    return FunctionResult(
        entry,
        Outcome.DIFFERENCES if diff or findings else Outcome.NO_DIFFERENCES,
        normal_diff=diff,
        signature_diff=code_diff(orig_signature, recomp_signature, entry.name),
        code_change_kind=body_change_kind(orig_body, recomp_body),
        data_findings=findings,
        unidentified_references=unidentified,
    )


def classify_inline(
    normal: FunctionResult,
    code: InlineCode,
    orig_refs: tuple[DataReference, ...],
    recomp_refs: tuple[DataReference, ...],
) -> FunctionResult:
    """Use the retry without score selection, retaining the ordinary diff.

    The retry removes inlining differences; it cannot qualify an ordinary
    result that already has none."""
    if normal.outcome == Outcome.NO_DIFFERENCES:
        return normal
    retry = classify(
        normal.entry,
        failures=code.failures,
        orig_code=code.orig,
        recomp_code=code.recomp,
        orig_refs=orig_refs,
        recomp_refs=recomp_refs,
    )
    return replace(
        retry,
        normal_diff=normal.normal_diff,
        signature_diff=normal.signature_diff,
        inline=InlineDiff(
            retry.normal_diff if not retry.failures else None, code.callees
        ),
    )
