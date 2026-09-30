"""Origin of stored prototypes used as independent retail ABI evidence.

Ghidra's SourceType is precedence, not provenance: older PDB importers wrote
USER_DEFINED. Untagged signatures therefore remain unknown. A retail review
must explicitly record ``retail-reviewed`` after inspecting binary evidence;
source/PDB agreement never creates that origin.
"""

import json

PROPERTY = "reccmp.signature-origin"


def _signature(function) -> str:
    return json.dumps(
        [
            str(function.getSignature()),
            str(function.getCallingConventionName()),
            bool(function.hasVarArgs()),
        ]
    )


def record_signature_origin(program, function, origin: str) -> None:
    properties = program.getUsrPropertyManager()
    stamps = properties.getStringPropertyMap(PROPERTY)
    if stamps is None:
        stamps = properties.createStringPropertyMap(PROPERTY)
    stamps.add(
        function.getEntryPoint(),
        json.dumps({"origin": origin, "signature": _signature(function)}),
    )


def _reviewed_origin(program, function) -> str | None:
    stamps = program.getUsrPropertyManager().getStringPropertyMap(PROPERTY)
    if stamps is None:
        return None
    value = stamps.getString(function.getEntryPoint())
    if value is None:
        return None
    try:
        stamp = json.loads(str(value))
    except (ValueError, TypeError):
        return None
    if not isinstance(stamp, dict):
        return None
    if stamp.get("signature") != _signature(function):
        return None
    origin = stamp.get("origin")
    return origin if isinstance(origin, str) else None


def independently_reviewed_signature(program, function) -> bool:
    return _reviewed_origin(program, function) == "retail-reviewed"


def independently_reviewed_return(program, function) -> bool:
    """A return-only binary review does not establish parameter count or types."""
    return _reviewed_origin(program, function) in {
        "retail-reviewed",
        "retail-return-reviewed",
    }
