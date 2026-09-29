"""Call observations use native body membership and paired address identities."""

from pathlib import Path
from types import SimpleNamespace as NS

from reccmp.compare.call_census import call_delta, direct_call_census
from reccmp.compare.db import PairBasis
from reccmp.compare.manifest import BinaryInput, FunctionEntry, Manifest, NamedObject
from reccmp.types import EntityType, ImageId


def test_call_census_excludes_indirect_calls_and_uses_native_body():
    binary = BinaryInput(Path("unused"), "a")
    manifest = Manifest(
        "T",
        binary,
        binary,
        (FunctionEntry(0x1000, 0x2000, "caller", PairBasis.ANNOTATION, None, False),),
        (NamedObject(0x1100, 0x2100, "helper", EntityType.FUNCTION, 1, 1, PairBasis.ANNOTATION),),
        (),
    )
    body = NS(getAddressRanges=lambda: [])

    def program(site, target):
        def instruction(data):
            return NS(
                getFlowType=lambda: NS(isCall=lambda: True),
                getAddress=lambda: NS(getOffset=lambda: site),
                getBytes=lambda: data,
            )

        def instructions(actual_body, forward):
            assert actual_body is body and forward
            # The native listing contains exactly the body, excluding adjacent
            # initializer calls that a catalog maximum extent would include.
            return [
                instruction(b"\xe8" + (target - site - 5).to_bytes(4, "little", signed=True)),
                instruction(b"\xff\xd0"),
            ]

        return NS(
            getAddressFactory=lambda: NS(getDefaultAddressSpace=lambda: NS(getAddress=lambda a: a)),
            getFunctionManager=lambda: NS(getFunctionAt=lambda a: NS(getBody=lambda: body)),
            getListing=lambda: NS(getInstructions=instructions),
        )

    census = direct_call_census(
        manifest, {ImageId.ORIG: program(0x1000, 0x1100), ImageId.RECOMP: program(0x2000, 0x2100)}
    )
    row = census["functions"][0]
    delta = call_delta(row["orig"]["calls"], row["recomp"]["calls"])
    assert delta == {"category": "identical-canonical-sequence", "deltas": [], "unresolved": False}
    assert len(row["orig"]["calls"]) == 1


def test_call_delta_does_not_equate_unresolved_target_names():
    original = [{"identity": "orig:0x1100", "paired": False, "name": "Same"}]
    rebuilt = [{"identity": "recomp:0x2100", "paired": False, "name": "Same"}]
    delta = call_delta(original, rebuilt)
    assert delta["category"] == "unresolved-target"
    assert delta["deltas"] == [{"retail": ["orig:0x1100"], "rebuild": ["recomp:0x2100"]}]
