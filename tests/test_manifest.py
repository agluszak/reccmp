"""The pairs and names reccmp hands to the code differ."""

from pathlib import Path
from unittest.mock import Mock

from reccmp.compare import Compare
from reccmp.compare.db import PairBasis
from reccmp.compare.manifest import NamedObject, build_manifest, select_addresses
from reccmp.cvdump import CvdumpAnalysis
from reccmp.types import EntityType, ImageId
from .raw_image import RawImage


def _catalog() -> Compare:
    catalog = Compare(
        RawImage.from_memory(b"\x00" * 0x100),
        RawImage.from_memory(b"\x00" * 0x100),
        Mock(spec=CvdumpAnalysis),
        "TEST",
    )
    with catalog.db.batch() as batch:
        batch.set(ImageId.ORIG, 0x10, type=EntityType.FUNCTION, name="Paired")
        batch.set(ImageId.RECOMP, 0x20, type=EntityType.FUNCTION, name="Paired")
        batch.match(0x10, 0x20, basis=PairBasis.ANNOTATION)
        batch.set(ImageId.ORIG, 0x30, type=EntityType.FUNCTION, name="Unpaired")
        batch.set(ImageId.ORIG, 0x40, type=EntityType.FUNCTION, name="Stub", stub=True)
        batch.set(ImageId.RECOMP, 0x40, type=EntityType.FUNCTION, name="Stub")
        batch.match(0x40, 0x40, basis=PairBasis.ANNOTATION)
        batch.set(ImageId.RECOMP, 0x50, type=EntityType.DATA, name="g_value", size=8)
        batch.match(0x60, 0x50, basis=PairBasis.ANNOTATION)
        batch.set(ImageId.ORIG, 0x70, type=EntityType.STRING, name='"hi"', size=3)
        batch.set(ImageId.RECOMP, 0x80, type=EntityType.FUNCTION, name="RecompOnly")
    return catalog


def _manifest(tmp_path: Path, catalog: Compare, **kwargs):
    orig = tmp_path / "orig.exe"
    recomp = tmp_path / "recomp.exe"
    orig.write_bytes(b"orig")
    recomp.write_bytes(b"recomp")
    return build_manifest(
        catalog, target_id="TEST", orig_path=orig, recomp_path=recomp, **kwargs
    )


def test_every_original_function_is_requested_paired_or_not(tmp_path: Path):
    manifest = _manifest(tmp_path, _catalog())

    entries = {entry.name: entry for entry in manifest.functions}
    # Stubs are declared unimplemented; recomp-only functions have no
    # original to compare against.
    assert set(entries) == {"Paired", "Unpaired"}
    assert entries["Paired"].recomp_addr == 0x20
    assert entries["Paired"].basis == PairBasis.ANNOTATION
    assert entries["Unpaired"].recomp_addr is None
    assert entries["Unpaired"].basis is None


def test_paired_objects_share_one_extent(tmp_path: Path):
    manifest = _manifest(tmp_path, _catalog())

    [value] = [obj for obj in manifest.objects if obj.name == "g_value"]
    assert value.orig_size is None and value.recomp_size == 8
    # The recompiled PDB's size stands for the original too.
    assert value.extent(ImageId.ORIG) == 8
    assert value.addr(ImageId.ORIG) == 0x60
    assert value.addr(ImageId.RECOMP) == 0x50


def test_unpaired_entities_keep_their_own_image(tmp_path: Path):
    manifest = _manifest(tmp_path, _catalog())

    unpaired = {(entity.image_id, entity.addr): entity for entity in manifest.unpaired}
    assert unpaired[(ImageId.ORIG, 0x70)].entity_type == EntityType.STRING
    assert unpaired[(ImageId.ORIG, 0x70)].size == 3
    assert unpaired[(ImageId.ORIG, 0x30)].name == "Unpaired"
    assert unpaired[(ImageId.RECOMP, 0x80)].name == "RecompOnly"


def test_selection_limits_requested_functions_not_names(tmp_path: Path):
    manifest = _manifest(tmp_path, _catalog(), select=select_addresses([0x10]))

    assert [entry.name for entry in manifest.functions] == ["Paired"]
    # Every pair still gives both programs the same vocabulary.
    assert {obj.name for obj in manifest.objects} >= {"Paired", "g_value"}


def test_manifest_records_its_inputs(tmp_path: Path):
    manifest = _manifest(tmp_path, _catalog())
    document = manifest.to_json()

    assert document["orig"]["sha256"] == manifest.orig.sha256
    assert document["functions"][0]["basis"] == "annotation"
    assert manifest.digest() == _manifest(tmp_path, _catalog()).digest()


def test_extent_prefers_the_own_image():
    obj = NamedObject(
        orig_addr=1,
        recomp_addr=2,
        name="x",
        entity_type=EntityType.DATA,
        orig_size=4,
        recomp_size=8,
        basis=PairBasis.ANNOTATION,
    )
    assert obj.extent(ImageId.ORIG) == 4
    assert obj.extent(ImageId.RECOMP) == 8
