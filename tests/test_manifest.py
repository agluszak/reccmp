"""The pairs and names reccmp hands to the code differ."""

import json
from dataclasses import replace
from pathlib import Path, PurePath
from unittest.mock import Mock

import pytest

from reccmp.compare import Compare
from reccmp.compare.db import PairBasis
from reccmp.compare.manifest import (
    Alias,
    Manifest,
    NamedObject,
    SourceLocation,
    build_manifest,
    select_addresses,
)
from reccmp.cvdump import CvdumpAnalysis
from reccmp.ghidriff.locations import Extents
from reccmp.types import EntityType, ImageId
from reccmp.tools import compare
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
        batch.set(
            ImageId.RECOMP,
            0x20,
            type=EntityType.FUNCTION,
            name="Paired",
            symbol="?Paired@@YAXXZ",
        )
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
    assert document["objects"][0]["recomp_symbol"] == "?Paired@@YAXXZ"
    assert manifest.digest() == _manifest(tmp_path, _catalog()).digest()


def test_prepared_identity_tracks_extent_but_not_source_location(tmp_path: Path):
    manifest = _manifest(tmp_path, _catalog())
    relocated = replace(
        manifest,
        functions=(
            replace(
                manifest.functions[0],
                source=SourceLocation(PurePath("file.cpp"), 42),
            ),
            *manifest.functions[1:],
        ),
    )
    assert relocated.digest() != manifest.digest()
    assert relocated.preparation_digest() == manifest.preparation_digest()

    object_with_new_extent = replace(manifest.objects[0], recomp_size=16)
    resized = replace(manifest, objects=(object_with_new_extent, *manifest.objects[1:]))
    assert resized.digest() != manifest.digest()
    assert resized.preparation_digest() != manifest.preparation_digest()


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


def test_unpaired_import_slot_is_not_compared_as_literal_data(tmp_path: Path):
    catalog = _catalog()
    with catalog.db.batch() as batch:
        batch.set(
            ImageId.ORIG,
            0x90,
            type=EntityType.IMPORT,
            name="SR.dll::RetailOnlyExport",
            size=4,
        )
        batch.set(
            ImageId.ORIG, 0x94, type=EntityType.DATA, name="adjacent_literal", size=4
        )
    manifest = _manifest(tmp_path, catalog)
    extents = Extents(manifest, ImageId.ORIG)
    assert not extents.is_data(0x90)
    assert extents.is_data(0x94)


def test_saved_manifest_roundtrip_preserves_pairing_and_alias_extent(tmp_path: Path):
    manifest = _manifest(tmp_path, _catalog())
    manifest = replace(manifest, aliases=(Alias(ImageId.RECOMP, 0x123, 0x10, 7),))
    restored = Manifest.from_json(manifest.to_json())
    assert restored == manifest
    assert restored.digest() == manifest.digest()
    assert restored.aliases[0].size == 7
    restored.validate_binaries()
    restored.recomp.path.write_bytes(b"replaced product")
    with pytest.raises(ValueError, match="Comparison binary changed"):
        restored.validate_binaries()


def test_manifest_cli_replays_saved_inputs_without_current_catalog(
    tmp_path: Path, monkeypatch
):
    manifest = _manifest(tmp_path, _catalog())
    path = tmp_path / "manifest.json"
    path.write_text(json.dumps(manifest.to_json()))
    calls = []
    monkeypatch.setattr(
        "sys.argv",
        ["compare", "--manifest", str(path), "--output", str(tmp_path / "report")],
    )
    monkeypatch.setattr(
        compare, "_run_engine", lambda _args, saved: calls.append(saved)
    )
    monkeypatch.setattr(
        compare,
        "argparse_parse_project_target",
        lambda _: pytest.fail("Replay must not resolve a current project"),
    )
    assert compare.main() == 0
    assert calls == [manifest]
    manifest.recomp.path.write_bytes(b"changed")
    with pytest.raises(ValueError, match="Comparison binary changed"):
        compare.main()
    assert calls == [manifest]
