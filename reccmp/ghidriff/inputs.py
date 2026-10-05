"""Comparison inputs shared by cache identity and report provenance."""

from dataclasses import dataclass, asdict
import hashlib
import json
from pathlib import Path

import ghidriff
import reccmp
from reccmp.analysis_cache import fingerprint_files
from reccmp.compare.manifest import Manifest


@dataclass(frozen=True)
class RunInputs:  # pylint: disable=too-many-instance-attributes
    manifest_sha256: str
    selection_sha256: str
    reccmp_version: str
    ghidra_version: str
    decompiler_sha256: str
    ghidriff_version: str
    analysis_key: str
    preparation_key: str
    normalization_key: str
    comparison_key: str
    reviewed_sha256: str
    decompiler_timeout: int
    threaded: bool
    max_ram_percent: float
    ghidra_project: str

    @classmethod
    def create(
        cls, manifest: Manifest, engine, args, reviewed_digest: str
    ) -> "RunInputs":
        # pylint: disable=import-outside-toplevel
        from .engine import ANALYSIS_REVISION, PREPARATION_REVISION

        ghidra_version = str(engine.get_ghidra_version())
        decompiler_sha256 = native_decompiler_digest(Path(engine.launcher.install_dir))
        analysis_key = (
            f"{manifest.target_id}-{manifest.orig.sha256[:12]}"
            f"-ghidra{ghidra_version}-native{decompiler_sha256[:12]}"
            f"-ghidriff{ghidriff.__version__}"
            f"-reccmp{ANALYSIS_REVISION}"
            f"{'-switchfocus1' if engine.focused_switch_analysis else ''}"
        )
        preparation_key = (
            f"v{PREPARATION_REVISION}:{manifest.preparation_digest()}:{reviewed_digest}"
        )
        normalization_key = fingerprint_files(
            [
                *Path(reccmp.__file__).parent.rglob("*.py"),
                *Path(ghidriff.__file__).parent.rglob("*.py"),
            ]
        )
        selection_sha256 = hashlib.sha256(
            json.dumps(sorted(entry.orig_addr for entry in manifest.functions)).encode()
        ).hexdigest()
        comparison_key = hashlib.sha256(
            json.dumps(
                [
                    manifest.digest(),
                    preparation_key,
                    analysis_key,
                    normalization_key,
                    decompiler_sha256,
                    args.decompiler_timeout,
                    args.threaded,
                    args.max_ram_percent,
                    reccmp.VERSION,
                    ghidriff.__version__,
                ]
            ).encode()
        ).hexdigest()
        projects = (
            args.ghidra_projects or manifest.recomp.path.parent / ".reccmp-cache/ghidra"
        )
        return cls(
            manifest.digest(),
            selection_sha256,
            reccmp.VERSION,
            ghidra_version,
            decompiler_sha256,
            ghidriff.__version__,
            analysis_key,
            preparation_key,
            normalization_key,
            comparison_key,
            reviewed_digest,
            args.decompiler_timeout,
            args.threaded,
            args.max_ram_percent,
            str(projects / analysis_key),
        )

    def to_json(self) -> dict:
        return asdict(self)


def native_decompiler_digest(install_dir: Path) -> str:
    """Native patches can change output without changing the Ghidra version."""
    # pylint: disable=import-outside-toplevel,import-error
    from ghidra.framework import Platform

    platform = Platform.CURRENT_PLATFORM
    path = (
        install_dir
        / "Ghidra"
        / "Features"
        / "Decompiler"
        / "os"
        / str(platform.getDirectoryName())
    )
    path /= "decompile" + str(platform.getExecutableExtension())
    return hashlib.sha256(path.read_bytes()).hexdigest()
