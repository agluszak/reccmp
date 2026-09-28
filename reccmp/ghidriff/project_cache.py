"""Pristine and prepared copies in reccmp's cached Ghidra project."""

# Ghidra Java packages exist only after the engine starts the JVM.
# pylint: disable=import-outside-toplevel

from pathlib import Path
from typing import Any

_PRISTINE_FOLDER = "pristine"
_PREPARED_FOLDER = "prepared"


def reset_programs(project: Any) -> None:
    """Restore analyzed programs without names or functions from a prior run."""
    from ghidra.util.task import TaskMonitor

    root = project.getRootFolder()
    pristine = root.getFolder(_PRISTINE_FOLDER) or root.createFolder(_PRISTINE_FOLDER)
    for domain_file in root.getFiles():
        saved = pristine.getFile(domain_file.getName())
        if saved is None:
            domain_file.copyTo(pristine, TaskMonitor.DUMMY)
        else:
            domain_file.delete()
            saved.copyTo(root, TaskMonitor.DUMMY)


def restore_prepared(project: Any, stamp: Path, key: str) -> bool:
    """Restore the last preparation when its complete manifest is unchanged."""
    from ghidra.util.task import TaskMonitor

    root = project.getRootFolder()
    prepared = root.getFolder(_PREPARED_FOLDER)
    files = list(root.getFiles())
    if (
        prepared is None
        or not stamp.is_file()
        or stamp.read_text(encoding="ascii") != key
        or any(prepared.getFile(file.getName()) is None for file in files)
    ):
        return False
    for file in files:
        saved = prepared.getFile(file.getName())
        file.delete()
        saved.copyTo(root, TaskMonitor.DUMMY)
    return True


def save_prepared(project: Any, stamp: Path, key: str) -> None:
    """Keep one complete prepared pair for an identical later request."""
    from ghidra.util.task import TaskMonitor

    stamp.unlink(missing_ok=True)
    root = project.getRootFolder()
    prepared = root.getFolder(_PREPARED_FOLDER) or root.createFolder(_PREPARED_FOLDER)
    for file in prepared.getFiles():
        file.delete()
    for file in root.getFiles():
        file.copyTo(prepared, TaskMonitor.DUMMY)
    stamp.write_text(key, encoding="ascii")
