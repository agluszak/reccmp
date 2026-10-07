"""Comparison input keys are independent of the installation location."""

import shutil
import sys
from importlib import import_module

from reccmp.ghidriff.inputs import package_fingerprint


def test_package_fingerprint_ignores_install_location(tmp_path, monkeypatch):
    source = tmp_path / "src" / "demo_pkg"
    source.mkdir(parents=True)
    (source / "__init__.py").write_text("VALUE = 1\n")
    (source / "mod.py").write_text("X = 2\n")
    copy = tmp_path / "other"
    shutil.copytree(tmp_path / "src", copy)

    def fingerprint(root):
        monkeypatch.syspath_prepend(str(root))
        sys.modules.pop("demo_pkg", None)
        package = import_module("demo_pkg")
        try:
            return package_fingerprint(package)
        finally:
            sys.modules.pop("demo_pkg", None)
            sys.path.remove(str(root))

    first = fingerprint(tmp_path / "src")
    assert fingerprint(copy) == first
    (copy / "demo_pkg" / "mod.py").write_text("X = 3\n")
    assert fingerprint(copy) != first
