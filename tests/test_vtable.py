from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from reccmp.tools import vtable


@pytest.mark.parametrize(
    "class_filter, names, expected",
    [(None, [], 1), ("Missing", ["Widget"], 1), ("Widget", ["Widget"], 0)],
)
def test_vtable_command_requires_a_selected_table(
    monkeypatch, capsys, class_filter, names, expected
):
    args = SimpleNamespace(filter=class_filter, no_color=True)
    catalog = Mock()
    catalog.get_vtables.return_value = [SimpleNamespace(name=name) for name in names]
    monkeypatch.setattr(vtable, "parse_args", lambda: args)
    monkeypatch.setattr(vtable, "argparse_parse_project_target", lambda _: object())
    monkeypatch.setattr(vtable.Compare, "from_target", lambda _: catalog)
    monkeypatch.setattr(
        vtable, "compare_vtable", lambda *_: SimpleNamespace(matches=True, slots=())
    )

    assert vtable.main() == expected
    assert ("100% match" in capsys.readouterr().out) is (expected == 0)
