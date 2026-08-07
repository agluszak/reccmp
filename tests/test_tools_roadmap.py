"""Tests for roadmap entity-type presentation."""

from reccmp.tools.roadmap import match_type_abbreviation
from reccmp.types import EntityType


def test_import_thunk_has_distinct_abbreviation():
    assert match_type_abbreviation(EntityType.IMPORT) == "imp"
    assert match_type_abbreviation(EntityType.IMPORT_THUNK) == "ith"
