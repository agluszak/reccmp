"""Values a build records about its environment rather than its program.

A ``__FILE__`` string records where a build found its source and ``__LINE__``
where a call sits in that file. Two builds of the same source differ in both,
so the comparison shows them as the build context they are.
"""

import re

SOURCE_FILE = "SOURCE_FILE"
SOURCE_LINE = "SOURCE_LINE"
_SOURCE_PATH = re.compile(
    r"(?:[A-Za-z]:|\\\\|/)[^\r\n\"]*\.(?:c|cc|cpp|cxx|h|hh|hpp|hxx|inl)",
    re.IGNORECASE,
)


def is_source_path(text: str) -> bool:
    """An absolute path to a C/C++ source or header file."""
    return _SOURCE_PATH.fullmatch(text) is not None
