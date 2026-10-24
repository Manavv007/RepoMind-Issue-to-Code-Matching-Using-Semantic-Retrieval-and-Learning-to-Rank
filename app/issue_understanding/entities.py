"""Extract technical entities that a bug report names explicitly.

Bug reports often name the code they are about: identifiers in backticks, calls
like ``foo()``, traceback frames (``line 12, in foo``), file paths, and dotted
module paths. These are strong, cheap localization signals. Extraction is purely
lexical and works on the raw issue text without altering it.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

_IDENTIFIER_PATTERNS = (
    re.compile(r"`([A-Za-z_][\w.]*)`"),  # `name` or `pkg.mod.name`
    re.compile(r"\b([A-Za-z_]\w*)\s*\("),  # call: name(
    re.compile(r"line \d+, in ([A-Za-z_]\w*)"),  # traceback frame
    re.compile(r"\b([a-z][a-z0-9]*_[a-z0-9_]+)\b"),  # snake_case
    re.compile(r"\b([a-z]+[A-Z][A-Za-z0-9]*)\b"),  # camelCase
    re.compile(r"\b([A-Z][a-z0-9]+[A-Z][A-Za-z0-9]*)\b"),  # PascalCase
)
_PATH_RE = re.compile(r"([\w./\\-]+\.py)\b")
_MODULE_RE = re.compile(r"\b([a-z_][a-z0-9_]*(?:\.[a-z_][a-z0-9_]*){2,})\b")
# Builtins too common in reports to point at a specific function.
_COMMON_NAMES = frozenset(
    "self print len str int list dict set type range open isinstance super".split()
)


@dataclass(frozen=True)
class IssueEntities:
    """Names, file paths, and dotted modules mentioned in an issue."""

    names: frozenset[str] = field(default_factory=frozenset)
    paths: frozenset[str] = field(default_factory=frozenset)
    modules: frozenset[str] = field(default_factory=frozenset)


def extract_entities(text: str) -> IssueEntities:
    names: set[str] = set()
    for pattern in _IDENTIFIER_PATTERNS:
        for match in pattern.findall(text):
            names.add(match.split(".")[-1])
    names -= _COMMON_NAMES
    paths = {p.replace("\\", "/").lstrip("./") for p in _PATH_RE.findall(text)}
    return IssueEntities(
        names=frozenset(names),
        paths=frozenset(p for p in paths if p),
        modules=frozenset(_MODULE_RE.findall(text)),
    )


def mentions_file(file_path: str, entities: IssueEntities) -> bool:
    """True if the issue names ``file_path`` by path suffix or dotted module."""
    basename = file_path.rsplit("/", 1)[-1]
    for path in entities.paths:
        if "/" in path or path == basename:
            if file_path.endswith(path) or path.endswith(file_path):
                return True
    dotted = (
        file_path[:-3].replace("/", ".") if file_path.endswith(".py") else file_path
    )
    return any(dotted.endswith(m) or m.startswith(dotted) for m in entities.modules)
