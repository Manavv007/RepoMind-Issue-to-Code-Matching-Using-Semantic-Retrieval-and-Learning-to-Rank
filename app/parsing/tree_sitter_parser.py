"""Tree-sitter parser wrappers.

Parsing is purely syntactic: source text is turned into a concrete syntax tree
and is never imported, compiled, or executed.
"""

from __future__ import annotations

from functools import cache

import tree_sitter_python
from tree_sitter import Language, Parser, Tree

from app.ingestion.errors import UnsupportedLanguageError

SUPPORTED_LANGUAGES: frozenset[str] = frozenset({"python"})


@cache
def _get_parser(language: str) -> Parser:
    if language == "python":
        return Parser(Language(tree_sitter_python.language()))
    raise UnsupportedLanguageError(
        f"Language '{language}' is not supported. "
        f"Supported: {', '.join(sorted(SUPPORTED_LANGUAGES))}."
    )


def parse_source(language: str, data: bytes) -> Tree:
    """Parse ``data`` with the Tree-sitter grammar for ``language``.

    Tree-sitter is error-tolerant: malformed code still yields a tree, and
    ``tree.root_node.has_error`` reports whether syntax errors were found.

    Raises:
        UnsupportedLanguageError: if ``language`` has no grammar configured.
    """
    return _get_parser(language).parse(data)
