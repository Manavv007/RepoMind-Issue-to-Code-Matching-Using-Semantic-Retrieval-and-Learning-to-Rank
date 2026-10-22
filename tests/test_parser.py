"""Tests for the Tree-sitter parser and symbol extractor."""

from __future__ import annotations

import pytest

from app.ingestion.errors import UnsupportedLanguageError
from app.parsing.symbol_extractor import extract_file
from app.parsing.tree_sitter_parser import parse_source

SAMPLE = '''\
"""Module doc."""
import os
from typing import (
    Any,
    Optional,
)

try:
    import ujson as json
except ImportError:
    import json


def top_level(a, b=1):
    """Add things.

    More detail.
    """
    helper(a)
    return os.path.join(a, str(b))


@decorator
@other.deco(1)
async def fetch(url: str) -> bytes:
    data = await client.get(url)
    return data


class Session(Base):
    """A session."""

    def __init__(self):
        self.refresh()

    @property
    def token(self):
        def inner():
            return 1
        return inner()

    class Meta:
        def helper(self): pass


if TYPE_CHECKING:
    def cond_defined(): ...
'''


def _by_name(units):
    return {u.qualified_name: u for u in units}


def test_extracts_expected_units_and_kinds():
    units = _by_name(extract_file("pkg/mod.py", "python", SAMPLE.encode()).units)
    assert set(units) == {
        "top_level",
        "fetch",
        "Session",
        "Session.__init__",
        "Session.token",
        "Session.Meta",
        "Session.Meta.helper",
        "cond_defined",
    }
    assert units["top_level"].kind == "function"
    assert units["fetch"].kind == "function"
    assert units["Session"].kind == "class"
    assert units["Session.__init__"].kind == "method"
    assert units["Session.Meta"].kind == "class"
    assert units["Session.Meta.helper"].kind == "method"
    assert units["cond_defined"].kind == "function"


def test_line_ranges_are_exact_and_include_decorators():
    units = _by_name(extract_file("m.py", "python", SAMPLE.encode()).units)
    assert (units["top_level"].start_line, units["top_level"].end_line) == (14, 20)
    # fetch starts at its first decorator (line 23), not at 'async def'.
    assert (units["fetch"].start_line, units["fetch"].end_line) == (23, 27)


def test_every_unit_source_matches_its_line_range():
    lines = SAMPLE.split("\n")
    for unit in extract_file("m.py", "python", SAMPLE.encode()).units:
        assert unit.source == "\n".join(lines[unit.start_line - 1 : unit.end_line])


def test_signature_docstring_decorators_async():
    units = _by_name(extract_file("m.py", "python", SAMPLE.encode()).units)
    assert units["top_level"].signature == "def top_level(a, b=1):"
    assert units["top_level"].docstring == "Add things.\n\nMore detail."
    fetch = units["fetch"]
    assert fetch.is_async
    assert fetch.signature == "async def fetch(url: str) -> bytes:"
    assert fetch.decorators == ("decorator", "other.deco(1)")
    assert units["Session"].signature == "class Session(Base):"
    assert units["Session"].docstring == "A session."
    assert units["Session.token"].decorators == ("property",)
    assert not units["top_level"].is_async


def test_parent_and_unit_id():
    units = _by_name(extract_file("pkg/mod.py", "python", SAMPLE.encode()).units)
    assert units["Session.__init__"].parent == "Session"
    assert units["Session.Meta.helper"].parent == "Session.Meta"
    assert units["top_level"].parent is None
    assert units["top_level"].unit_id == "pkg/mod.py::top_level:14"


def test_calls_are_syntactic_and_ordered():
    units = _by_name(extract_file("m.py", "python", SAMPLE.encode()).units)
    calls = units["top_level"].calls
    assert [c.name for c in calls] == ["helper", "join", "str"]
    assert calls[1].expression == "os.path.join"
    assert all(c.provenance == "syntactic" for c in calls)
    assert units["Session.__init__"].calls[0].expression == "self.refresh"
    assert units["Session"].calls == ()  # classes carry no call list


def test_nested_functions_excluded_by_default_included_on_request():
    default = _by_name(extract_file("m.py", "python", SAMPLE.encode()).units)
    assert "Session.token.inner" not in default
    # ...but their text stays inside the enclosing method.
    assert "def inner" in default["Session.token"].source

    nested = _by_name(
        extract_file(
            "m.py", "python", SAMPLE.encode(), include_nested_functions=True
        ).units
    )
    assert nested["Session.token.inner"].kind == "function"


def test_module_level_imports_including_try_blocks():
    record = extract_file("m.py", "python", SAMPLE.encode()).record
    assert record.imports == (
        "import os",
        "from typing import ( Any, Optional, )",
        "import ujson as json",
        "import json",
    )


def test_file_record_metadata():
    raw = SAMPLE.encode()
    record = extract_file("m.py", "python", raw).record
    assert record.path == "m.py"
    assert record.size_bytes == len(raw)
    assert record.num_lines == SAMPLE.count("\n")
    assert not record.has_syntax_errors
    assert len(record.sha256) == 64


def test_syntax_errors_are_flagged_but_valid_symbols_still_extracted():
    raw = b"def ok():\n    return 1\n\ndef broken(:\n"
    result = extract_file("bad.py", "python", raw)
    assert result.record.has_syntax_errors
    assert "ok" in {u.qualified_name for u in result.units}


def test_crlf_line_endings_are_normalized_in_source_but_lines_match():
    raw = b"def a():\r\n    return 1\r\n\r\ndef b():\r\n    return 2\r\n"
    units = _by_name(extract_file("crlf.py", "python", raw).units)
    assert units["a"].source == "def a():\n    return 1"
    assert (units["b"].start_line, units["b"].end_line) == (4, 5)


def test_unicode_and_bom_are_handled():
    text = 'def grüße():\n    """Привет 🌍"""\n    return "日本語"\n'
    raw = b"\xef\xbb\xbf" + text.encode("utf-8")
    units = extract_file("u.py", "python", raw).units
    assert units[0].symbol_name == "grüße"
    assert units[0].docstring == "Привет 🌍"
    assert units[0].source == text.rstrip("\n")


def test_empty_and_comment_only_files():
    assert extract_file("e.py", "python", b"").units == []
    result = extract_file("c.py", "python", b"# only a comment\n")
    assert result.units == []
    assert result.record.num_lines == 1


def test_one_line_and_lambda_assignments():
    units = _by_name(
        extract_file("m.py", "python", b"def f(): pass\nf2 = lambda: 1\n").units
    )
    assert set(units) == {"f"}
    assert units["f"].signature == "def f():"


def test_extraction_is_deterministic():
    first = extract_file("m.py", "python", SAMPLE.encode())
    second = extract_file("m.py", "python", SAMPLE.encode())
    assert first == second


def test_unsupported_language_raises():
    with pytest.raises(UnsupportedLanguageError):
        parse_source("cobol", b"")
