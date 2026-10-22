"""Symbol extraction: turn a parsed Python file into code units and metadata.

Extracts functions, methods, and classes with exact line ranges, signatures,
docstrings, decorators, syntactic call references, and module-level imports.

Notes:
* Line numbers are 1-based and inclusive; a decorated definition starts at its
  first decorator. ``CodeUnit.source`` is exactly the text of those whole lines
  (indentation and trailing comments included, LF line endings).
* Nested functions (functions defined inside functions) are *not* emitted as
  separate units by default; their text remains part of the enclosing unit.
  Methods of nested classes are emitted with dotted qualified names.
* Class units contain the full class text (including method bodies), so
  downstream consumers that index only leaf code should filter on ``kind``.
* Calls are syntactic only and are not resolved to definitions.
"""

from __future__ import annotations

import hashlib
import inspect
import re
from dataclasses import dataclass

from tree_sitter import Node

from app.parsing.tree_sitter_parser import parse_source
from app.schemas.code_unit import CallRef, CodeUnit, FileRecord, UnitKind

_BOM = b"\xef\xbb\xbf"
_WS_RE = re.compile(r"\s+")
_IDENT_RE = re.compile(r"[A-Za-z_]\w*")
_MAX_EXPRESSION_CHARS = 200

# Statement/clause nodes that may contain definitions (e.g. ``if TYPE_CHECKING:``).
_CONTAINER_TYPES: frozenset[str] = frozenset(
    {
        "block",
        "if_statement",
        "elif_clause",
        "else_clause",
        "try_statement",
        "except_clause",
        "except_group_clause",
        "finally_clause",
        "with_statement",
        "for_statement",
        "while_statement",
        "match_statement",
        "case_clause",
    }
)
_IMPORT_TYPES: frozenset[str] = frozenset({"import_statement", "import_from_statement"})


@dataclass(frozen=True)
class ExtractedFile:
    """Result of extracting one source file."""

    record: FileRecord
    units: list[CodeUnit]


def extract_file(
    rel_path: str,
    language: str,
    raw: bytes,
    *,
    include_nested_functions: bool = False,
) -> ExtractedFile:
    """Parse ``raw`` bytes and extract file metadata plus code units."""
    data = raw[len(_BOM) :] if raw.startswith(_BOM) else raw
    tree = parse_source(language, data)
    root = tree.root_node
    extractor = _PythonExtractor(rel_path, language, data, include_nested_functions)
    units = extractor.extract_units(root)
    imports = extractor.extract_imports(root)

    num_lines = data.count(b"\n") + (1 if data and not data.endswith(b"\n") else 0)
    record = FileRecord(
        path=rel_path,
        language=language,
        sha256=hashlib.sha256(raw).hexdigest(),
        size_bytes=len(raw),
        num_lines=num_lines,
        has_syntax_errors=root.has_error,
        imports=tuple(imports),
    )
    return ExtractedFile(record=record, units=units)


class _PythonExtractor:
    def __init__(
        self,
        rel_path: str,
        language: str,
        data: bytes,
        include_nested: bool,
    ) -> None:
        self._path = rel_path
        self._language = language
        self._data = data
        self._include_nested = include_nested

    # ------------------------------------------------------------- helpers
    def _text(self, node: Node) -> str:
        return self._data[node.start_byte : node.end_byte].decode("utf-8", "replace")

    @staticmethod
    def _collapse(text: str) -> str:
        return _WS_RE.sub(" ", text).strip()

    # --------------------------------------------------------------- units
    def extract_units(self, root: Node) -> list[CodeUnit]:
        found: list[tuple[int, str, CodeUnit]] = []
        # Stack items: (node whose children are scanned, scope names, enclosing kind)
        stack: list[tuple[Node, tuple[str, ...], str | None]] = [(root, (), None)]
        while stack:
            node, scope, enclosing = stack.pop()
            for child in node.children:
                if not child.is_named:
                    continue
                definition = child
                decorators: tuple[str, ...] = ()
                if child.type == "decorated_definition":
                    inner = child.child_by_field_name("definition")
                    if inner is None:
                        continue
                    definition = inner
                    decorators = tuple(
                        self._collapse(self._text(d)).lstrip("@").strip()
                        for d in child.children
                        if d.type == "decorator"
                    )

                if definition.type == "function_definition":
                    if enclosing == "function" and not self._include_nested:
                        continue
                    unit = self._build_unit(
                        child, definition, decorators, scope, enclosing
                    )
                    if unit is None:
                        continue
                    found.append((child.start_byte, unit.qualified_name, unit))
                    body = definition.child_by_field_name("body")
                    if self._include_nested and body is not None:
                        stack.append((body, (*scope, unit.symbol_name), "function"))
                elif definition.type == "class_definition":
                    if enclosing == "function" and not self._include_nested:
                        continue
                    unit = self._build_unit(
                        child, definition, decorators, scope, enclosing
                    )
                    if unit is None:
                        continue
                    found.append((child.start_byte, unit.qualified_name, unit))
                    body = definition.child_by_field_name("body")
                    if body is not None:
                        stack.append((body, (*scope, unit.symbol_name), "class"))
                elif child.type in _CONTAINER_TYPES:
                    stack.append((child, scope, enclosing))

        found.sort(key=lambda item: (item[0], item[1]))
        return [unit for _, _, unit in found]

    def _build_unit(
        self,
        outer: Node,
        definition: Node,
        decorators: tuple[str, ...],
        scope: tuple[str, ...],
        enclosing: str | None,
    ) -> CodeUnit | None:
        name_node = definition.child_by_field_name("name")
        if name_node is None:
            return None
        name = self._text(name_node)
        if not name:
            return None

        is_class = definition.type == "class_definition"
        kind: UnitKind
        if is_class:
            kind = "class"
        elif enclosing == "class":
            kind = "method"
        else:
            kind = "function"

        qualified = ".".join((*scope, name))
        parent = ".".join(scope) if enclosing == "class" and scope else None

        start_line = outer.start_point.row + 1
        end_row = outer.end_point.row
        if outer.end_point.column == 0 and end_row > outer.start_point.row:
            end_row -= 1  # node ends exactly at a line start; last line is the previous
        end_line = max(end_row + 1, start_line)

        source = self._line_aligned_source(outer)
        body = definition.child_by_field_name("body")
        calls: tuple[CallRef, ...] = ()
        if not is_class and body is not None:
            calls = self._extract_calls(body)

        return CodeUnit(
            unit_id=f"{self._path}::{qualified}:{start_line}",
            language=self._language,
            file_path=self._path,
            kind=kind,
            symbol_name=name,
            qualified_name=qualified,
            parent=parent,
            start_line=start_line,
            end_line=end_line,
            signature=self._signature(definition, body),
            docstring=self._docstring(body),
            source=source,
            content_hash=hashlib.sha256(source.encode("utf-8")).hexdigest(),
            is_async=any(c.type == "async" for c in definition.children),
            decorators=decorators,
            calls=calls,
        )

    def _line_aligned_source(self, node: Node) -> str:
        """Return the node's text expanded to whole lines (LF-normalized).

        The result is exactly the text of lines ``start_line..end_line`` of the
        file, so indentation and trailing comments are preserved and a unit can
        be verified against the indexed commit line-for-line.
        """
        start = self._data.rfind(b"\n", 0, node.start_byte) + 1
        end = self._data.find(b"\n", node.end_byte)
        if end == -1:
            end = len(self._data)
        text = self._data[start:end].decode("utf-8", "replace")
        return text.replace("\r\n", "\n").rstrip("\r")

    def _signature(self, definition: Node, body: Node | None) -> str:
        end = body.start_byte if body is not None else definition.end_byte
        header = self._data[definition.start_byte : end].decode("utf-8", "replace")
        header = self._collapse(header)
        if body is None:
            header = header.split(":")[0] + ":"
        return header

    def _docstring(self, body: Node | None) -> str | None:
        if body is None or body.type != "block":
            return None
        first = next((c for c in body.children if c.is_named), None)
        if first is None or first.type != "expression_statement":
            return None
        string = first.children[0] if first.children else None
        if string is None or string.type != "string":
            return None
        content = "".join(
            self._text(c) for c in string.children if c.type == "string_content"
        )
        doc = inspect.cleandoc(content)
        return doc or None

    def _extract_calls(self, body: Node) -> tuple[CallRef, ...]:
        calls: list[tuple[int, CallRef]] = []
        stack: list[Node] = [body]
        while stack:
            node = stack.pop()
            if node.type == "call":
                callee = node.child_by_field_name("function")
                if callee is not None:
                    ref = self._call_ref(callee, node.start_point.row + 1)
                    if ref is not None:
                        calls.append((node.start_byte, ref))
            stack.extend(node.children)
        calls.sort(key=lambda item: item[0])
        return tuple(ref for _, ref in calls)

    def _call_ref(self, callee: Node, line: int) -> CallRef | None:
        expression = self._collapse(self._text(callee))[:_MAX_EXPRESSION_CHARS]
        if not expression:
            return None
        if callee.type == "identifier":
            name = expression
        elif callee.type == "attribute":
            attr = callee.child_by_field_name("attribute")
            name = self._text(attr) if attr is not None else expression
        else:
            idents = _IDENT_RE.findall(expression)
            if not idents:
                return None
            name = idents[-1]
        return CallRef(name=name, expression=expression, line=line)

    # ------------------------------------------------------------- imports
    def extract_imports(self, root: Node) -> list[str]:
        found: list[tuple[int, str]] = []
        stack: list[Node] = [root]
        while stack:
            node = stack.pop()
            for child in node.children:
                if child.type in _IMPORT_TYPES:
                    found.append((child.start_byte, self._collapse(self._text(child))))
                elif child.type in _CONTAINER_TYPES:
                    stack.append(child)
        found.sort(key=lambda item: item[0])
        return [text for _, text in found]
