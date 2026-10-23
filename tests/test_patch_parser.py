"""Tests for the unified-diff parser."""

from __future__ import annotations

import pytest

from evaluation.patch_parser import PatchParseError, parse_unified_diff

MODIFY = """\
diff --git a/pkg/a.py b/pkg/a.py
index 111..222 100644
--- a/pkg/a.py
+++ b/pkg/a.py
@@ -3,4 +3,4 @@ def f():
 line3
-line4
+line4new
 line5
 line6
"""


def test_modification_records_deleted_line_and_insertion_point():
    (fp,) = parse_unified_diff(MODIFY)
    assert (fp.old_path, fp.new_path) == ("pkg/a.py", "pkg/a.py")
    assert fp.deleted_old_lines == [4]
    assert [(i.after_old_line, i.lines) for i in fp.insertions] == [(4, ("line4new",))]
    assert fp.added_lines == ["line4new"]
    assert fp.hunk_count == 1
    assert not fp.is_new and not fp.is_deleted


def test_pure_insertion_uses_line_before_edit():
    patch = "--- a/a.py\n+++ b/a.py\n@@ -5,0 +6,2 @@\n+x\n+y\n"
    (fp,) = parse_unified_diff(patch)
    assert fp.deleted_old_lines == []
    assert [(i.after_old_line, i.lines) for i in fp.insertions] == [(5, ("x", "y"))]


def test_insertion_between_context_lines():
    patch = "--- a/a.py\n+++ b/a.py\n@@ -2,2 +2,3 @@\n two\n+inserted\n three\n"
    (fp,) = parse_unified_diff(patch)
    assert [(i.after_old_line, i.lines) for i in fp.insertions] == [(2, ("inserted",))]


def test_new_and_deleted_files():
    patch = (
        "--- /dev/null\n+++ b/new.py\n@@ -0,0 +1,2 @@\n+a\n+b\n"
        "--- a/gone.py\n+++ /dev/null\n@@ -1,2 +0,0 @@\n-x\n-y\n"
    )
    new, gone = parse_unified_diff(patch)
    assert new.is_new and new.path == "new.py" and new.added_lines == ["a", "b"]
    assert gone.is_deleted and gone.path == "gone.py"
    assert gone.deleted_old_lines == [1, 2]


def test_multiple_files_and_hunks():
    patch = (
        "--- a/a.py\n+++ b/a.py\n@@ -1,2 +1,2 @@\n-a\n+A\n b\n"
        "@@ -10,2 +10,2 @@\n c\n-d\n+D\n"
        "--- a/b.py\n+++ b/b.py\n@@ -1 +1 @@\n-x\n+y\n"
    )
    a, b = parse_unified_diff(patch)
    assert a.hunk_count == 2 and a.deleted_old_lines == [1, 11]
    assert b.path == "b.py" and b.deleted_old_lines == [1]


def test_no_newline_marker_is_ignored():
    patch = (
        "--- a/a.py\n+++ b/a.py\n@@ -1 +1 @@\n-old\n"
        "\\ No newline at end of file\n+new\n\\ No newline at end of file\n"
    )
    (fp,) = parse_unified_diff(patch)
    assert fp.deleted_old_lines == [1] and fp.added_lines == ["new"]


def test_removed_line_that_looks_like_a_file_header():
    # Removing the SQL comment "-- foo" yields the diff line "--- foo".
    patch = "--- a/a.py\n+++ b/a.py\n@@ -1,2 +1,1 @@\n--- foo\n keep\n"
    (fp,) = parse_unified_diff(patch)
    assert fp.deleted_old_lines == [1]
    assert len(parse_unified_diff(patch)) == 1


def test_empty_context_line_without_leading_space():
    patch = "--- a/a.py\n+++ b/a.py\n@@ -1,3 +1,3 @@\n a\n\n-c\n+C\n"
    (fp,) = parse_unified_diff(patch)
    assert fp.deleted_old_lines == [3]


def test_crlf_patch_text():
    (fp,) = parse_unified_diff(MODIFY.replace("\n", "\r\n"))
    assert fp.deleted_old_lines == [4]
    assert fp.added_lines == ["line4new"]


def test_header_without_counts_defaults_to_one():
    (fp,) = parse_unified_diff("--- a/a.py\n+++ b/a.py\n@@ -7 +7 @@\n-a\n+b\n")
    assert fp.deleted_old_lines == [7]


def test_truncated_hunk_raises():
    with pytest.raises(PatchParseError):
        parse_unified_diff("--- a/a.py\n+++ b/a.py\n@@ -1,5 +1,5 @@\n a\n-b\n+c\n")


def test_hunk_without_file_header_raises():
    with pytest.raises(PatchParseError):
        parse_unified_diff("@@ -1 +1 @@\n-a\n+b\n")


def test_garbage_inside_hunk_raises():
    with pytest.raises(PatchParseError):
        parse_unified_diff("--- a/a.py\n+++ b/a.py\n@@ -1,2 +1,2 @@\n a\n!bad\n")


def test_empty_patch_has_no_files():
    assert parse_unified_diff("") == []
