"""Tests for patch -> ground-truth labeling against a pre-fix index."""

from __future__ import annotations

from app.parsing.symbol_extractor import extract_file
from evaluation.labeler import label_patch

SOURCE = """\
import os


def alpha(x):
    y = x + 1
    return y


class K:
    def beta(self):
        a = 1
        return a

    def gamma(self):
        return 2


def delta():
    return 3
"""
# Line map: alpha 4-6 | K 9-15 | beta 10-12 | gamma 14-15 | delta 18-19


def _label(patch: str, extra_files: set[str] | None = None, path: str = "m.py"):
    extracted = extract_file(path, "python", SOURCE.encode())
    return label_patch(
        patch,
        {path: extracted.units},
        {path} | (extra_files or set()),
    )


def _diff(body: str, path: str = "m.py") -> str:
    return f"--- a/{path}\n+++ b/{path}\n{body}"


def _names(truth) -> list[str]:
    return [u.qualified_name for u in truth.units]


def test_changed_line_labels_enclosing_function():
    truth = _label(_diff("@@ -5 +5 @@\n-    y = x + 1\n+    y = x + 2\n"))
    assert _names(truth) == ["alpha"]
    assert truth.files == ("m.py",)
    assert truth.unlocated_changes == 0
    unit = truth.units[0]
    assert (unit.start_line, unit.end_line, unit.kind) == (4, 6, "function")
    assert unit.unit_id == "m.py::alpha:4"


def test_change_inside_method_labels_method_not_class():
    truth = _label(_diff("@@ -11 +11 @@\n-        a = 1\n+        a = 2\n"))
    assert _names(truth) == ["K.beta"]
    assert truth.units[0].kind == "method"


def test_pure_insertion_inside_body():
    truth = _label(_diff("@@ -11,0 +12 @@\n+        b = 2\n"))
    assert _names(truth) == ["K.beta"]
    assert truth.unlocated_changes == 0


def test_insertion_after_last_line_extends_body_when_indented_deeper():
    deeper = _label(_diff("@@ -6,0 +7 @@\n+    cleanup()\n"))
    assert _names(deeper) == ["alpha"]
    method_tail = _label(_diff("@@ -12,0 +13 @@\n+        extra = 1\n"))
    assert _names(method_tail) == ["K.beta"]


def test_insertion_after_last_line_at_same_indent_is_a_new_sibling():
    sibling = _label(_diff("@@ -6,0 +7,2 @@\n+def new_func():\n+    pass\n"))
    assert sibling.units == ()
    assert sibling.unlocated_changes == 1
    new_method = _label(
        _diff("@@ -12,0 +13,2 @@\n+    def extra(self):\n+        pass\n")
    )
    assert new_method.units == ()


def test_module_level_edit_is_unlocated_but_file_is_relevant():
    truth = _label(_diff("@@ -1 +1 @@\n-import os\n+import os, sys\n"))
    assert truth.units == ()
    assert truth.files == ("m.py",)
    assert truth.unlocated_changes == 1
    assert truth.has_file_labels and not truth.has_function_labels


def test_class_body_edit_is_not_labeled_as_a_function():
    truth = _label(_diff("@@ -9 +9 @@\n-class K:\n+class K(object):\n"))
    assert truth.units == ()
    assert truth.unlocated_changes == 1


def test_edit_spanning_two_functions_labels_both():
    # Remove lines 12-15: tail of beta through gamma.
    truth = _label(
        _diff(
            "@@ -12,4 +12,0 @@\n-        return a\n-\n"
            "-    def gamma(self):\n-        return 2\n"
        )
    )
    assert _names(truth) == ["K.beta", "K.gamma"]


def test_multiple_hunks_across_functions_are_deduplicated():
    patch = _diff(
        "@@ -5 +5 @@\n-    y = x + 1\n+    y = x + 2\n"
        "@@ -6 +6 @@\n-    return y\n+    return y * 2\n"
        "@@ -19 +19 @@\n-    return 3\n+    return 4\n"
    )
    assert _names(_label(patch)) == ["alpha", "delta"]


def test_rewrite_is_not_double_counted_as_unlocated():
    truth = _label(_diff("@@ -1 +1 @@\n-import os\n+import sys\n"))
    assert truth.unlocated_changes == 1


def test_new_non_python_and_unindexed_files_are_separated():
    patch = (
        "--- /dev/null\n+++ b/new_mod.py\n@@ -0,0 +1 @@\n+x = 1\n"
        "--- a/docs/changes.rst\n+++ b/docs/changes.rst\n@@ -1 +1 @@\n-a\n+b\n"
        "--- a/not_indexed.py\n+++ b/not_indexed.py\n@@ -1 +1 @@\n-a\n+b\n"
        "--- a/m.py\n+++ b/m.py\n@@ -5 +5 @@\n-    y = x + 1\n+    y = x + 2\n"
    )
    truth = _label(patch)
    assert truth.files == ("m.py",)
    assert truth.new_files == ("new_mod.py",)
    assert "docs/changes.rst" in truth.non_python_files
    assert "not_indexed.py" in truth.non_python_files


def test_deleted_python_file_is_relevant_without_units():
    patch = "--- a/m.py\n+++ /dev/null\n@@ -1,2 +0,0 @@\n-import os\n-\n"
    truth = _label(patch)
    assert truth.files == ("m.py",)
    assert truth.units == ()


def test_labels_are_deterministic_and_sorted():
    patch = _diff(
        "@@ -19 +19 @@\n-    return 3\n+    return 4\n"
        "@@ -5 +5 @@\n-    y = x + 1\n+    y = x + 2\n"
    )
    first, second = _label(patch), _label(patch)
    assert first == second
    assert [u.start_line for u in first.units] == [4, 18]


def test_nested_function_edit_labels_the_enclosing_unit():
    src = "def outer():\n    def inner():\n        return 1\n    return inner()\n"
    extracted = extract_file("n.py", "python", src.encode())
    patch = (
        "--- a/n.py\n+++ b/n.py\n@@ -3 +3 @@\n-        return 1\n+        return 2\n"
    )
    truth = label_patch(patch, {"n.py": extracted.units}, {"n.py"})
    assert [u.qualified_name for u in truth.units] == ["outer"]
