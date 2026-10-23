"""Tests for leakage controls: patch hints, dedup, and repo-aware splits."""

from __future__ import annotations

import pytest

from evaluation.leakage import (
    DedupItem,
    assign_splits,
    count_patch_lines_in_issue,
    find_duplicates,
    jaccard,
    mentions_gold_file,
    repos_spanning_splits,
    sha256_text,
    shingles,
)
from evaluation.schema import EvalCase, GroundTruth, IssueInput, Provenance


# ----------------------------------------------------------------- patch hints
def test_patch_lines_quoted_in_issue_are_counted():
    issue = "Fix: replace it with\n    cright[-n:, -m:] = right_coord_matrix   \nthanks"
    added = ["        cright[-n:, -m:] = right_coord_matrix", "return x", ")"]
    assert count_patch_lines_in_issue(issue, added) == 1


def test_short_and_punctuation_lines_do_not_count_as_hints():
    issue = "return x\nelse:\n)\n"
    assert count_patch_lines_in_issue(issue, ["return x", "else:", ")", "  "]) == 0


def test_unrelated_issue_has_no_hint():
    assert (
        count_patch_lines_in_issue(
            "Something is slow.", ["value = compute_result(a, b)"]
        )
        == 0
    )


def test_hint_matching_ignores_whitespace_differences():
    issue = "do  this:  result = compute_result(a,   b)"
    assert count_patch_lines_in_issue(issue, ["result = compute_result(a, b)"]) == 1


def test_mentions_gold_file_variants():
    files = ["pkg/sub/module.py"]
    assert mentions_gold_file("see pkg/sub/module.py line 3", files)
    assert mentions_gold_file("File pkg.sub.module raised", files)
    assert mentions_gold_file("in module.py there is a bug", files)
    assert not mentions_gold_file("something unrelated happened", files)
    assert not mentions_gold_file("see __init__.py", ["pkg/__init__.py"])


# ------------------------------------------------------------------ duplicates
def _item(cid: str, text: str, patch: str = "p") -> DedupItem:
    return DedupItem(cid, text, sha256_text(patch + cid))


def test_identical_issue_text_is_a_duplicate_of_the_earlier_case():
    text = "Session cookies are dropped when following redirects with POST requests"
    dups = find_duplicates([_item("b-2", text), _item("a-1", text)])
    assert dups == {"b-2": "a-1"}  # earliest id is kept


def test_near_identical_text_is_caught_but_different_text_is_not():
    base = (
        "The parser crashes when the input file contains a trailing comma in the list "
        * 3
    )
    similar = base + "extra"
    other = (
        "Memory usage grows without bound when the cache is never evicted "
        "for stale keys"
    )
    dups = find_duplicates([_item("a", base), _item("b", similar), _item("c", other)])
    assert dups == {"b": "a"}


def test_same_gold_patch_marks_duplicate_even_with_different_text():
    same = sha256_text("identical patch")
    items = [
        DedupItem("a", "first description of the problem", same),
        DedupItem("b", "completely different words about something else", same),
    ]
    assert find_duplicates(items) == {"b": "a"}


def test_dedup_is_order_independent():
    text = "same text here for the duplicate detection check to find easily"
    forward = find_duplicates([_item("a", text), _item("b", text), _item("c", text)])
    backward = find_duplicates([_item("c", text), _item("b", text), _item("a", text)])
    assert forward == backward == {"b": "a", "c": "a"}


def test_jaccard_and_shingles_edge_cases():
    assert jaccard(frozenset(), frozenset()) == 1.0
    assert jaccard(shingles("a b c d"), shingles("a b c d")) == 1.0
    assert jaccard(shingles("a b c d"), shingles("w x y z")) == 0.0
    assert shingles("") == frozenset()


# ---------------------------------------------------------------------- splits
def test_explicit_heldout_repos_are_used_verbatim():
    splits = assign_splits(["a/x", "b/y", "c/z"], heldout_repos=["b/y"])
    assert splits == {"a/x": "dev", "b/y": "test", "c/z": "dev"}


def test_fraction_split_is_deterministic_and_never_empty_or_total():
    repos = [f"o/r{i}" for i in range(6)]
    first = assign_splits(repos, seed=3)
    assert first == assign_splits(reversed(repos), seed=3)
    values = set(first.values())
    assert values == {"dev", "test"}
    assert sum(1 for s in first.values() if s == "test") == 2  # round(0.4 * 6)


def test_split_depends_on_seed():
    repos = [f"o/r{i}" for i in range(8)]
    assert any(
        assign_splits(repos, seed=0) != assign_splits(repos, seed=s)
        for s in range(1, 6)
    )


def test_two_repos_get_one_split_each_and_single_repo_is_test():
    two = assign_splits(["a/x", "b/y"])
    assert sorted(two.values()) == ["dev", "test"]
    assert assign_splits(["a/x"]) == {"a/x": "test"}
    assert assign_splits([]) == {}


def test_repos_spanning_splits_detects_violations():
    def case(cid: str, repo: str, split: str) -> EvalCase:
        return EvalCase(
            case_id=cid,
            repo=repo,
            repo_url=f"https://github.com/{repo}",
            base_commit="a" * 40,
            issue=IssueInput(body="x"),
            ground_truth=GroundTruth(),
            suitable=True,
            split=split,  # type: ignore[arg-type]
            provenance=Provenance(
                dataset="d", instance_id=cid, license_note="n", patch_sha256="h"
            ),
        )

    ok = [case("1", "a/x", "dev"), case("2", "a/x", "dev"), case("3", "b/y", "test")]
    assert repos_spanning_splits(ok) == []
    bad = ok + [case("4", "a/x", "test")]
    assert repos_spanning_splits(bad) == ["a/x"]


@pytest.mark.parametrize("fraction", [0.1, 0.5, 0.9])
def test_fraction_never_produces_empty_side_for_multiple_repos(fraction: float):
    splits = assign_splits(["a/1", "b/2", "c/3", "d/4"], heldout_fraction=fraction)
    assert set(splits.values()) == {"dev", "test"}
