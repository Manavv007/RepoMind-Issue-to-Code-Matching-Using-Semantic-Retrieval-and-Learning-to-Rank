"""Data-leakage controls for the evaluation set (project.md §10.3).

* **Repo-aware split**: all cases from one repository land in the same split, so
  tuning or training on ``dev`` never sees the repositories used for ``test``.
* **Deduplication**: near-identical issues and identical gold patches are
  collapsed so one underlying bug cannot be scored twice.
* **Patch-hint check**: flags issues that already quote lines of the fix.
"""

from __future__ import annotations

import hashlib
import re
from collections.abc import Iterable, Sequence
from dataclasses import dataclass

from evaluation.schema import EvalCase, Split

_WORD_RE = re.compile(r"[a-z0-9_]+")
_WS_RE = re.compile(r"\s+")
_MIN_HINT_CHARS = 25
_TRIVIAL_LINE_RE = re.compile(r"^[\W_]*$")


def normalize_whitespace(text: str) -> str:
    return _WS_RE.sub(" ", text).strip()


def shingles(text: str, n: int = 3) -> frozenset[str]:
    """Word n-grams of lowercased alphanumeric tokens."""
    words = _WORD_RE.findall(text.lower())
    if len(words) < n:
        return frozenset({" ".join(words)}) if words else frozenset()
    return frozenset(" ".join(words[i : i + n]) for i in range(len(words) - n + 1))


def jaccard(a: frozenset[str], b: frozenset[str]) -> float:
    if not a and not b:
        return 1.0
    union = len(a | b)
    return len(a & b) / union if union else 0.0


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


# ------------------------------------------------------------------ patch hints
def count_patch_lines_in_issue(
    issue_text: str, added_lines: Iterable[str], min_chars: int = _MIN_HINT_CHARS
) -> int:
    """Count distinct added patch lines that appear verbatim in the issue.

    Lines are compared after whitespace normalization. Short or punctuation-only
    lines (``)``, ``else:``, ``return``) are ignored since they match by chance.
    """
    haystack = normalize_whitespace(issue_text)
    seen: set[str] = set()
    for raw in added_lines:
        line = normalize_whitespace(raw)
        if len(line) < min_chars or _TRIVIAL_LINE_RE.match(line):
            continue
        if line in haystack:
            seen.add(line)
    return len(seen)


def mentions_gold_file(issue_text: str, gold_files: Sequence[str]) -> bool:
    """Informational: does the issue name a gold file or its dotted module path?"""
    text = issue_text
    for path in gold_files:
        module = path[:-3].replace("/", ".") if path.endswith(".py") else path
        basename = path.rsplit("/", 1)[-1]
        if path in text or module in text:
            return True
        if len(basename) >= 6 and basename != "__init__.py" and basename in text:
            return True
    return False


# ------------------------------------------------------------------ duplicates
@dataclass(frozen=True)
class DedupItem:
    case_id: str
    issue_text: str
    patch_sha256: str


def find_duplicates(
    items: Sequence[DedupItem], threshold: float = 0.8
) -> dict[str, str]:
    """Map each duplicate ``case_id`` to the earlier case it duplicates.

    Items are processed in ``case_id`` order so the kept representative is
    deterministic. Two cases are duplicates when they share a gold-patch hash or
    their issue texts have 3-gram Jaccard similarity >= ``threshold``.
    """
    kept: list[tuple[DedupItem, frozenset[str]]] = []
    duplicates: dict[str, str] = {}
    for item in sorted(items, key=lambda i: i.case_id):
        sig = shingles(item.issue_text)
        match: str | None = None
        for other, other_sig in kept:
            if item.patch_sha256 == other.patch_sha256 or (
                jaccard(sig, other_sig) >= threshold
            ):
                match = other.case_id
                break
        if match is None:
            kept.append((item, sig))
        else:
            duplicates[item.case_id] = match
    return duplicates


# ---------------------------------------------------------------------- splits
def assign_splits(
    repos: Iterable[str],
    *,
    heldout_repos: Iterable[str] | None = None,
    heldout_fraction: float = 0.4,
    seed: int = 0,
) -> dict[str, Split]:
    """Assign each repository to ``dev`` or ``test`` deterministically.

    ``test`` is the held-out split used for final reporting. If ``heldout_repos``
    is given it is used verbatim; otherwise repositories are ranked by a seeded
    hash and the first ``round(fraction * n)`` (at least one, never all when
    ``n >= 2``) become ``test``. A lone repository is placed in ``test``.
    """
    unique = sorted(set(repos))
    if heldout_repos is not None:
        held = set(heldout_repos)
        return {r: ("test" if r in held else "dev") for r in unique}
    if len(unique) <= 1:
        return {r: "test" for r in unique}
    ranked = sorted(
        unique, key=lambda r: hashlib.sha256(f"{seed}:{r}".encode()).hexdigest()
    )
    n_test = min(max(1, round(heldout_fraction * len(ranked))), len(ranked) - 1)
    held = set(ranked[:n_test])
    return {r: ("test" if r in held else "dev") for r in unique}


def repos_spanning_splits(cases: Iterable[EvalCase]) -> list[str]:
    """Repositories that appear in more than one split (should be empty)."""
    seen: dict[str, set[str]] = {}
    for case in cases:
        if case.split is not None:
            seen.setdefault(case.repo, set()).add(case.split)
    return sorted(repo for repo, splits in seen.items() if len(splits) > 1)
