"""Tests for issue entity extraction and the mention-boosted BM25 retriever."""

from __future__ import annotations

import json

import pytest

from app.config import PROJECT_ROOT
from app.issue_understanding.entities import (
    IssueEntities,
    extract_entities,
    mentions_file,
)
from app.parsing.symbol_extractor import extract_file
from app.retrieval.base import RetrievalQuery
from app.retrieval.bm25 import BM25Retriever
from app.retrieval.mention_boost import MentionBoostConfig, MentionBoostRetriever
from app.retrieval.registry import create_retriever


def test_extracts_backticks_calls_tracebacks_and_identifier_styles():
    text = (
        "Calling `Session.resolve_redirects` fails. parse_header() raises, see\n"
        '  File "requests/sessions.py", line 12, in send_request\n'
        "Also getAdapter and SessionRedirect look wrong; HTTPAdapter too."
    )
    e = extract_entities(text)
    for name in (
        "resolve_redirects",
        "parse_header",
        "send_request",
        "getAdapter",
        "SessionRedirect",
    ):
        assert name in e.names, name
    # Known limit, kept to match the dev-tuned prototype: a bare name that starts
    # with an acronym is only caught when backticked or called.
    assert "HTTPAdapter" not in e.names
    assert "HTTPAdapter" in extract_entities("`HTTPAdapter` breaks").names
    assert "requests/sessions.py" in e.paths


def test_common_builtins_and_plain_words_are_not_names():
    e = extract_entities("print(len(x)) and str(value) fail when the user logs in")
    assert not {"print", "len", "str", "user", "logs"} & e.names


def test_modules_and_windows_paths():
    e = extract_entities(
        r"error in django.db.models.query from .\src\_pytest\logging.py"
    )
    assert "django.db.models.query" in e.modules
    assert "src/_pytest/logging.py" in e.paths


@pytest.mark.parametrize(
    ("file_path", "entities", "expected"),
    [
        (
            "requests/sessions.py",
            IssueEntities(paths=frozenset({"requests/sessions.py"})),
            True,
        ),
        ("src/pkg/mod.py", IssueEntities(paths=frozenset({"pkg/mod.py"})), True),
        ("pkg/mod.py", IssueEntities(paths=frozenset({"mod.py"})), True),
        ("pkg/other.py", IssueEntities(paths=frozenset({"mod.py"})), False),
        (
            "django/db/models/query.py",
            IssueEntities(modules=frozenset({"django.db.models.query"})),
            True,
        ),
        (
            "django/db/models/base.py",
            IssueEntities(modules=frozenset({"django.db.models.query"})),
            False,
        ),
        ("pkg/mod.py", IssueEntities(), False),
    ],
)
def test_mentions_file(file_path, entities, expected):
    assert mentions_file(file_path, entities) is expected


SRC = {
    "pkg/auth.py": (
        "def restore_session(token):\n    return load(token)\n\n\n"
        "def logout(s):\n    return s\n"
    ),
    "pkg/views.py": "def render(user):\n    return template(user)\n",
}


def units():
    out = []
    for path, src in SRC.items():
        out += extract_file(path, "python", src.encode()).units
    return out


def q(text: str) -> RetrievalQuery:
    return RetrievalQuery(query_id="q", issue_body=text)


def test_named_symbol_and_file_are_boosted_with_signals():
    r = MentionBoostRetriever()
    r.prepare(units())
    results = r.retrieve(
        q("`logout` misbehaves after a template change in pkg/auth.py"), 3
    )
    assert results[0].qualified_name == "logout"
    top = results[0].signals
    assert top["mention"] == 1.0 and top["path_mention"] == 1.0
    assert [c.rank for c in results] == list(range(1, len(results) + 1))


def test_zero_weights_reproduce_bm25_order():
    pool = units()
    bm25 = BM25Retriever()
    bm25.prepare(pool)
    boosted = MentionBoostRetriever(MentionBoostConfig(w_mention=0.0, w_path=0.0))
    boosted.prepare(pool)
    query = q("restore_session token template user")
    assert [c.unit_id for c in boosted.retrieve(query, 5)] == [
        c.unit_id for c in bm25.retrieve(query, 5)
    ]


def test_path_mention_alone_can_surface_units_without_lexical_overlap():
    r = MentionBoostRetriever()
    r.prepare(units())
    results = r.retrieve(q("something odd happens in pkg/views.py"), 5)
    assert results[0].file_path == "pkg/views.py"


def test_empty_pool_and_registry():
    r = MentionBoostRetriever()
    r.prepare([])
    assert r.retrieve(q("anything"), 5) == []
    built = create_retriever("bm25-mentions", params={"w_path": 1.0})
    assert built.describe()["w_path"] == 1.0
    with pytest.raises(ValueError):
        create_retriever("bm25-mentions", params={"nope": 1})


def test_defaults_match_the_dev_selected_experiment_weights():
    report = PROJECT_ROOT / "evaluation" / "reports" / "exp_bm25_signals.json"
    if not report.exists():
        pytest.skip("experiment report not present")
    w_mention, w_path, w_file = json.loads(report.read_text())["chosen"]
    cfg = MentionBoostConfig()
    assert (cfg.w_mention, cfg.w_path, w_file) == (w_mention, w_path, 0.0)
