"""Tests for the code-aware tokenizer."""

from __future__ import annotations

import pytest

from app.retrieval.tokenizer import STOPWORDS, split_identifier, tokenize


@pytest.mark.parametrize(
    ("identifier", "expected"),
    [
        ("snake_case_name", ["snake", "case", "name"]),
        ("camelCaseName", ["camel", "case", "name"]),
        ("PascalCaseName", ["pascal", "case", "name"]),
        ("HTTPAdapter", ["http", "adapter"]),
        ("getHTTPResponse", ["get", "http", "response"]),
        ("parseXML", ["parse", "xml"]),
        ("__init__", ["init"]),
        ("_private_helper", ["private", "helper"]),
        ("utf8", ["utf", "8"]),
        ("CONSTANT_VALUE", ["constant", "value"]),
        ("x", ["x"]),
        ("", []),
    ],
)
def test_split_identifier(identifier: str, expected: list[str]):
    assert split_identifier(identifier) == expected


def test_identifiers_are_kept_whole_and_split():
    assert tokenize("restoreSession", use_stopwords=False) == [
        "restoresession",
        "restore",
        "session",
    ]
    assert tokenize("send_request", use_stopwords=False) == [
        "send_request",
        "send",
        "request",
    ]


def test_camel_issue_term_overlaps_snake_code_term_through_subwords():
    issue = set(tokenize("restoreSession fails"))
    code = set(tokenize("def restore_session(token):"))
    assert {"restore", "session"} <= issue & code


def test_single_word_identifier_is_not_duplicated():
    assert tokenize("Session", use_stopwords=False) == ["session"]


def test_dotted_paths_and_file_names_split_into_parts():
    tokens = tokenize("requests/sessions.py", use_stopwords=False)
    assert tokens == ["requests", "sessions", "py"]
    assert "adapters" in tokenize("requests.adapters.HTTPAdapter")


def test_stopwords_and_short_tokens_are_removed():
    tokens = tokenize("The user is not logged in and x = 1 return self.value")
    assert "the" not in tokens and "is" not in tokens and "and" not in tokens
    assert "return" not in tokens and "self" not in tokens
    assert "x" not in tokens and "1" not in tokens
    assert {"user", "logged", "value"} <= set(tokens)


def test_stopwords_can_be_disabled():
    assert "the" in tokenize("the user", use_stopwords=False)
    assert "the" in STOPWORDS


def test_technical_tokens_survive():
    text = "KeyError: 'session_id' at line 404 in HTTP/1.1 handler, utf8 failure"
    tokens = tokenize(text)
    for expected in ("keyerror", "session_id", "404", "http", "handler", "utf8"):
        assert expected in tokens, expected


def test_numbers_and_unicode_identifiers():
    assert "404" in tokenize("error 404")
    assert "grüße" in tokenize("def grüße(): pass")


def test_tokenization_is_deterministic_and_handles_empty_input():
    text = "Traceback: restoreSession raised ValueError"
    assert tokenize(text) == tokenize(text)
    assert tokenize("") == []
    assert tokenize("   \n\t") == []
    assert tokenize("!!! ??? ---") == []
