"""Code-aware tokenizer shared by lexical retrieval over issues and source code.

Goals (project.md §6.3): never destroy technical tokens. Identifiers are split
into sub-words *and* kept whole, so ``restoreSession`` in a bug report matches
both ``restore_session`` (via ``restore``/``session``) and ``restoreSession``
itself (via the whole-identifier token).

    >>> tokenize("HTTPAdapter.send_request", use_stopwords=False)
    ['httpadapter', 'http', 'adapter', 'send_request', 'send', 'request']
"""

from __future__ import annotations

import re

# Identifier-like words (Unicode aware) or digit runs.
_WORD_RE = re.compile(r"[^\W\d]\w*|\d+")
# camelCase / PascalCase / ACRONYM / digit pieces within one underscore-free chunk.
_PIECE_RE = re.compile(r"[A-Z]+(?=[A-Z][a-z])|[A-Z]?[a-z]+|[A-Z]+|\d+")

_MIN_TOKEN_LEN = 2

ENGLISH_STOPWORDS: frozenset[str] = frozenset(
    """a about above after again against all also am an and any are as at be because
    been before being below between both but by can could did do does doing down
    during each few for from further had has have having he her here hers him his
    how i if in into is it its itself just me more most my no nor not now of off on
    once only or other our ours out over own same she should so some such than that
    the their theirs them then there these they this those through to too under
    until up very was we were what when where which while who whom why will with
    would you your yours""".split()
)

# Python syntax words and filename noise: present in nearly every function, so they
# carry no signal for ranking.
CODE_STOPWORDS: frozenset[str] = frozenset(
    """self cls def class return import elif else pass lambda try except finally
    raise assert yield async await global nonlocal del py""".split()
)

STOPWORDS: frozenset[str] = ENGLISH_STOPWORDS | CODE_STOPWORDS


def split_identifier(identifier: str) -> list[str]:
    """Split ``snake_case``, ``camelCase``, ``PascalCase`` and ``ACRONYMs``.

    Returned pieces are lowercased and do not include the whole identifier.
    """
    pieces: list[str] = []
    for chunk in identifier.split("_"):
        if chunk:
            pieces.extend(p.lower() for p in _PIECE_RE.findall(chunk))
    return pieces


def tokenize(text: str, *, use_stopwords: bool = True) -> list[str]:
    """Tokenize free text or source code into lowercase search terms.

    For every identifier the lowercased whole token is emitted first, followed by
    its sub-words when it has more than one. Tokens shorter than two characters
    are dropped; stopwords are removed unless ``use_stopwords`` is False.
    """
    tokens: list[str] = []
    for match in _WORD_RE.finditer(text):
        word = match.group(0)
        whole = word.lower()
        pieces = split_identifier(word) if not word.isdigit() else []
        candidates = [whole]
        if len(pieces) > 1 or (pieces and pieces[0] != whole):
            candidates.extend(pieces)
        for token in candidates:
            if len(token) < _MIN_TOKEN_LEN:
                continue
            if use_stopwords and token in STOPWORDS:
                continue
            tokens.append(token)
    return tokens
