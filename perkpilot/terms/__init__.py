"""Synthetic, clearly labelled paraphrases of card terms, split into numbered clauses."""
from __future__ import annotations

import re
from dataclasses import dataclass
from functools import lru_cache

from ..config import CARDS, TERMS_DIR

CLAUSE_RE = re.compile(r"^\[(?P<id>[A-Z]{2}-\d+)\] (?P<title>.+?) — (?P<body>.+)$", re.M)


@dataclass(frozen=True)
class Clause:
    card: str
    clause_id: str
    title: str
    start: int  # offset of the clause line in the card's terms document
    end: int
    text: str  # the full clause line, == document[start:end]


@lru_cache(maxsize=None)
def load_terms(card: str) -> tuple[str, tuple[Clause, ...]]:
    if card not in CARDS:
        raise KeyError(f"unknown card {card!r}")
    doc = (TERMS_DIR / f"{card}.txt").read_text()
    clauses = tuple(
        Clause(card, m["id"], m["title"], m.start(), m.end(), m.group(0)) for m in CLAUSE_RE.finditer(doc)
    )
    return doc, clauses


def clause_by_id(clause_id: str) -> Clause | None:
    prefix = clause_id.split("-")[0]
    for card, (_, p) in CARDS.items():
        if p == prefix:
            for c in load_terms(card)[1]:
                if c.clause_id == clause_id:
                    return c
    return None


def available_cards() -> list[str]:
    return [c for c in CARDS if (TERMS_DIR / f"{c}.txt").exists()]
