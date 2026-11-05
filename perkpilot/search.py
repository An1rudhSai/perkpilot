"""Hybrid clause search: BM25 + dense embeddings, fused by weighted normalized score.

The dense side uses a real embeddings endpoint when PERKPILOT_EMBED_MODEL is set and an API key
is configured; otherwise a local hashed n-gram embedding so it works offline and in CI.
"""
from __future__ import annotations

import math
import re
import zlib
from collections import Counter
from functools import lru_cache

import numpy as np

from .config import llm_settings
from .terms import Clause, available_cards, load_terms

TOKEN = re.compile(r"[a-z0-9$%+.']+")
STOP = set("""a an the of to in on at for and or is are be by with from your you it this that do does
can i my me what which how much many any per will was there these those than then as if""".split())
DIM = 1024


def tokenize(text: str) -> list[str]:
    toks = [t.strip(".'") for t in TOKEN.findall(text.lower())]
    return [stem(t) for t in toks if t and t not in STOP]


def stem(t: str) -> str:
    for suf in ("ings", "ing", "ed", "es", "s"):
        if len(t) > 4 and t.endswith(suf):
            return t[: -len(suf)]
    return t


def hashed_embedding(text: str) -> np.ndarray:
    """Local fallback embedding: hashed word unigrams + character trigrams, L2-normalized."""
    v = np.zeros(DIM)
    words = tokenize(text)
    feats = words + [w[i:i + 3] for w in words for i in range(max(len(w) - 2, 1))]
    for f in feats:
        h = zlib.crc32(f.encode())
        v[h % DIM] += 1.0 if (h >> 16) & 1 else -1.0
    n = np.linalg.norm(v)
    return v / n if n else v


class BM25:
    def __init__(self, docs: list[list[str]], k1: float = 1.4, b: float = 0.75):
        self.docs, self.k1, self.b = docs, k1, b
        self.avg = sum(map(len, docs)) / max(len(docs), 1)
        df = Counter(t for d in docs for t in set(d))
        n = len(docs)
        self.idf = {t: math.log(1 + (n - f + 0.5) / (f + 0.5)) for t, f in df.items()}
        self.tf = [Counter(d) for d in docs]

    def scores(self, query: list[str]) -> np.ndarray:
        out = np.zeros(len(self.docs))
        for i, (tf, d) in enumerate(zip(self.tf, self.docs)):
            norm = self.k1 * (1 - self.b + self.b * len(d) / self.avg)
            out[i] = sum(self.idf.get(q, 0) * tf[q] * (self.k1 + 1) / (tf[q] + norm) for q in query if q in tf)
        return out


class ClauseIndex:
    def __init__(self, clauses: list[Clause], alpha: float = 0.5):
        self.clauses, self.alpha = clauses, alpha
        self.bm25 = BM25([tokenize(c.text) for c in clauses])
        self._embed = self._pick_embedder()
        self.vecs = np.stack([self._embed(c.text) for c in clauses])
        self.embedder = "hashed-ngram (local)" if self._embed is hashed_embedding else "remote"

    @staticmethod
    def _pick_embedder():
        s = llm_settings()
        if s["mode"] == "openai" and s["embed_model"] and s["api_key"]:
            from .llm import OpenAICompatLLM

            client = OpenAICompatLLM(s["base_url"], s["api_key"], s["model"])

            def remote(text: str) -> np.ndarray:
                v = np.array(client.embed([text], s["embed_model"])[0])
                return v / (np.linalg.norm(v) or 1)

            return remote
        return hashed_embedding

    def search(self, query: str, card: str | None = None, k: int = 5) -> list[dict]:
        bm = self.bm25.scores(tokenize(query))
        bm = bm / bm.max() if bm.max() > 0 else bm
        dense = np.clip(self.vecs @ self._embed(query), 0, None)
        score = self.alpha * bm + (1 - self.alpha) * dense
        order = np.argsort(-score, kind="stable")
        hits = []
        for i in order:
            c = self.clauses[i]
            if card and c.card != card:
                continue
            hits.append({"clause_id": c.clause_id, "card": c.card, "title": c.title, "text": c.text,
                         "start": c.start, "end": c.end, "score": round(float(score[i]), 4),
                         "bm25": round(float(bm[i]), 4), "dense": round(float(dense[i]), 4)})
            if len(hits) == k:
                break
        return hits


@lru_cache(maxsize=1)
def default_index() -> ClauseIndex:
    return ClauseIndex([c for card in available_cards() for c in load_terms(card)[1]])


def coverage(question: str, text: str, ignore: set[str] = frozenset()) -> float:
    """Share of the question's content words that appear in `text` (used for abstention)."""
    q = [t for t in tokenize(question) if t not in ignore]
    if not q:
        return 0.0
    body = set(tokenize(text))
    return sum(t in body for t in q) / len(q)
