"""The agent's tools. Each returns plain JSON; every rule-derived fact carries clause ids + spans.

The toolbox records every clause id it has returned, so the agent's final answer can be checked:
a citation is only valid if the clause actually came back from a tool during this run.
"""
from __future__ import annotations

import json
from datetime import date
from typing import Any

from .config import AS_OF, BASELINE_CARD, CARDS, POINT_VALUE_CENTS
from .credits import credit_status, fee_value
from .db import connect, flagged_rules, load_terms_db, query_transactions
from .engine import simulate
from .schema import CardTerms, Transaction
from .search import default_index

CARD_ENUM = list(CARDS)

TOOL_SPECS: list[dict] = [
    {"name": "search_terms", "description": "Hybrid BM25 + embedding search over the card terms clauses.",
     "parameters": {"type": "object", "properties": {
         "card": {"type": ["string", "null"], "enum": CARD_ENUM + [None]},
         "query": {"type": "string"}}, "required": ["query"]}},
    {"name": "get_rules", "description": "Structured, span-cited rules for a card. category is an earning "
     "category (dining, us_supermarkets, flights, hotels, travel, drugstores, gas, other), 'credits', 'fees', or null for all.",
     "parameters": {"type": "object", "properties": {
         "card": {"type": "string", "enum": CARD_ENUM}, "category": {"type": ["string", "null"]}},
         "required": ["card"]}},
    {"name": "get_transactions", "description": "The cardholder's (synthetic) transactions.",
     "parameters": {"type": "object", "properties": {"filters": {"type": "object", "properties": {
         "card": {"type": "string"}, "start": {"type": "string", "description": "YYYY-MM-DD"},
         "end": {"type": "string"}, "category": {"type": "string"}, "merchant": {"type": "string"},
         "limit": {"type": "integer"}}}}, "required": []}},
    {"name": "credit_status", "description": "Used / remaining / expiry for each credit in the period containing "
     "the given month. card may be 'all'.",
     "parameters": {"type": "object", "properties": {
         "card": {"type": "string", "enum": CARD_ENUM + ["all"]}, "month": {"type": "string", "description": "YYYY-MM"}},
         "required": ["card"]}},
    {"name": "simulate", "description": "Rank all cards for one purchase with the rule engine (exclusions, caps, "
     "booking channel, foreign fees).",
     "parameters": {"type": "object", "properties": {
         "merchant": {"type": "string"}, "amount": {"type": "number"},
         "merchant_category": {"type": "string"},
         "payment_method": {"type": "string", "enum": ["card_present", "online", "mobile_card_reader",
                                                       "third_party_processor", "digital_wallet"]},
         "foreign": {"type": "boolean"}},
         "required": ["merchant", "amount", "merchant_category", "payment_method"]}},
    {"name": "fee_value", "description": "Is the annual fee worth it? Credits actually used plus extra rewards vs. "
     "the no-fee baseline card over the trailing 12 months.",
     "parameters": {"type": "object", "properties": {"card": {"type": "string", "enum": CARD_ENUM}},
                    "required": ["card"]}},
]


def _strip(obj: Any) -> Any:
    """Make Sourced fields compact for the LLM: {"value", "clause_id", "span": [start, end]}."""
    if isinstance(obj, dict):
        if "value" in obj and "span" in obj and isinstance(obj["span"], dict):
            sp = obj["span"]
            return {"value": _strip(obj["value"]), "clause_id": sp["clause_id"], "span": [sp["start"], sp["end"]]}
        return {k: _strip(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_strip(x) for x in obj]
    return obj


class Toolbox:
    def __init__(self, terms: dict[str, CardTerms], txns: list[Transaction], as_of: date = AS_OF,
                 flagged: dict[str, list[str]] | None = None, allowed: set[str] | None = None):
        self.terms, self.txns, self.as_of = terms, txns, as_of
        self.flagged = flagged or {}
        self.allowed = allowed  # restrict the tool set (used by the eval's RAG-only setup)
        self.seen_clauses: set[str] = set()
        self.calls: list[dict] = []

    @classmethod
    def from_db(cls, db_path=None, **kw) -> "Toolbox":
        with connect(db_path) as con:
            return cls(load_terms_db(con), query_transactions(con), flagged=flagged_rules(con), **kw)

    @property
    def specs(self) -> list[dict]:
        return [s for s in TOOL_SPECS if self.allowed is None or s["name"] in self.allowed]

    def call(self, name: str, args: dict) -> Any:
        if self.allowed is not None and name not in self.allowed:
            out: Any = {"error": f"tool {name} is not available"}
        else:
            try:
                out = getattr(self, name)(**args)
            except Exception as e:  # tool errors go back to the model instead of crashing the loop
                out = {"error": f"{type(e).__name__}: {e}"}
        self._collect(out)
        self.calls.append({"name": name, "args": args})
        return out

    def _collect(self, out: Any) -> None:
        s = json.dumps(out, default=str)
        import re

        self.seen_clauses.update(re.findall(r'"clause_id": "([A-Z]{2}-\d+)"', s))

    # ---------------------------------------------------------------- tools

    def search_terms(self, query: str, card: str | None = None, k: int = 5) -> list[dict]:
        return default_index().search(query, card=card, k=k)

    def get_rules(self, card: str, category: str | None = None) -> dict:
        t = self.terms[card]
        d = json.loads(t.model_dump_json())
        out: dict[str, Any] = {"card": card, "name": t.name}
        if category in (None, "fees"):
            out["fees"] = _strip(d["fees"])
        if category in (None, "credits"):
            out["credits"] = _strip(d["credits"])
        if category not in ("fees", "credits"):
            rules = [r for r in d["earning_rules"] if category is None or r["category"]["value"] == category]
            out["earning_rules"] = _strip(rules)
        for r in out.get("earning_rules", []) + out.get("credits", []):
            if r["id"] in self.flagged:
                r["validator_flags"] = self.flagged[r["id"]]
        return out

    def get_transactions(self, filters: dict | None = None) -> list[dict]:
        f = filters or {}
        rows = [t for t in self.txns
                if (not f.get("card") or t.card == f["card"]) and (not f.get("category") or t.category == f["category"])
                and (not f.get("start") or t.date.isoformat() >= f["start"])
                and (not f.get("end") or t.date.isoformat() <= f["end"])
                and (not f.get("merchant") or f["merchant"].lower() in t.merchant.lower())]
        return [json.loads(t.model_dump_json()) for t in rows[: f.get("limit") or 200]]

    def credit_status(self, card: str = "all", month: str | None = None) -> list[dict]:
        if month:
            y, m = map(int, month.split("-"))
            ref = self.as_of if (y, m) == (self.as_of.year, self.as_of.month) else date(y, m, 1)
        else:
            ref = self.as_of
        cards = list(self.terms) if card == "all" else [card]
        out = []
        for c in cards:
            out += [json.loads(s.model_dump_json()) for s in credit_status(self.terms[c], self.txns, ref, self.as_of)]
        return out

    def simulate(self, merchant: str, amount: float, merchant_category: str, payment_method: str = "card_present",
                 foreign: bool = False) -> dict:
        res = simulate(merchant, float(amount), merchant_category, payment_method, rules=self.terms,
                       foreign=foreign, point_values=POINT_VALUE_CENTS, flagged=self.flagged)
        return json.loads(res.model_dump_json())

    def fee_value(self, card: str) -> dict:
        fv = fee_value(self.terms[card], self.terms[BASELINE_CARD], self.txns, self.as_of,
                       POINT_VALUE_CENTS[card], POINT_VALUE_CENTS[BASELINE_CARD])
        return json.loads(fv.model_dump_json())
