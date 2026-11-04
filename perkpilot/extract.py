"""Terms text -> structured, span-cited rules.

Each clause is sent to the LLM, which returns values plus verbatim quotes. A quote is then
located inside the clause to get absolute character offsets. A field whose quote cannot be
found is rejected (and logged); a rule missing a required field is rejected entirely.
"""
from __future__ import annotations

import json
import re

from pydantic import BaseModel, Field, ValidationError

from .config import CARDS
from .llm import LLM
from .schema import (
    CardTerms, Credit, EarningRule, Fee, RawClauseExtraction, RawField, Rejection, Sourced, Span,
)
from .terms import Clause, load_terms

EXTRACT_SYSTEM = """TASK: extract_clause
You turn one clause of credit-card benefit terms into JSON. Return an object with keys
"earning_rules" (list), "credits" (list) and "fees" (object). Every field is an object
{"value": ..., "quote": "<exact, verbatim substring of the clause that supports the value>"}.
Never invent a value that the clause does not state; omit the field instead.

earning_rules[]: category (one of dining, us_supermarkets, flights, hotels, travel, drugstores,
gas, other), multiplier (points per dollar, number), cap (value {"amount", "period":
calendar_year|calendar_month|cardmember_year, "after_multiplier"}) or null, exclusions (list of
fields; values from mobile_card_reader, third_party_processor, superstore, warehouse_club,
convenience_store, meal_kit, gift_card, cash_advance, balance_transfer, fee), booking_channel
(value is a list from airline_direct, hotel_direct, amex_travel, chase_travel) or null.
credits[]: name, amount (dollars per period), period (monthly|semi_annual|annual), reset
(calendar_month|calendar_half|calendar_year|cardmember_year), eligible_merchants (list of
fields), enrollment_required (bool), exclusions (list of fields).
fees: annual_fee (dollars), foreign_transaction_fee_pct (percent; 0 if none)."""


class ExtractionResult(BaseModel):
    terms: CardTerms
    rejections: list[Rejection] = Field(default_factory=list)


def _find(quote: str, clause: Clause) -> tuple[int, int] | None:
    """Locate a quote in the clause: exact first, then case/whitespace-insensitive."""
    if not quote:
        return None
    i = clause.text.find(quote)
    if i >= 0:
        return clause.start + i, clause.start + i + len(quote)
    pattern = r"\s+".join(re.escape(w) for w in quote.split())
    m = re.search(pattern, clause.text, re.I)
    if m:
        return clause.start + m.start(), clause.start + m.end()
    return None


class _Resolver:
    def __init__(self, card: str, clause: Clause, rejections: list[Rejection]):
        self.card, self.clause, self.rejections = card, clause, rejections

    def one(self, raw: RawField | None, path: str) -> Sourced | None:
        if raw is None:
            return None
        loc = _find(raw.quote, self.clause)
        if loc is None:
            self.rejections.append(Rejection(
                card=self.card, clause_id=self.clause.clause_id, field=path, value=json.dumps(raw.value, default=str),
                quote=raw.quote, reason="quote not found in clause text"))
            return None
        return Sourced(value=raw.value, span=Span(clause_id=self.clause.clause_id, start=loc[0], end=loc[1]))

    def many(self, raws: list[RawField], path: str) -> list[Sourced]:
        return [s for i, r in enumerate(raws) if (s := self.one(r, f"{path}[{i}]")) is not None]

    def reject_rule(self, path: str, missing: str) -> None:
        self.rejections.append(Rejection(
            card=self.card, clause_id=self.clause.clause_id, field=path, value="", quote="",
            reason=f"rule dropped: required field {missing!r} has no supporting span"))


def _call(llm: LLM, card: str, clause: Clause) -> RawClauseExtraction:
    messages = [
        {"role": "system", "content": EXTRACT_SYSTEM},
        {"role": "user", "content": json.dumps({"card": card, "clause_id": clause.clause_id,
                                                "title": clause.title, "text": clause.text})},
    ]
    last_err: Exception | None = None
    for _ in range(2):  # one repair attempt on malformed output
        res = llm.chat(messages, json_mode=True)
        try:
            return RawClauseExtraction.model_validate_json(res.content or "{}")
        except ValidationError as e:
            last_err = e
            messages += [{"role": "assistant", "content": res.content},
                         {"role": "user", "content": f"That JSON did not match the schema: {e}. Return corrected JSON only."}]
    raise RuntimeError(f"extraction failed for {clause.clause_id}: {last_err}")


def _slug(s: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", s.lower()).strip("_")


def extract_card(card: str, llm: LLM) -> ExtractionResult:
    _, clauses = load_terms(card)
    rejections: list[Rejection] = []
    rules: list[EarningRule] = []
    credits: list[Credit] = []
    fee = Fee(card=card)
    for clause in clauses:
        raw = _call(llm, card, clause)
        r = _Resolver(card, clause, rejections)
        for i, er in enumerate(raw.earning_rules):
            path = f"earning_rules[{i}]"
            category, mult = r.one(er.category, f"{path}.category"), r.one(er.multiplier, f"{path}.multiplier")
            if category is None or mult is None:
                r.reject_rule(path, "category" if category is None else "multiplier")
                continue
            rules.append(EarningRule(
                id=f"{card}.{category.value}" + (f".{i}" if i else ""), card=card, clause_id=clause.clause_id,
                category=category, multiplier=mult, cap=r.one(er.cap, f"{path}.cap"),
                exclusions=r.many(er.exclusions, f"{path}.exclusions"),
                booking_channel=r.one(er.booking_channel, f"{path}.booking_channel")))
        for i, cr in enumerate(raw.credits):
            path = f"credits[{i}]"
            name, amount, period = (r.one(cr.name, f"{path}.name"), r.one(cr.amount, f"{path}.amount"),
                                    r.one(cr.period, f"{path}.period"))
            if name is None or amount is None or period is None:
                r.reject_rule(path, "name" if name is None else "amount" if amount is None else "period")
                continue
            credits.append(Credit(
                id=f"{card}.{_slug(name.value)}", card=card, clause_id=clause.clause_id, name=name, amount=amount,
                period=period, reset=r.one(cr.reset, f"{path}.reset"),
                eligible_merchants=r.many(cr.eligible_merchants, f"{path}.eligible_merchants"),
                enrollment_required=r.one(cr.enrollment_required, f"{path}.enrollment_required"),
                exclusions=r.many(cr.exclusions, f"{path}.exclusions")))
        if raw.fees.annual_fee:
            fee.annual_fee = r.one(raw.fees.annual_fee, "fees.annual_fee") or fee.annual_fee
        if raw.fees.foreign_transaction_fee_pct:
            fee.foreign_transaction_fee_pct = (
                r.one(raw.fees.foreign_transaction_fee_pct, "fees.foreign_transaction_fee_pct")
                or fee.foreign_transaction_fee_pct)
    terms = CardTerms(card=card, name=CARDS[card][0], earning_rules=rules, credits=credits, fees=fee)
    return ExtractionResult(terms=terms, rejections=rejections)


def span_text(card: str, start: int, end: int) -> str:
    return load_terms(card)[0][start:end]
