"""Second-pass validation: a separate LLM call re-reads each rule against its source clause.

Disagreements are not auto-corrected; they are logged (SQLite + JSONL) for human review and
the rule is flagged so answers that depend on it can say so.
"""
from __future__ import annotations

import json

from .llm import LLM
from .schema import CardTerms, ValidationRecord, ValidationVerdict
from .terms import clause_by_id

VALIDATE_SYSTEM = """TASK: validate_rule
You are auditing an automatically extracted credit-card rule. You get the rule (values only)
and the exact clause it was extracted from. Check every value against the clause. Flag values
the clause does not state, values taken from a sentence about a different case, and amounts
or conditions in the clause that the rule fails to capture. Return JSON:
{"agrees": true|false, "issues": ["..."]}."""


def _values(model) -> dict:
    """Strip spans: {"field": {"value": v, "span": ...}} -> {"field": v}."""
    out = {}
    for k, v in model.model_dump().items():
        if isinstance(v, dict) and "value" in v and "span" in v:
            out[k] = v["value"]
        elif isinstance(v, list) and v and isinstance(v[0], dict) and "span" in v[0]:
            out[k] = [x["value"] for x in v]
        else:
            out[k] = v
    return out


def _ask(llm: LLM, kind: str, rule: dict, clause_text: str) -> ValidationVerdict:
    msg = [{"role": "system", "content": VALIDATE_SYSTEM},
           {"role": "user", "content": json.dumps({"kind": kind, "rule": rule, "clause_text": clause_text})}]
    res = llm.chat(msg, json_mode=True)
    try:
        return ValidationVerdict.model_validate_json(res.content or "{}")
    except Exception as e:  # a validator that can't answer is itself a disagreement worth reviewing
        return ValidationVerdict(agrees=False, issues=[f"validator output unparseable: {e}"])


def validate_terms(terms: CardTerms, llm: LLM) -> list[ValidationRecord]:
    records: list[ValidationRecord] = []
    for r in terms.earning_rules:
        v = _ask(llm, "earning_rule", _values(r), clause_by_id(r.clause_id).text)
        records.append(ValidationRecord(card=terms.card, rule_id=r.id, kind="earning_rule",
                                        agrees=v.agrees, issues=v.issues, validator=llm.name))
    for c in terms.credits:
        v = _ask(llm, "credit", _values(c), clause_by_id(c.clause_id).text)
        records.append(ValidationRecord(card=terms.card, rule_id=c.id, kind="credit",
                                        agrees=v.agrees, issues=v.issues, validator=llm.name))
    fee_vals = _values(terms.fees)
    clauses = {s.span.clause_id for s in (terms.fees.annual_fee, terms.fees.foreign_transaction_fee_pct) if s}
    for cid in sorted(clauses):
        present = [k for k in ("annual_fee", "foreign_transaction_fee_pct")
                   if getattr(terms.fees, k) and getattr(terms.fees, k).span.clause_id == cid]
        v = _ask(llm, "fee", {**fee_vals, "_present": present}, clause_by_id(cid).text)
        records.append(ValidationRecord(card=terms.card, rule_id=f"{terms.card}.fees.{cid}", kind="fee",
                                        agrees=v.agrees, issues=v.issues, validator=llm.name))
    return records
