"""Offline "scripted LLM": a deterministic stand-in so extraction, validation, the agent and the
eval all run with no API key and no network.

It implements the same `chat()` interface as the real client and dispatches on the
`TASK: <name>` line at the top of the system prompt. Its behaviour is hand-written heuristics
(regexes over the terms text, keyword intent rules), NOT a language model. Any number produced
in this mode is labelled "offline/scripted" in the eval output.
"""
from __future__ import annotations

import json
import re
from typing import Any

from .llm import ChatResult

# ---------------------------------------------------------------- extraction

TITLE_CATEGORIES = {
    "restaurants": "dining",
    "dining": "dining",
    "u.s. supermarkets": "us_supermarkets",
    "flights": "flights",
    "prepaid hotels": "hotels",
    "hotels": "hotels",
    "chase travel": "travel",
    "drugstores": "drugstores",
    "gas stations": "gas",
    "everything else": "other",
}

EXCLUSION_VOCAB = {
    "mobile card reader": "mobile_card_reader",
    "third-party payment processor": "third_party_processor",
    "superstores": "superstore",
    "warehouse clubs": "warehouse_club",
    "convenience stores": "convenience_store",
    "meal-kit": "meal_kit",
    "gift card": "gift_card",
    "cash advances": "cash_advance",
    "balance transfers": "balance_transfer",
    "fees": "fee",
}

CHANNEL_PHRASES = {
    "directly with airlines": "airline_direct",
    "directly with the airline": "airline_direct",
    "directly with the hotel": "hotel_direct",
    "through amex travel": "amex_travel",
    "through chase travel": "chase_travel",
}

NEGATION = re.compile(r"\b(not|do not|does not|may not|excluded)\b", re.I)
NUM = r"(\d[\d,]*(?:\.\d+)?)"


def _num(s: str) -> float:
    return float(s.replace(",", ""))


def _field(value: Any, quote: str) -> dict:
    return {"value": value, "quote": quote}


def _sentences(text: str) -> list[str]:
    # split on ". " but not inside "U.S." / "Wine.com"
    return [s for s in re.split(r"(?<=[a-z0-9)%])\.\s+(?=[A-Z])", text) if s]


def _list_items(s: str) -> list[str]:
    return [x.strip() for x in re.split(r",\s*|\s+and\s+", s) if x.strip()]


def extract_clause_offline(title: str, text: str) -> dict:
    out: dict[str, Any] = {"earning_rules": [], "credits": [], "fees": {}}
    low_title = title.lower()

    if low_title == "annual fee":
        m = re.search(r"\$" + NUM, text)
        if m:
            out["fees"]["annual_fee"] = _field(_num(m[1]), m[0])
        return out
    if low_title == "foreign transactions":
        m = re.search(r"no foreign transaction fee", text, re.I)
        if m:
            out["fees"]["foreign_transaction_fee_pct"] = _field(0.0, m[0])
        else:
            m = re.search(NUM + r"% of each transaction", text)
            if m:
                out["fees"]["foreign_transaction_fee_pct"] = _field(_num(m[1]), m[0])
        return out

    is_credit = "credit" in low_title or "uber cash" in low_title
    if is_credit:
        m = re.search(r"up to \$" + NUM, text)
        if not m:
            return out
        credit: dict[str, Any] = {"name": _field(title, title), "amount": _field(_num(m[1]), m[0])}
        for pat, val in [(r"twice per year", "semi_annual"), (r"each month", "monthly"),
                         (r"each (?:calendar|cardmember) year", "annual")]:
            pm = re.search(pat, text, re.I)
            if pm:
                credit["period"] = _field(val, pm[0])
                break
        else:
            return out  # no period -> not a credit we can represent
        for pat, val in [(r"first day of each calendar month", "calendar_month"),
                         (r"once from January through June and once from July through December", "calendar_half"),
                         (r"each calendar year on January 1", "calendar_year"),
                         (r"account anniversary", "cardmember_year")]:
            rm = re.search(pat, text, re.I)
            if rm:
                credit["reset"] = _field(val, rm[0])
                break
        mm = re.search(r"Eligible merchants: (.+?)\.(?=\s|$)", text)
        if mm:
            credit["eligible_merchants"] = [_field(x, x) for x in _list_items(mm[1])]
        em = re.search(r"no (?:separate )?enrollment is required", text, re.I)
        if em:
            credit["enrollment_required"] = _field(False, em[0])
        else:
            em = re.search(r"enrollment is required", text, re.I)
            if em:
                credit["enrollment_required"] = _field(True, em[0])
        xm = re.search(r"Excluded: (.+?)\.(?=\s|$)", text)
        if xm:
            credit["exclusions"] = [_field(x, x) for x in _list_items(xm[1])]
        out["credits"].append(credit)
        return out

    category = TITLE_CATEGORIES.get(low_title)
    mult = re.search(NUM + r" points? per dollar", text)
    if not category or not mult:
        return out
    rule: dict[str, Any] = {
        "category": _field(category, title),
        "multiplier": _field(_num(mult[1]), mult[0]),
    }
    cm = re.search(
        r"up to \$" + NUM + r" in purchases per (calendar year|cardmember year|calendar month), then "
        + NUM + r" points? per dollar", text)
    if cm:
        rule["cap"] = _field(
            {"amount": _num(cm[1]), "period": cm[2].replace(" ", "_"), "after_multiplier": _num(cm[3])}, cm[0])
    lows = text.lower()
    hits = sorted((lows.find(p), p, v) for p, v in CHANNEL_PHRASES.items() if p in lows)
    if hits:
        start = hits[0][0]
        end = max(i + len(p) for i, p, _ in hits)
        rule["booking_channel"] = _field(sorted({v for _, _, v in hits}), text[start:end])
    excl = []
    for sent in _sentences(text):
        if not NEGATION.search(sent):
            continue
        for phrase, val in EXCLUSION_VOCAB.items():
            m = re.search(r"\b" + re.escape(phrase), sent, re.I)
            if m and val not in [e["value"] for e in excl]:
                excl.append(_field(val, m[0]))
    rule["exclusions"] = excl
    out["earning_rules"].append(rule)
    return out


class ScriptedLLM:
    name = "offline-scripted"

    def chat(self, messages: list[dict], tools: list[dict] | None = None, json_mode: bool = False) -> ChatResult:
        system = messages[0]["content"] if messages and messages[0]["role"] == "system" else ""
        task = system.splitlines()[0].removeprefix("TASK:").strip() if system.startswith("TASK:") else ""
        handler = getattr(self, f"_task_{task}", None)
        if handler is None:
            raise ValueError(f"ScriptedLLM has no script for task {task!r}")
        return handler(messages, tools)

    def _task_extract_clause(self, messages, tools) -> ChatResult:
        payload = json.loads(messages[-1]["content"])
        return ChatResult(content=json.dumps(extract_clause_offline(payload["title"], payload["text"])))


# ---------------------------------------------------------------- validation

def _fmt(x: float) -> str:
    return f"{x:,.0f}" if float(x).is_integer() else f"{x:g}"


def validate_offline(kind: str, rule: dict, clause: str) -> dict:
    """Deterministic re-check of one extracted rule against its clause text."""
    issues: list[str] = []
    low = clause.lower()
    if kind == "earning_rule":
        m = rule["multiplier"]
        rate = f"{_fmt(m)} point"
        if rate not in low:
            issues.append(f"multiplier {m:g} is not stated in the clause")
        for ch in rule.get("booking_channel") or []:
            phrases = [p for p, v in CHANNEL_PHRASES.items() if v == ch]
            sents = [s for s in _sentences(clause) if any(p in s.lower() for p in phrases)]
            if not sents:
                issues.append(f"booking channel {ch} is not mentioned")
            elif not any(rate in s.lower() for s in sents):
                other = re.search(NUM + r" points? per dollar", sents[0])
                stated = f" ({other[0]})" if other else ""
                issues.append(f"booking channel {ch} comes from a sentence with a different rate{stated}")
        cap = rule.get("cap")
        if cap and f"${_fmt(cap['amount'])}" not in clause:
            issues.append(f"cap amount ${cap['amount']:g} is not stated")
    elif kind == "credit":
        amt = rule["amount"]
        if f"${_fmt(amt)}" not in clause:
            issues.append(f"amount ${amt:g} is not stated")
        others = {_num(x) for x in re.findall(r"\$" + NUM, clause)} - {amt}
        for o in sorted(others):
            issues.append(f"clause also mentions ${o:g}, which the rule does not capture")
        period_words = {"monthly": "each month", "semi_annual": "twice per year", "annual": " year"}
        if period_words[rule["period"]] not in low:
            issues.append(f"period {rule['period']} is not supported by the clause wording")
        er = rule.get("enrollment_required")
        if er is True and "enrollment is required" not in low:
            issues.append("enrollment_required=true but the clause does not say enrollment is required")
        if er is False and "enrollment is required" in low and "no " not in low.split("enrollment is required")[0][-12:]:
            issues.append("enrollment_required=false but the clause says enrollment is required")
    elif kind == "fee":
        for key in ("annual_fee", "foreign_transaction_fee_pct"):
            v = rule.get(key)
            if v is None:
                continue
            ok = (f"${_fmt(v)}" in clause) if key == "annual_fee" else (
                f"{_fmt(v)}%" in clause or (v == 0 and "no foreign transaction fee" in low))
            if not ok and key in rule.get("_present", []):
                issues.append(f"{key}={v:g} is not stated in its clause")
    return {"agrees": not issues, "issues": issues}


def _task_validate_rule(self, messages, tools) -> ChatResult:
    p = json.loads(messages[-1]["content"])
    return ChatResult(content=json.dumps(validate_offline(p["kind"], p["rule"], p["clause_text"])))


ScriptedLLM._task_validate_rule = _task_validate_rule
