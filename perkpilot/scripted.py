"""Offline scripted policies for the three eval setups (agent, RAG-only, plain LLM).

These are keyword/regex heuristics standing in for a model so everything runs without an API
key. They are deliberately simple and are NOT evidence of how a real LLM would perform.
"""
from __future__ import annotations

import json
import re
from datetime import date

from .config import AS_OF, CARDS
from .offline import NUM, TITLE_CATEGORIES
from .search import coverage, tokenize

CARD_WORDS = [("platinum", "amex_platinum"), ("gold", "amex_gold"), ("sapphire", "sapphire_reserve"),
              ("reserve", "sapphire_reserve"), ("freedom", "freedom_unlimited"), ("unlimited", "freedom_unlimited")]
CATEGORY_WORDS = [
    ("gift_card", ["gift card"]),
    ("drugstores", ["drugstore", "pharmacy", "cvs", "walgreens", "prescription"]),
    ("us_supermarkets", ["grocer", "supermarket", "whole foods", "trader joe", "costco"]),
    ("flights", ["flight", "airfare", "plane ticket"]),
    ("hotels", ["hotel"]),
    ("gas", ["gas station", "gas ", "fuel"]),
    ("dining", ["restaurant", "dinner", "lunch", "brunch", "takeout", "food truck", "taco", "cafe", "bistro",
                "dining", "coffee"]),
]
MERCHANTS = ["Costco", "Whole Foods", "Trader Joe's", "Delta", "United", "Chase Travel", "Amex Travel", "Expedia",
             "Marriott", "Hilton", "CVS", "Walgreens", "Shell", "Target", "Walmart Supercenter"]
FOREIGN = ["abroad", "overseas", "foreign", "paris", "london", "tokyo", "rome", "outside the us",
           "outside the united states", "mexico city"]
IGNORE = set(tokenize("card cards amex chase american express gold platinum sapphire reserve freedom unlimited "
                      "get does have include includes offer come with much is there"))
DATA_INTENTS = {"expiring", "fee_worth", "credit_remaining"}


# ---------------------------------------------------------------- question understanding

def cards_in(q: str) -> list[str]:
    l, out = q.lower(), []
    for w, c in CARD_WORDS:
        if w in l and c not in out:
            out.append(c)
    return out


def category_in(q: str) -> str | None:
    l = q.lower() + " "
    return next((cat for cat, words in CATEGORY_WORDS if any(w in l for w in words)), None)


def purchase_in(q: str) -> dict:
    l = q.lower()
    m = re.search(r"\$" + NUM, q)
    merchant = next((x for x in MERCHANTS if x.lower() in l), None)
    if re.search(r"(through|via|on) chase travel", l):
        merchant = "Chase Travel"
    if re.search(r"(through|via|on) amex travel", l):
        merchant = "Amex Travel"
    if "food truck" in l:
        merchant = "Food truck"
    pm = ("mobile_card_reader" if re.search(r"card reader|square", l) else
          "third_party_processor" if "third-party" in l or "paypal" in l else
          "online" if "online" in l else "card_present")
    cat = category_in(q) or "other"
    if merchant in ("Delta", "United") and cat == "other":
        cat = "flights"
    return {"merchant": merchant or "Merchant", "amount": float(m[1].replace(",", "")) if m else 100.0,
            "merchant_category": cat, "payment_method": pm, "foreign": any(w in l for w in FOREIGN)}


def intent(q: str) -> str:
    l = q.lower()
    if re.search(r"which card|what card|best card|which of my cards|should i (use|pay|put)|card should i", l):
        return "which_card"
    if re.search(r"expir|lose|use it or", l):
        return "expiring"
    if "worth" in l and ("fee" in l or "keep" in l or "worth it" in l):
        return "fee_worth"
    if re.search(r"\b(left|remaining|unused)\b", l) and ("credit" in l or "uber cash" in l):
        return "credit_remaining"
    if "annual fee" in l:
        return "annual_fee"
    if "foreign transaction" in l:
        return "ftf"
    if "enroll" in l:
        return "enrollment"
    if "reset" in l:
        return "reset"
    if re.search(r"\bcap\b|spending limit", l):
        return "cap"
    if re.search(r"booked directly|booked through|booking channel", l) and re.search(r"\d+x|\d+ points", l):
        return "channel_check"
    if re.search(r"points per dollar|how many points|multiplier|\bearn", l) and category_in(q):
        return "multiplier"
    if re.search(r"(credit|uber cash)\b", l) and re.search(r"how much|amount|how big", l):
        return "credit_amount"
    if re.search(r"which merchants|where can i use|eligible merchants", l):
        return "merchants"
    return "search"


def match_credit(q: str, credits: list[dict], key=lambda c: c["name"]["value"]):
    qt = set(tokenize(q.replace("Uber Cash", "uber_cash"))) - IGNORE
    best, score = None, 0
    for c in credits:
        nt = set(tokenize(key(c).replace("Uber Cash", "uber_cash"))) - {"credit"}
        s = len(qt & nt)
        if s > score:
            best, score = c, s
    return best


def cite(f: dict) -> dict:
    return {"clause_id": f["clause_id"], "start": f["span"][0], "end": f["span"][1]}


def answer(text, value=None, claims=(), recs=(), abstained=False) -> str:
    return json.dumps({"answer": text, "abstained": abstained, "value": value, "claims": list(claims),
                       "recommendations": list(recs)})


IDK = answer("I don't know. The card terms I have don't cover that.", abstained=True)


def name(card: str) -> str:
    return CARDS[card][0]


# ---------------------------------------------------------------- full agent

def agent_plan(q: str) -> list[tuple[str, dict]]:
    it, cards = intent(q), cards_in(q)
    card = cards[0] if cards else None
    if it == "which_card":
        return [("simulate", purchase_in(q))]
    if it == "expiring":
        return [("credit_status", {"card": card or "all", "month": AS_OF.strftime("%Y-%m")})]
    if it == "fee_worth":
        return [("fee_value", {"card": c}) for c in (cards or ["amex_gold", "amex_platinum", "sapphire_reserve"])]
    if it == "credit_remaining" and card:
        return [("credit_status", {"card": card, "month": AS_OF.strftime("%Y-%m")})]
    if card and it in ("annual_fee", "ftf"):
        return [("get_rules", {"card": card, "category": "fees"})]
    if card and it in ("enrollment", "reset", "credit_amount", "merchants"):
        return [("get_rules", {"card": card, "category": "credits"})]
    if card and it in ("cap", "multiplier", "channel_check") and category_in(q):
        return [("get_rules", {"card": card, "category": category_in(q)})]
    return [("search_terms", {"card": card, "query": q})]


def agent_compose(q: str, results: list[tuple[str, dict, object]]) -> str | list[tuple[str, dict]]:
    it = intent(q)
    name_, args, out = results[-1]
    if name_ == "search_terms":
        return _from_search(q, out)
    if isinstance(out, dict) and "error" in out:
        return IDK
    if name_ == "simulate":
        return _which_card(out)
    if name_ == "credit_status":
        return _expiring(q, out) if it == "expiring" else _remaining(q, out)
    if name_ == "fee_value":
        return _fee_worth([r[2] for r in results])
    fact = _fact(q, it, out)
    if fact is None:  # structured rules don't answer it: fall back to clause search once
        if not any(r[0] == "search_terms" for r in results):
            return [("search_terms", {"card": args.get("card"), "query": q})]
        return IDK
    return fact


def _from_search(q: str, hits: list[dict]) -> str:
    if not hits:
        return IDK
    top = hits[0]
    if coverage(q, top["text"], IGNORE) < 0.6:
        return IDK
    body = top["text"].split(" — ", 1)[-1]
    m = re.search(r"\$?" + NUM, body)
    val = float(m[1].replace(",", "")) if m else None
    c = {"clause_id": top["clause_id"], "start": top["start"], "end": top["end"]}
    return answer(f"Per {top['clause_id']}: {body}", val, [{"text": body, "citations": [c]}])


def _which_card(sim: dict) -> str:
    ranked = sim["ranked"]
    recs, claims = [], []
    for o in ranked:
        why = f"{o['multiplier']:g}x -> {o['points']:g} points, ${o['expected_value']:.2f} expected value"
        if o["notes"]:
            why += "; " + "; ".join(o["notes"])
        recs.append({"card": o["card"], "expected_value": o["expected_value"], "points": o["points"],
                     "rationale": why, "citations": o["citations"]})
        if o["citations"]:
            claims.append({"text": f"{name(o['card'])}: {why}", "citations": o["citations"]})
    top = ranked[0]
    text = (f"Use {name(top['card'])}: {top['multiplier']:g}x on this ${sim['amount']:.2f} "
            f"{sim['merchant_category']} purchase, about ${top['expected_value']:.2f} back. "
            + "Ranking: " + ", ".join(f"{name(o['card'])} ${o['expected_value']:.2f}" for o in ranked) + ".")
    return answer(text, top["card"], claims, recs)


def _horizon(q: str) -> int:
    l = q.lower()
    m = re.search(r"next (\d+) days", l)
    if m:
        return int(m[1])
    if "this month" in l or "end of the month" in l:
        return (date(AS_OF.year + (AS_OF.month == 12), AS_OF.month % 12 + 1, 1) - AS_OF).days - 1
    if "this year" in l or "end of the year" in l:
        return (date(AS_OF.year, 12, 31) - AS_OF).days
    return 30


def _expiring(q: str, status: list[dict]) -> str:
    n = _horizon(q)
    hits = sorted((s for s in status if s["remaining"] > 0 and s["days_left"] <= n),
                  key=lambda s: (s["days_left"], -s["remaining"]))
    if not hits:
        return answer(f"No credits with a remaining balance expire in the next {n} days.", [], [])
    claims = [{"text": f"{name(s['card'])} {s['name']}: ${s['remaining']:.2f} of ${s['amount']:.0f} left, "
                       f"expires {s['window_end']} ({s['days_left']} days)", "citations": s["citations"]}
              for s in hits]
    total = sum(s["remaining"] for s in hits)
    text = f"{len(hits)} credits worth ${total:.2f} expire in the next {n} days: " + "; ".join(c["text"] for c in claims)
    return answer(text, sorted(s["credit_id"] for s in hits), claims)


def _remaining(q: str, status: list[dict]) -> str:
    s = match_credit(q, status, key=lambda c: c["name"])
    if s is None:
        return IDK
    text = (f"{name(s['card'])} {s['name']}: ${s['used']:.2f} used, ${s['remaining']:.2f} left of "
            f"${s['amount']:.0f} for {s['window_start']} to {s['window_end']}.")
    return answer(text, s["remaining"], [{"text": text, "citations": s["citations"]}])


def _fee_worth(vals: list[dict]) -> str:
    claims, parts = [], []
    for v in vals:
        if "error" in v:
            continue
        verdict = "worth it" if v["worth_it"] else "not worth it"
        t = (f"{name(v['card'])}: ${v['credits_used']:.2f} in credits used + ${v['extra_rewards']:.2f} extra rewards "
             f"vs {name(v['baseline_card'])} - ${v['annual_fee']:.0f} fee = ${v['net_value']:.2f} net ({verdict}, "
             f"{v['window_start']} to {v['window_end']})")
        parts.append(t)
        claims.append({"text": t, "citations": v["citations"]})
    if not claims:
        return IDK
    value = vals[0]["worth_it"] if len(vals) == 1 else sorted(v["card"] for v in vals if v.get("worth_it"))
    return answer(" | ".join(parts) + ". Credits counted at face value; points at 1 cent.", value, claims)


def _fact(q: str, it: str, rules: dict) -> str | None:
    card = rules["card"]
    if it in ("annual_fee", "ftf"):
        f = rules["fees"].get("annual_fee" if it == "annual_fee" else "foreign_transaction_fee_pct")
        if not f:
            return None
        text = (f"The {name(card)} annual fee is ${f['value']:,.0f}." if it == "annual_fee" else
                f"The {name(card)} foreign transaction fee is {f['value']:g}%.")
        return answer(text, f["value"], [{"text": text, "citations": [cite(f)]}])
    if it in ("enrollment", "reset", "credit_amount", "merchants"):
        c = match_credit(q, rules["credits"])
        if c is None:
            return None
        field = {"enrollment": "enrollment_required", "reset": "reset", "credit_amount": "amount",
                 "merchants": "eligible_merchants"}[it]
        f = c.get(field)
        if not f:
            return None
        if it == "merchants":
            vals = [m["value"] for m in f]
            text = f"{name(card)} {c['name']['value']} eligible merchants: {', '.join(vals)}."
            return answer(text, vals, [{"text": text, "citations": [cite(m) for m in f]}])
        label = f"{name(card)} {c['name']['value']}"
        if it == "enrollment":
            text = f"Enrollment {'is' if f['value'] else 'is not'} required for the {label}."
        elif it == "reset":
            text = f"The {label} resets on a {str(f['value']).replace('_', ' ')} basis."
        else:
            text = f"The {label} is up to ${f['value']:g} per {c['period']['value'].replace('_', '-')} period."
        return answer(text, f["value"], [{"text": text, "citations": [cite(f)]}])
    rules_ = rules.get("earning_rules") or []
    if not rules_:
        return None
    r = rules_[0]
    flag = f" (note: validator flagged this rule: {'; '.join(r['validator_flags'])})" if r.get("validator_flags") else ""
    if it == "cap":
        if not r.get("cap"):
            return None
        cap = r["cap"]["value"]
        text = (f"{name(card)} earns {r['multiplier']['value']:g}x on up to ${cap['amount']:,.0f} per "
                f"{cap['period'].replace('_', ' ')}, then {cap['after_multiplier']:g}x.")
        return answer(text, cap["amount"], [{"text": text, "citations": [cite(r["cap"])]}])
    if it == "channel_check":
        m = re.search(r"(\d+(?:\.\d+)?)(?:x| points)", q.lower())
        asked = float(m[1])
        l = q.lower()
        ch = ("hotel_direct" if "directly with the hotel" in l or ("directly" in l and "hotel" in l) else
              "airline_direct" if "directly" in l else "amex_travel" if "amex travel" in l else
              "chase_travel" if "chase travel" in l else None)
        allowed = (r.get("booking_channel") or {}).get("value")
        ok = r["multiplier"]["value"] >= asked and (allowed is None or ch in allowed)
        cites = [cite(r["multiplier"])] + ([cite(r["booking_channel"])] if r.get("booking_channel") else [])
        text = (f"{'Yes' if ok else 'No'}: {name(card)} earns {r['multiplier']['value']:g}x on {r['category']['value']}"
                + (f" booked via {', '.join(allowed)}" if allowed else "") + "." + flag)
        return answer(text, ok, [{"text": text, "citations": cites}])
    text = f"{name(card)} earns {r['multiplier']['value']:g} points per dollar on {r['category']['value']}." + flag
    return answer(text, r["multiplier"]["value"], [{"text": text, "citations": [cite(r["multiplier"])]}])


# ---------------------------------------------------------------- RAG-only reader

def rag_answer(q: str, clauses: list[dict]) -> str:
    it = intent(q)
    if it in DATA_INTENTS:
        return answer("I can't tell from the terms alone; I would need your transactions.", abstained=True)
    cards = cards_in(q)
    pool = [c for c in clauses if not cards or c["card"] in cards] or clauses

    def cited(text, value, c):
        return answer(text, value, [{"text": text, "citations": [{"clause_id": c["clause_id"]}]}])

    def first(pattern, text):
        m = re.search(pattern, text, re.I)
        return float(m[1].replace(",", "")) if m else None

    if it == "which_card":
        cat = category_in(q) or "other"
        best = None
        for c in clauses:
            if TITLE_CATEGORIES.get(c["title"].lower()) == cat:
                v = first(NUM + r" points? per dollar", c["text"])
                if v is not None and (best is None or v > best[0]):
                    best = (v, c)
        if not best:
            return IDK
        return cited(f"{name(best[1]['card'])} earns {best[0]:g} points per dollar ({best[1]['clause_id']}).",
                     best[1]["card"], best[1])
    if it in ("annual_fee", "ftf"):
        title = "annual fee" if it == "annual_fee" else "foreign transactions"
        c = next((c for c in pool if c["title"].lower() == title), None)
        if not c:
            return IDK
        v = first(r"\$" + NUM, c["text"]) if it == "annual_fee" else (
            0.0 if "no foreign transaction fee" in c["text"].lower() else first(NUM + "%", c["text"]))
        return cited(c["text"], v, c)
    if it in ("multiplier", "cap", "channel_check"):
        cat = category_in(q)
        c = next((c for c in pool if TITLE_CATEGORIES.get(c["title"].lower()) == cat), None)
        if not c:
            return IDK
        if it == "multiplier":
            return cited(c["text"], first(NUM + r" points? per dollar", c["text"]), c)
        if it == "cap":
            v = first(r"up to \$" + NUM + " in purchases", c["text"])
            return cited(c["text"], v, c) if v else IDK
        l = q.lower()  # naive reader: the channel phrase is mentioned, so it must qualify
        phrase = "directly with the hotel" if "hotel" in l else "directly with"
        return cited(c["text"], phrase in c["text"].lower(), c)
    if it in ("enrollment", "reset", "credit_amount", "merchants"):
        creds = [c for c in pool if "credit" in c["title"].lower() or "uber cash" in c["title"].lower()]
        c = match_credit(q, creds, key=lambda c: c["title"])
        if not c:
            return IDK
        t = c["text"].lower()
        if it == "enrollment":
            v = "enrollment is required" in t and not re.search(r"no (separate )?enrollment", t)
        elif it == "reset":
            v = ("calendar_month" if "calendar month" in t else "calendar_half" if "january through june" in t
                 else "cardmember_year" if "anniversary" in t else "calendar_year")
        elif it == "credit_amount":
            v = first(r"up to \$" + NUM, c["text"])
        else:
            m = re.search(r"Eligible merchants: (.+?)\.(?=\s|$)", c["text"])
            v = [x.strip() for x in re.split(r",\s*|\s+and\s+", m[1])] if m else None
        return cited(c["text"], v, c)
    top = pool[0] if pool else None
    if not top or coverage(q, top["text"], IGNORE) < 0.6:
        return IDK
    return cited(top["text"], first(r"\$?" + NUM, top["text"].split(" — ", 1)[-1]), top)


# ---------------------------------------------------------------- plain (closed-book) baseline

# A hand-written "parametric memory": some values match the synthetic terms, some are stale on purpose.
BELIEF_FEES = {"amex_gold": 250, "amex_platinum": 695, "sapphire_reserve": 550, "freedom_unlimited": 0}
BELIEF_FTF = {"amex_gold": 0, "amex_platinum": 0, "sapphire_reserve": 0, "freedom_unlimited": 3}
BELIEF_MULT = {("amex_gold", "dining"): 4, ("amex_gold", "us_supermarkets"): 4, ("amex_gold", "flights"): 3,
               ("amex_platinum", "flights"): 5, ("amex_platinum", "hotels"): 5, ("sapphire_reserve", "dining"): 3,
               ("sapphire_reserve", "flights"): 10, ("sapphire_reserve", "hotels"): 10,
               ("freedom_unlimited", "dining"): 3, ("freedom_unlimited", "drugstores"): 3}
BELIEF_CREDITS = {"dining": 10, "uber": 15, "resy": 50, "saks": 50, "entertainment": 20, "airline": 200,
                  "travel": 300, "dunkin": 7, "doordash": 5, "hotel": 200, "clear": 199}


def plain_answer(q: str) -> str:
    it, cards = intent(q), cards_in(q)
    card = cards[0] if cards else "amex_gold"
    l = q.lower()
    if it == "which_card":
        cat = category_in(q) or "other"
        best = max(CARDS, key=lambda c: BELIEF_MULT.get((c, cat), 1.5 if c == "freedom_unlimited" else 1))
        return answer(f"{name(best)} is the best card for that.", best)
    if it == "annual_fee":
        return answer(f"The annual fee is ${BELIEF_FEES[card]}.", float(BELIEF_FEES[card]))
    if it == "ftf":
        return answer(f"{BELIEF_FTF[card]}% foreign transaction fee.", float(BELIEF_FTF[card]))
    if it == "multiplier":
        v = BELIEF_MULT.get((card, category_in(q)), 1)
        return answer(f"{v}x.", float(v))
    if it == "cap":
        return answer("Up to $25,000 per year.", 25000.0)
    if it == "channel_check":
        return answer("Yes.", True)
    if it == "enrollment":
        return answer("Yes, enrollment is required.", True)
    if it == "reset":
        return answer("It resets every calendar year.", "calendar_year")
    if it in ("credit_amount", "credit_remaining"):
        v = next((a for k, a in BELIEF_CREDITS.items() if k in l), 100)
        return answer(f"${v}.", float(v))
    if it == "expiring":
        return answer("Your monthly credits expire at the end of the month.",
                      sorted(f"{c}.{n}" for c, n in [("amex_gold", "dining_credit"), ("amex_gold", "uber_cash"),
                                                      ("amex_platinum", "uber_cash")]))
    if it == "fee_worth":
        return answer("Yes, if you use the credits it pays for itself.", True)
    m = re.search(r"\$?" + NUM, q)
    return answer("Yes, that is included with the card.", True if not m else float(m[1].replace(",", "")))
