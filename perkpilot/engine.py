"""Pure rule engine. No I/O, no LLM: rules in, ranked outcomes out.

`simulate` answers "what would each card earn on this purchase?", applying category matching,
booking-channel requirements, exclusions, annual caps and foreign transaction fees. Every
outcome cites the clauses (and spans) of the rules that produced it.
"""
from __future__ import annotations

from collections.abc import Mapping

from pydantic import BaseModel, Field

from .schema import CardTerms, Citation, EarningRule, Sourced

TRAVEL_CATEGORIES = {"flights", "hotels", "travel", "rental_car"}
CATEGORY_ALIASES = {
    "restaurants": "dining", "restaurant": "dining", "food": "dining",
    "supermarkets": "us_supermarkets", "groceries": "us_supermarkets", "grocery": "us_supermarkets",
    "airfare": "flights", "flight": "flights", "hotel": "hotels", "pharmacy": "drugstores", "drugstore": "drugstores",
}
# Merchant names that imply a merchant type (used for exclusion checks) or a booking channel.
KNOWN_MERCHANT_TYPES = {
    "costco": "warehouse_club", "sam's club": "warehouse_club", "bj's": "warehouse_club",
    "walmart supercenter": "superstore", "target": "superstore", "7-eleven": "convenience_store",
    "hellofresh": "meal_kit", "blue apron": "meal_kit",
}
CHANNEL_MERCHANTS = {"amex travel": "amex_travel", "chase travel": "chase_travel", "the edit": "chase_travel",
                     "expedia": "third_party_site", "booking.com": "third_party_site"}


class CardOutcome(BaseModel):
    card: str
    rule_id: str | None
    multiplier: float
    points: float
    rewards_value: float  # dollars, at the assumed cents-per-point
    ftf_cost: float
    expected_value: float  # rewards_value - ftf_cost
    citations: list[Citation] = Field(default_factory=list)
    notes: list[str] = Field(default_factory=list)


class SimulationResult(BaseModel):
    merchant: str
    amount: float
    merchant_category: str
    payment_method: str
    foreign: bool
    channel: str
    ranked: list[CardOutcome]


def normalize_category(cat: str) -> str:
    c = cat.strip().lower().replace(" ", "_")
    return CATEGORY_ALIASES.get(c, c)


def booking_channel(merchant: str, category: str) -> str:
    m = merchant.lower()
    for key, ch in CHANNEL_MERCHANTS.items():
        if key in m:
            return ch
    if category == "flights":
        return "airline_direct"
    if category == "hotels":
        return "hotel_direct"
    return "direct"


def merchant_type(merchant: str) -> str | None:
    m = merchant.lower()
    return next((t for k, t in KNOWN_MERCHANT_TYPES.items() if k in m), None)


def _cite(s: Sourced | None) -> Citation | None:
    return Citation(clause_id=s.span.clause_id, start=s.span.start, end=s.span.end) if s else None


def _category_matches(rule_cat: str, cat: str, foreign: bool) -> bool:
    if rule_cat == "travel":
        return cat in TRAVEL_CATEGORIES
    if rule_cat == "us_supermarkets":
        return cat == "us_supermarkets" and not foreign
    return rule_cat == cat


def _exclusion_hit(rule: EarningRule, signals: set[str]) -> Sourced[str] | None:
    return next((e for e in rule.exclusions if e.value in signals), None)


def evaluate_card(terms: CardTerms, merchant: str, amount: float, category: str, payment_method: str,
                  foreign: bool = False, spend_to_date: Mapping[str, float] | None = None,
                  cents_per_point: float = 1.0, flagged: Mapping[str, list[str]] | None = None) -> CardOutcome:
    """Earn for one card. `spend_to_date` maps rule_id -> spend already counted toward that rule's cap."""
    cat = normalize_category(category)
    channel = booking_channel(merchant, cat)
    signals = {payment_method, cat} | ({merchant_type(merchant)} - {None})
    notes: list[str] = []
    citations: list[Citation] = []

    candidates = []
    for r in terms.earning_rules:
        if r.category.value == "other" or not _category_matches(r.category.value, cat, foreign):
            continue
        if r.booking_channel and channel not in r.booking_channel.value:
            notes.append(f"{r.id}: needs booking via {'/'.join(r.booking_channel.value)}, this is {channel}")
            citations.append(_cite(r.booking_channel))
            continue
        hit = _exclusion_hit(r, signals)
        if hit:
            notes.append(f"{r.id}: excluded ({hit.value})")
            citations.append(_cite(hit))
            continue
        candidates.append(r)

    rule = max(candidates, key=lambda r: r.multiplier.value, default=None)
    if rule is None:
        rule = next((r for r in terms.earning_rules if r.category.value == "other"), None)
        if rule is not None and (hit := _exclusion_hit(rule, signals)):
            notes.append(f"{rule.id}: does not earn points ({hit.value})")
            citations.append(_cite(hit))
            return _outcome(terms, None, 0.0, 0.0, amount, foreign, cents_per_point, citations, notes)
    if rule is None:
        notes.append("no earning rule applies")
        return _outcome(terms, None, 0.0, 0.0, amount, foreign, cents_per_point, citations, notes)

    mult = rule.multiplier.value
    citations.insert(0, _cite(rule.multiplier))
    points = amount * mult
    if rule.cap:
        cap = rule.cap.value
        already = (spend_to_date or {}).get(rule.id, 0.0)
        room = max(cap.amount - already, 0.0)
        if amount > room:
            points = room * mult + (amount - room) * cap.after_multiplier
            notes.append(f"{rule.id}: ${amount - room:,.2f} is over the ${cap.amount:,.0f} cap and earns "
                         f"{cap.after_multiplier:g}x")
            citations.append(_cite(rule.cap))
    if flagged and rule.id in flagged:
        notes.append(f"{rule.id}: flagged by validator for review: {'; '.join(flagged[rule.id])}")
    return _outcome(terms, rule.id, mult, points, amount, foreign, cents_per_point, citations, notes)


def _outcome(terms, rule_id, mult, points, amount, foreign, cpp, citations, notes) -> CardOutcome:
    ftf = 0.0
    if foreign and terms.fees.foreign_transaction_fee_pct:
        pct = terms.fees.foreign_transaction_fee_pct.value
        ftf = round(amount * pct / 100, 2)
        if pct:
            notes.append(f"{pct:g}% foreign transaction fee")
        citations.append(_cite(terms.fees.foreign_transaction_fee_pct))
    value = round(points * cpp / 100, 2)
    return CardOutcome(card=terms.card, rule_id=rule_id, multiplier=mult, points=round(points, 2),
                       rewards_value=value, ftf_cost=ftf, expected_value=round(value - ftf, 2),
                       citations=[c for c in citations if c], notes=notes)


def simulate(merchant: str, amount: float, merchant_category: str, payment_method: str = "card_present", *,
             rules: Mapping[str, CardTerms], foreign: bool = False,
             spend_to_date: Mapping[str, float] | None = None,
             point_values: Mapping[str, float] | None = None,
             flagged: Mapping[str, list[str]] | None = None) -> SimulationResult:
    """Rank every card by expected value for one purchase. Pure: same inputs, same output."""
    if amount < 0:
        raise ValueError("amount must be non-negative")
    cat = normalize_category(merchant_category)
    outcomes = [
        evaluate_card(t, merchant, amount, cat, payment_method, foreign, spend_to_date,
                      (point_values or {}).get(card, 1.0), flagged)
        for card, t in rules.items()
    ]
    order = list(rules)
    outcomes.sort(key=lambda o: (-o.expected_value, order.index(o.card)))
    return SimulationResult(merchant=merchant, amount=amount, merchant_category=cat, payment_method=payment_method,
                            foreign=foreign, channel=booking_channel(merchant, cat), ranked=outcomes)
