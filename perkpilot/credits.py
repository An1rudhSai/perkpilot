"""Credit tracking and annual-fee value, computed from extracted rules + transactions. Pure."""
from __future__ import annotations

import calendar
import re
from collections import defaultdict
from datetime import date, timedelta

from pydantic import BaseModel, Field

from .config import BASELINE_CARD
from .engine import evaluate_card
from .schema import CardTerms, Citation, Credit, Transaction

# Account-open dates for the synthetic cardholder (cardmember-year credits reset on the anniversary).
CARD_OPEN_DATES = {"amex_gold": date(2022, 6, 1), "amex_platinum": date(2023, 1, 10),
                   "sapphire_reserve": date(2023, 3, 15), "freedom_unlimited": date(2021, 9, 1)}
# Eligible-merchant phrases that describe a kind of purchase rather than a merchant name.
CATEGORY_GROUPS = {"travel purchases": {"flights", "hotels", "travel", "rental_car", "rideshare"},
                   "selected airline": {"airline_fee"}}
_SUFFIX = re.compile(r"\s+(in the united states|locations|stores|restaurants)$")


class CreditStatus(BaseModel):
    credit_id: str
    card: str
    name: str
    amount: float
    period: str
    window_start: date
    window_end: date  # the credit expires (unused balance is lost) at the end of this day
    used: float
    remaining: float
    days_left: int
    enrollment_required: bool | None
    citations: list[Citation] = Field(default_factory=list)


class FeeValue(BaseModel):
    card: str
    window_start: date
    window_end: date
    annual_fee: float
    credits_used: float
    credits_by_name: dict[str, float]
    rewards_value: float
    baseline_card: str
    baseline_rewards_value: float
    extra_rewards: float
    net_value: float
    worth_it: bool
    citations: list[Citation] = Field(default_factory=list)
    caveats: list[str] = Field(default_factory=list)


def _reset_kind(c: Credit) -> str:
    if c.reset:
        return c.reset.value
    return {"monthly": "calendar_month", "semi_annual": "calendar_half", "annual": "calendar_year"}[c.period.value]


def window(c: Credit, d: date) -> tuple[date, date]:
    kind = _reset_kind(c)
    if kind == "calendar_month":
        return d.replace(day=1), d.replace(day=calendar.monthrange(d.year, d.month)[1])
    if kind == "calendar_half":
        return (date(d.year, 1, 1), date(d.year, 6, 30)) if d.month <= 6 else (date(d.year, 7, 1), date(d.year, 12, 31))
    if kind == "calendar_year":
        return date(d.year, 1, 1), date(d.year, 12, 31)
    opened = CARD_OPEN_DATES.get(c.card, date(d.year, 1, 1))
    start = opened.replace(year=d.year)
    if start > d:
        start = start.replace(year=d.year - 1)
    return start, start.replace(year=start.year + 1) - timedelta(days=1)


def matches(c: Credit, t: Transaction) -> bool:
    if t.card != c.card:
        return False
    merchant = t.merchant.lower()
    for em in c.eligible_merchants:
        phrase = em.value.lower()
        for key, cats in CATEGORY_GROUPS.items():
            if key in phrase:
                if t.category in cats:
                    return True
                break
        else:
            phrase = phrase.removeprefix("prepaid bookings with ")
            while _SUFFIX.search(phrase):
                phrase = _SUFFIX.sub("", phrase)
            if phrase in merchant or merchant in phrase:
                return True
    return False


def _cites(c: Credit) -> list[Citation]:
    out = [Citation(clause_id=c.clause_id, start=c.amount.span.start, end=c.amount.span.end)]
    if c.reset:
        out.append(Citation(clause_id=c.clause_id, start=c.reset.span.start, end=c.reset.span.end))
    return out


def credit_status(terms: CardTerms, txns: list[Transaction], ref: date, as_of: date) -> list[CreditStatus]:
    """Status of each credit in the period containing `ref`, counting purchases up to `as_of`."""
    out = []
    for c in terms.credits:
        ws, we = window(c, ref)
        used = sum(t.amount for t in txns if ws <= t.date <= min(we, as_of) and matches(c, t))
        used = round(min(used, c.amount.value), 2)
        out.append(CreditStatus(
            credit_id=c.id, card=c.card, name=c.name.value, amount=c.amount.value, period=c.period.value,
            window_start=ws, window_end=we, used=used, remaining=round(c.amount.value - used, 2),
            days_left=(we - as_of).days, enrollment_required=c.enrollment_required.value if c.enrollment_required else None,
            citations=_cites(c)))
    return out


def card_rewards(terms: CardTerms, txns: list[Transaction], cpp: float = 1.0) -> float:
    """Total expected value of `txns` if they were charged to `terms.card`, tracking caps."""
    spend: dict[tuple[str, int], float] = defaultdict(float)
    total = 0.0
    for t in sorted(txns, key=lambda t: (t.date, t.id)):
        year_spend = {rid: v for (rid, y), v in spend.items() if y == t.date.year}
        o = evaluate_card(terms, t.merchant, t.amount, t.category, t.payment_method, t.foreign, year_spend, cpp)
        if o.rule_id:
            spend[(o.rule_id, t.date.year)] += t.amount
        total += o.expected_value
    return round(total, 2)


def fee_value(terms: CardTerms, baseline: CardTerms, txns: list[Transaction], as_of: date,
              cpp: float = 1.0, baseline_cpp: float = 1.0) -> FeeValue:
    """Credits actually used + extra rewards vs. the no-fee baseline, minus the annual fee, trailing 365 days."""
    start = as_of - timedelta(days=364)
    mine = [t for t in txns if t.card == terms.card and start <= t.date <= as_of]
    used_by_period: dict[tuple[str, date], float] = defaultdict(float)
    for t in mine:
        for c in terms.credits:
            if matches(c, t):
                used_by_period[(c.id, window(c, t.date)[0])] += t.amount
    by_name: dict[str, float] = defaultdict(float)
    cites: list[Citation] = []
    for c in terms.credits:
        got = sum(min(v, c.amount.value) for (cid, _), v in used_by_period.items() if cid == c.id)
        by_name[c.name.value] = round(got, 2)
        if got:
            cites += _cites(c)[:1]
    credits_used = round(sum(by_name.values()), 2)
    rewards = card_rewards(terms, mine, cpp)
    base = card_rewards(baseline, mine, baseline_cpp)
    fee = terms.fees.annual_fee.value if terms.fees.annual_fee else 0.0
    if terms.fees.annual_fee:
        s = terms.fees.annual_fee.span
        cites.insert(0, Citation(clause_id=s.clause_id, start=s.start, end=s.end))
    net = round(credits_used + (rewards - base) - fee, 2)
    caveats = ["Credits are valued at face value; your own valuation may be lower.",
               f"Points valued at {cpp:g} cents each (an assumption, not part of the terms)."]
    return FeeValue(card=terms.card, window_start=start, window_end=as_of, annual_fee=fee, credits_used=credits_used,
                    credits_by_name=dict(by_name), rewards_value=rewards, baseline_card=baseline.card,
                    baseline_rewards_value=base, extra_rewards=round(rewards - base, 2), net_value=net,
                    worth_it=net > 0 or (fee == 0 and net >= 0), citations=cites, caveats=caveats)


__all__ = ["credit_status", "fee_value", "window", "matches", "BASELINE_CARD"]
