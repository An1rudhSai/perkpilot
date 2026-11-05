"""Generator for realistic but entirely synthetic transactions (one fictional cardholder,
four cards, the 12 months ending on AS_OF). Seeded, so the dataset is reproducible."""
from __future__ import annotations

import random
from datetime import date, timedelta

from .config import AS_OF
from .schema import Transaction

START = AS_OF - timedelta(days=364)


def _months(start: date, end: date):
    d = date(start.year, start.month, 1)
    while d <= end:
        yield d
        d = date(d.year + (d.month == 12), d.month % 12 + 1, 1)


class _Gen:
    def __init__(self, seed: int):
        self.rng = random.Random(seed)
        self.out: list[Transaction] = []

    def add(self, d: date, card, merchant, category, lo, hi=None, pm="card_present", foreign=False):
        if d < START or d > AS_OF:
            return
        amt = lo if hi is None else round(self.rng.uniform(lo, hi), 2)
        self.out.append(Transaction(id=f"t{len(self.out):05d}", date=d, card=card, merchant=merchant,
                                    category=category, amount=amt, payment_method=pm, foreign=foreign))

    def day(self, month: date, lo=1, hi=28) -> date:
        return month.replace(day=self.rng.randint(lo, hi))


def generate(seed: int = 7) -> list[Transaction]:
    g = _Gen(seed)
    r = g.rng
    for m in _months(START, AS_OF):
        ym = (m.year, m.month)
        # --- Amex Gold: groceries, restaurants, dining partners
        for w in range(4):
            g.add(m + timedelta(days=w * 7 + r.randint(0, 6)), "amex_gold",
                  r.choice(["Whole Foods Market", "Trader Joe's", "Safeway"]), "us_supermarkets", 45, 160)
        for _ in range(5):
            g.add(g.day(m), "amex_gold", r.choice(["Joe's Bistro", "Sushi Nakamura", "Pho 88", "Luigi's Trattoria"]),
                  "dining", 22, 95)
        g.add(g.day(m), "amex_gold", "Costco Wholesale", "us_supermarkets", 120, 240)  # warehouse club: 1x on Gold
        if ym not in {(2026, 2), (2026, 8)}:
            g.add(g.day(m, 3, 26), "amex_gold", "Grubhub", "dining", 14, 32)
        if ym != (2026, 4):
            g.add(g.day(m, 3, 26), "amex_gold", "Uber Eats", "dining", 12, 28)
        if r.random() < 0.5:
            g.add(g.day(m), "amex_gold", "Dunkin' #4471", "dining", 4.5, 9)
        if m.month in (3, 9):
            g.add(g.day(m), "amex_gold", "Taco Truck (Square reader)", "dining", 11, 18, pm="mobile_card_reader")
        # --- Amex Platinum: Uber rides, streaming, travel
        if ym != (2026, 1):
            g.add(g.day(m, 3, 26), "amex_platinum", "Uber", "rideshare", 18, 42)
        g.add(m.replace(day=2), "amex_platinum", "Disney+", "streaming", 13.99)
        # --- Sapphire Reserve: dining, DoorDash
        for _ in range(2):
            g.add(g.day(m), "sapphire_reserve", r.choice(["Blue Hill Cafe", "Ramen Lab"]), "dining", 30, 80)
        if m.month % 2 == 0:
            g.add(g.day(m, 3, 26), "sapphire_reserve", "DoorDash", "dining", 18, 35)
        # --- Freedom Unlimited: drugstores, gas, shopping
        for _ in range(2):
            g.add(g.day(m), "freedom_unlimited", r.choice(["CVS Pharmacy", "Walgreens"]), "drugstores", 8, 45)
        for _ in range(3):
            g.add(g.day(m), "freedom_unlimited", "Shell", "gas", 35, 65)
        for _ in range(3):
            g.add(g.day(m), "freedom_unlimited", r.choice(["Amazon", "Target", "Best Buy"]), "shopping", 20, 140)

    # one-off and seasonal purchases (fixed so the credit picture is easy to reason about)
    g.add(date(2026, 2, 11), "amex_gold", "Resy - Carbone", "dining", 140.00)
    g.add(date(2026, 9, 19), "amex_gold", "Resy - Via Carota", "dining", 30.00)  # H2 Resy: $20 left
    g.add(date(2026, 2, 6), "amex_platinum", "CLEAR", "travel", 199.00)
    g.add(date(2026, 3, 2), "amex_platinum", "Delta Air Lines", "flights", 486.20, pm="online")
    g.add(date(2026, 3, 9), "amex_platinum", "Delta Air Lines (checked bag)", "airline_fee", 35.00)
    g.add(date(2026, 7, 14), "amex_platinum", "Delta Air Lines (checked bag)", "airline_fee", 35.00)
    g.add(date(2026, 7, 14), "amex_platinum", "Delta Air Lines", "flights", 612.40, pm="online")
    g.add(date(2026, 10, 3), "amex_platinum", "Delta Air Lines (seat fee)", "airline_fee", 35.00)
    g.add(date(2026, 5, 20), "amex_platinum", "Amex Travel - FHR The Langham", "hotels", 655.00, pm="online")
    g.add(date(2026, 4, 12), "amex_platinum", "Saks Fifth Avenue", "shopping", 78.00)  # H1 Saks used; H2 unused
    g.add(date(2026, 4, 2), "sapphire_reserve", "Chase Travel - Hyatt Regency", "hotels", 412.00, pm="online")
    g.add(date(2026, 6, 8), "sapphire_reserve", "United Airlines", "flights", 389.00, pm="online")
    g.add(date(2026, 6, 15), "sapphire_reserve", "Le Petit Zinc, Paris", "dining", 96.50, foreign=True)
    g.add(date(2026, 3, 21), "sapphire_reserve", "Sapphire Exclusive Tables - Le Bernardin", "dining", 182.00)
    g.add(date(2026, 6, 16), "freedom_unlimited", "Pret A Manger, London", "dining", 18.40, foreign=True)
    g.add(date(2026, 8, 30), "freedom_unlimited", "Expedia", "hotels", 260.00, pm="online")
    return sorted(g.out, key=lambda t: (t.date, t.id))
