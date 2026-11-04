"""Pure-function tests for simulate(). Rules come from the offline extractor, so these also
pin the extracted values the engine relies on."""
import pytest

from perkpilot.engine import simulate
from perkpilot.extract import extract_card
from perkpilot.offline import ScriptedLLM
from perkpilot.terms import available_cards


@pytest.fixture(scope="module")
def rules():
    return {c: extract_card(c, ScriptedLLM()).terms for c in available_cards()}


def by_card(res):
    return {o.card: o for o in res.ranked}


def test_restaurant_goes_to_gold(rules):
    res = simulate("Joe's Bistro", 100, "dining", "card_present", rules=rules)
    assert res.ranked[0].card == "amex_gold"
    assert res.ranked[0].points == 400
    assert res.ranked[0].citations[0].clause_id == "AG-3"


def test_mobile_card_reader_exclusion_drops_gold_to_1x(rules):
    res = simulate("Taco Truck", 20, "dining", "mobile_card_reader", rules=rules)
    gold = by_card(res)["amex_gold"]
    assert gold.multiplier == 1
    assert any("excluded (mobile_card_reader)" in n for n in gold.notes)
    assert res.ranked[0].card in {"sapphire_reserve", "freedom_unlimited"}
    assert res.ranked[0].multiplier == 3


def test_warehouse_club_is_not_a_supermarket(rules):
    res = simulate("Costco Wholesale", 200, "supermarkets", "card_present", rules=rules)
    assert by_card(res)["amex_gold"].multiplier == 1
    assert res.ranked[0].card == "freedom_unlimited"  # 1.5x on everything else


def test_us_supermarket_cap(rules):
    res = simulate("Whole Foods", 1000, "supermarkets", rules=rules,
                   spend_to_date={"amex_gold.us_supermarkets": 24_500})
    gold = by_card(res)["amex_gold"]
    assert gold.points == 500 * 4 + 500 * 1
    assert any(c.clause_id == "AG-4" for c in gold.citations)


def test_cap_already_exhausted(rules):
    res = simulate("Whole Foods", 100, "supermarkets", rules=rules,
                   spend_to_date={"amex_gold.us_supermarkets": 30_000})
    assert by_card(res)["amex_gold"].points == 100


def test_foreign_supermarket_is_not_us_supermarket(rules):
    res = simulate("Tesco", 50, "supermarkets", foreign=True, rules=rules)
    assert by_card(res)["amex_gold"].multiplier == 1


def test_booking_channel(rules):
    direct = by_card(simulate("Delta Air Lines", 500, "flights", "online", rules=rules))
    assert direct["amex_platinum"].multiplier == 5
    assert direct["sapphire_reserve"].multiplier == 4
    portal = by_card(simulate("Chase Travel - United", 500, "flights", "online", rules=rules))
    assert portal["sapphire_reserve"].multiplier == 8
    assert portal["amex_platinum"].multiplier == 1  # not booked direct or via Amex Travel
    otas = by_card(simulate("Expedia", 500, "hotels", "online", rules=rules))
    assert otas["sapphire_reserve"].multiplier == 1 and otas["amex_platinum"].multiplier == 1


def test_foreign_transaction_fee(rules):
    res = simulate("Le Bistro Paris", 100, "dining", foreign=True, rules=rules)
    fu = by_card(res)["freedom_unlimited"]
    assert fu.ftf_cost == 3.0 and fu.expected_value == 0.0
    assert any(c.clause_id == "FU-2" for c in fu.citations)
    assert res.ranked[0].card == "amex_gold"


def test_gift_cards_earn_nothing_on_amex(rules):
    res = by_card(simulate("Gift Card Mall", 100, "gift_card", "online", rules=rules))
    assert res["amex_gold"].points == 0 and res["amex_platinum"].points == 0
    assert res["freedom_unlimited"].points == 150  # FU's terms don't exclude gift cards


def test_point_values_change_ranking(rules):
    res = simulate("Corner Store", 100, "other", rules=rules,
                   point_values={"amex_platinum": 2.0, "freedom_unlimited": 1.0})
    assert res.ranked[0].card == "amex_platinum"  # 1x at 2.0cpp beats 1.5x at 1.0cpp


def test_is_pure_and_deterministic(rules):
    a = simulate("Joe's Bistro", 42.5, "restaurants", rules=rules)
    b = simulate("Joe's Bistro", 42.5, "restaurants", rules=rules)
    assert a == b and a.merchant_category == "dining"


def test_validator_flag_surfaces_in_notes(rules):
    flagged = {"amex_platinum.hotels": ["booking channel hotel_direct comes from a sentence with a different rate"]}
    res = by_card(simulate("Marriott", 300, "hotels", rules=rules, flagged=flagged))
    assert any("flagged by validator" in n for n in res["amex_platinum"].notes)


def test_negative_amount_rejected(rules):
    with pytest.raises(ValueError):
        simulate("x", -1, "dining", rules=rules)
