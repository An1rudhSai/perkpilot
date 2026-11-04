from perkpilot.extract import extract_card
from perkpilot.offline import ScriptedLLM
from perkpilot.terms import available_cards
from perkpilot.validate import validate_terms


def test_all_four_cards_extract_without_rejections():
    assert available_cards() == ["amex_gold", "amex_platinum", "sapphire_reserve", "freedom_unlimited"]
    for card in available_cards():
        res = extract_card(card, ScriptedLLM())
        assert res.terms.fees.annual_fee is not None
        assert res.terms.earning_rules
        assert res.rejections == []


def test_validator_flags_known_disagreements():
    terms = extract_card("amex_platinum", ScriptedLLM()).terms
    bad = {v.rule_id: v.issues for v in validate_terms(terms, ScriptedLLM()) if not v.agrees}
    # AP-4: "Hotel stays booked directly with the hotel earn 1 point" was misread as a 5x channel
    assert "amex_platinum.hotels" in bad
    # AP-7: the extra $20 Uber Cash in December is not captured by a single monthly amount
    assert any("$20" in i for i in bad["amex_platinum.uber_cash"])
