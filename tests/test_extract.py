import json

from perkpilot.extract import extract_card, span_text
from perkpilot.llm import ChatResult
from perkpilot.offline import ScriptedLLM


def test_gold_extraction_has_spans_for_every_field():
    res = extract_card("amex_gold", ScriptedLLM())
    t = res.terms
    assert t.fees.annual_fee.value == 325
    assert t.fees.foreign_transaction_fee_pct.value == 0
    dining = next(r for r in t.earning_rules if r.category.value == "dining")
    assert dining.multiplier.value == 4
    assert dining.cap.value.amount == 50000
    assert {e.value for e in dining.exclusions} >= {"mobile_card_reader", "third_party_processor"}
    # every span points at text that actually contains the quoted value
    assert span_text("amex_gold", dining.multiplier.span.start, dining.multiplier.span.end) == "4 points per dollar"
    dc = next(c for c in t.credits if c.id == "amex_gold.dining_credit")
    assert dc.amount.value == 10 and dc.period.value == "monthly" and dc.enrollment_required.value is True
    assert [m.value for m in dc.eligible_merchants][:2] == ["Grubhub", "The Cheesecake Factory"]


class LyingLLM(ScriptedLLM):
    """Returns a multiplier whose quote does not exist in the clause."""

    def _task_extract_clause(self, messages, tools):
        out = json.loads(super()._task_extract_clause(messages, tools).content)
        for r in out["earning_rules"]:
            r["multiplier"]["quote"] = "10 points per dollar on everything"
        for c in out["credits"]:
            c["eligible_merchants"].append({"value": "Amazon", "quote": "Amazon"})
        return ChatResult(content=json.dumps(out))


def test_fields_without_supporting_span_are_rejected():
    res = extract_card("amex_gold", LyingLLM())
    assert res.terms.earning_rules == []  # multiplier is required, so every rule is dropped
    assert any("required field 'multiplier'" in r.reason for r in res.rejections)
    assert all("Amazon" not in [m.value for m in c.eligible_merchants] for c in res.terms.credits)
    assert any(r.quote == "Amazon" for r in res.rejections)
