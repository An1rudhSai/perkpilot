import json

from perkpilot.agent import ground, run_agent
from perkpilot.llm import ChatResult, ToolCall
from perkpilot.offline import ScriptedLLM
from perkpilot.schema import Answer, Citation, Claim


def ask(q, toolbox):
    return run_agent(q, ScriptedLLM(), toolbox).answer


def test_which_card_ranks_and_cites(toolbox):
    a = ask("Which card should I use for a $80 dinner?", toolbox)
    assert a.value == "amex_gold" and not a.abstained
    assert a.recommendations[0].card == "amex_gold"
    assert a.claims[0].citations[0].clause_id == "AG-3"


def test_point_worth_is_a_search_fallback_not_fee_math(toolbox):
    a = ask("How much is a Freedom Unlimited point worth when redeemed for cash back?", toolbox)
    assert a.value == 1 and a.claims[0].citations[0].clause_id == "FU-7"


def test_abstains_when_terms_are_silent(toolbox):
    a = ask("Does the Amex Gold include cell phone protection?", toolbox)
    assert a.abstained and a.claims == []


def test_every_claim_is_cited(toolbox):
    for q in ["Which credits expire this month?", "Is the Amex Gold annual fee worth it?",
              "How much is left on my Amex Gold Resy credit?"]:
        a = ask(q, toolbox)
        assert a.claims and all(c.citations for c in a.claims)
        assert {c.clause_id for cl in a.claims for c in cl.citations} <= toolbox.seen_clauses


def test_ground_drops_citations_no_tool_returned():
    ans = Answer(answer="x", claims=[Claim(text="made up", citations=[Citation(clause_id="AG-99")])])
    out = ground(ans, {"AG-1"})
    assert out.abstained and out.dropped_claims == ["made up"]


class OneShotLLM:
    """Simulates a model that skips tools and cites a clause it never retrieved."""
    name = "fake"

    def chat(self, messages, tools=None, json_mode=False):
        return ChatResult(content=json.dumps({"answer": "Gold earns 10x", "value": 10,
                                              "claims": [{"text": "Gold earns 10x", "citations": [{"clause_id": "AG-3"}]}]}))


def test_uncited_or_unretrieved_claims_become_abstention(toolbox):
    run = run_agent("How many points does Gold earn at restaurants?", OneShotLLM(), toolbox)
    assert run.answer.abstained


class BadJsonThenToolLLM:
    name = "fake2"

    def __init__(self):
        self.n = 0

    def chat(self, messages, tools=None, json_mode=False):
        self.n += 1
        if self.n == 1:
            return ChatResult(tool_calls=[ToolCall(id="c1", name="get_rules",
                                                   arguments={"card": "amex_gold", "category": "fees"})])
        if self.n == 2:
            return ChatResult(content="not json")
        return ChatResult(content=json.dumps({"answer": "$325", "value": 325, "claims": [
            {"text": "fee is $325", "citations": [{"clause_id": "AG-1"}]}]}))


def test_loop_repairs_bad_json_once(toolbox):
    run = run_agent("What is the Gold annual fee?", BadJsonThenToolLLM(), toolbox)
    assert run.answer.value == 325 and not run.answer.abstained and run.steps == 3
