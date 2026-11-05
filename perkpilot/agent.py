"""The agent loop, written by hand: model -> tool calls -> tool results -> ... -> final JSON answer.

After the loop, `ground()` enforces the citation contract: every claim must cite at least one
clause that a tool actually returned in this run. Uncited claims are dropped; an answer left
with no grounded claims becomes an abstention.
"""
from __future__ import annotations

import json

from pydantic import BaseModel, Field, ValidationError

from .config import AS_OF
from .llm import LLM, assistant_message
from .schema import Answer
from .tools import Toolbox

AGENT_SYSTEM = f"""TASK: agent
You are perkpilot, an assistant that answers questions about the user's credit cards using ONLY
the card terms and the user's transactions, reached through your tools. Today is {AS_OF.isoformat()}.

Rules:
- Use tools for every fact. Never rely on memory of real-world card terms; these terms are synthetic.
- Every factual claim must cite clause ids (e.g. "AG-3") that appeared in tool results, with the
  span offsets when a rule gives them.
- For "which card" questions call simulate and rank cards by expected_value.
- For expiring credits call credit_status. For "is the fee worth it" call fee_value.
- If the terms do not cover the question, set "abstained": true and answer "I don't know".
- Do arithmetic with tools, not in your head.

When done, reply with ONLY a JSON object:
{{"answer": str, "abstained": bool, "value": <machine-readable answer: card id, number, bool, list of ids, or null>,
  "claims": [{{"text": str, "citations": [{{"clause_id": str, "start": int|null, "end": int|null}}]}}],
  "recommendations": [{{"card": str, "expected_value": float, "points": float, "rationale": str, "citations": [...]}}]}}"""

IDK = "I don't know. The card terms I have don't cover that."


class AgentRun(BaseModel):
    question: str
    answer: Answer
    tool_calls: list[dict] = Field(default_factory=list)
    steps: int = 0


def parse_answer(content: str | None) -> Answer:
    text = (content or "").strip()
    if text.startswith("```"):
        text = text.strip("`").removeprefix("json").strip()
    return Answer.model_validate_json(text)


def ground(ans: Answer, valid_clauses: set[str]) -> Answer:
    if ans.abstained:
        return ans.model_copy(update={"claims": [], "recommendations": [], "value": None})
    kept, dropped = [], list(ans.dropped_claims)
    for c in ans.claims:
        cites = [x for x in c.citations if x.clause_id in valid_clauses]
        if cites:
            kept.append(c.model_copy(update={"citations": cites}))
        else:
            dropped.append(c.text)
    recs = [r.model_copy(update={"citations": [x for x in r.citations if x.clause_id in valid_clauses]})
            for r in ans.recommendations]
    if not kept and not any(r.citations for r in recs):
        return Answer(answer=IDK, abstained=True, dropped_claims=dropped)
    return ans.model_copy(update={"claims": kept, "recommendations": recs, "dropped_claims": dropped})


def run_agent(question: str, llm: LLM, toolbox: Toolbox, max_steps: int = 8) -> AgentRun:
    messages: list[dict] = [{"role": "system", "content": AGENT_SYSTEM}, {"role": "user", "content": question}]
    repaired = False
    for step in range(1, max_steps + 1):
        res = llm.chat(messages, tools=toolbox.specs)
        if res.tool_calls:
            messages.append(assistant_message(res))
            for call in res.tool_calls:
                out = toolbox.call(call.name, call.arguments)
                messages.append({"role": "tool", "tool_call_id": call.id, "name": call.name,
                                 "content": json.dumps(out, default=str)})
            continue
        try:
            ans = parse_answer(res.content)
        except ValidationError as e:
            if repaired:
                ans = Answer(answer=IDK, abstained=True, dropped_claims=[f"unparseable final answer: {e}"])
            else:
                repaired = True
                messages += [assistant_message(res), {"role": "user", "content":
                             f"Your answer was not valid JSON for the schema: {e}. Reply with the JSON object only."}]
                continue
        return AgentRun(question=question, answer=ground(ans, toolbox.seen_clauses), tool_calls=toolbox.calls,
                        steps=step)
    return AgentRun(question=question, answer=Answer(answer=IDK, abstained=True,
                                                     dropped_claims=["step limit reached"]),
                    tool_calls=toolbox.calls, steps=max_steps)
