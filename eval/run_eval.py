"""Evaluation: extraction precision/recall per field, and a 3-way Q&A comparison
(plain LLM vs. RAG-only vs. full agent). Writes eval/results.md and eval/results.json.

    python -m eval.run_eval              # full run, offline scripted LLM unless an API key is set
    python -m eval.run_eval --smoke      # small subset for CI, prints only
"""
from __future__ import annotations

import argparse
import json
import time
from collections import defaultdict
from pathlib import Path

from perkpilot.agent import IDK, parse_answer, run_agent
from perkpilot.extract import extract_card
from perkpilot.llm import LLM, get_llm
from perkpilot.schema import Answer
from perkpilot.search import default_index
from perkpilot.terms import available_cards
from perkpilot.tools import Toolbox
from perkpilot.transactions import generate
from perkpilot.validate import validate_terms

HERE = Path(__file__).parent

RAG_SYSTEM = """TASK: rag_qa
Answer the user's credit-card question using ONLY the clauses provided. You have no access to the
user's transactions. Cite clause ids. If the clauses don't answer it, abstain. Reply with JSON:
{"answer": str, "abstained": bool, "value": <card id | number | bool | list | string | null>,
 "claims": [{"text": str, "citations": [{"clause_id": str}]}]}"""

PLAIN_SYSTEM = """TASK: plain_qa
Answer the user's credit-card question from your own knowledge. Card ids: amex_gold, amex_platinum,
sapphire_reserve, freedom_unlimited. If you don't know, abstain. Reply with JSON:
{"answer": str, "abstained": bool, "value": <card id | number | bool | list | string | null>, "claims": []}"""


# ---------------------------------------------------------------- extraction

def _get(obj, path: str):
    for part in path.split("."):
        if obj is None:
            return None
        obj = getattr(obj, part, None)
        if hasattr(obj, "value") and hasattr(obj, "span"):
            obj = obj.value
    return obj


def eval_extraction(llm: LLM) -> dict:
    gold = json.loads((HERE / "gold_extraction.json").read_text())["checks"]
    terms, rejections, disagreements = {}, [], []
    for card in available_cards():
        res = extract_card(card, llm)
        terms[card] = res.terms
        rejections += [r.model_dump() for r in res.rejections]
        disagreements += [v.model_dump() for v in validate_terms(res.terms, llm) if not v.agrees]
    per_field: dict[str, dict[str, int]] = defaultdict(lambda: {"correct": 0, "wrong": 0, "missing": 0, "spurious": 0})
    failures = []
    for c in gold:
        t = terms[c["card"]]
        if c["kind"] == "fee":
            obj = t.fees
        elif c["kind"] == "earning":
            obj = next((r for r in t.earning_rules if r.category.value == c["key"]), None)
        else:
            obj = next((x for x in t.credits if x.id == c["key"]), None)
        field = c["field"].rstrip("~")
        got = _get(obj, field)
        if c["field"].endswith("~"):
            got = [x.value for x in (got or [])]
            ok, present = c["expected"] in got, bool(got)
        else:
            norm = sorted(got) if isinstance(got, list) else got
            exp = sorted(c["expected"]) if isinstance(c["expected"], list) else c["expected"]
            ok, present = norm == exp, got is not None
        bucket = per_field[field]
        if ok:
            bucket["correct"] += 1
        elif c["expected"] is None:
            bucket["spurious"] += 1
        elif not present:
            bucket["missing"] += 1
        else:
            bucket["wrong"] += 1
        if not ok:
            failures.append({**c, "got": got})
    rows = {}
    for f, b in sorted(per_field.items()):
        pred = b["correct"] + b["wrong"] + b["spurious"]
        true = b["correct"] + b["wrong"] + b["missing"]
        rows[f] = {**b, "n": sum(b.values()),
                   "precision": b["correct"] / pred if pred else 1.0, "recall": b["correct"] / true if true else 1.0}
    tot = {k: sum(r[k] for r in rows.values()) for k in ("correct", "wrong", "missing", "spurious")}
    tot["precision"] = tot["correct"] / (tot["correct"] + tot["wrong"] + tot["spurious"])
    tot["recall"] = tot["correct"] / (tot["correct"] + tot["wrong"] + tot["missing"])
    return {"n_checks": len(gold), "per_field": rows, "overall": tot, "failures": failures,
            "rejections": rejections, "validator_disagreements": disagreements, "terms": terms}


# ---------------------------------------------------------------- Q&A

def run_plain(llm: LLM, q: str) -> Answer:
    res = llm.chat([{"role": "system", "content": PLAIN_SYSTEM}, {"role": "user", "content": q}], json_mode=True)
    return _safe(res.content)


def run_rag(llm: LLM, q: str) -> Answer:
    hits = default_index().search(q, k=6)
    payload = {"question": q, "clauses": [{k: h[k] for k in ("clause_id", "card", "title", "text")} for h in hits]}
    res = llm.chat([{"role": "system", "content": RAG_SYSTEM}, {"role": "user", "content": json.dumps(payload)}],
                   json_mode=True)
    return _safe(res.content)


def _safe(content) -> Answer:
    try:
        return parse_answer(content)
    except Exception:
        return Answer(answer=IDK, abstained=True)


def correct(item: dict, a: Answer) -> bool:
    t, exp, v = item["type"], item["answer"], a.value
    if t == "abstain":
        return a.abstained
    if a.abstained or v is None:
        return False
    if t == "card":
        return v in exp
    if t == "number":
        return isinstance(v, (int, float)) and not isinstance(v, bool) and abs(v - exp) <= max(0.01, 0.01 * abs(exp))
    if t == "bool":
        return v is exp
    if t == "set":
        return isinstance(v, list) and set(v) == set(exp)
    return str(v) == str(exp)


def cited(a: Answer) -> set[str]:
    return {c.clause_id for cl in a.claims for c in cl.citations} | {
        c.clause_id for r in a.recommendations for c in r.citations}


def score(items: list[dict], answers: list[Answer]) -> dict:
    ans_items = [(i, a) for i, a in zip(items, answers) if i["type"] != "abstain"]
    una_items = [(i, a) for i, a in zip(items, answers) if i["type"] == "abstain"]
    answered = [(i, a) for i, a in zip(items, answers) if not a.abstained]
    answered_ans = [(i, a) for i, a in ans_items if not a.abstained]
    halluc = [(i, a) for i, a in answered if i["type"] == "abstain" or not correct(i, a)]
    cit_ok = [(i, a) for i, a in answered_ans if cited(a) & set(i["supporting_clauses"])]
    return {
        "n": len(items),
        "answer_correctness": sum(correct(i, a) for i, a in ans_items) / max(len(ans_items), 1),
        "overall_accuracy": sum(correct(i, a) for i, a in zip(items, answers)) / max(len(items), 1),
        "citation_accuracy": len(cit_ok) / max(len(answered_ans), 1),
        "abstention_rate_unanswerable": sum(a.abstained for _, a in una_items) / max(len(una_items), 1),
        "false_abstention_rate": sum(a.abstained for _, a in ans_items) / max(len(ans_items), 1),
        "hallucination_rate": len(halluc) / max(len(answered), 1),
    }


def eval_qa(llm: LLM, items: list[dict], terms) -> dict:
    txns = generate()
    flagged = {}
    for card, t in terms.items():
        flagged.update({v.rule_id: v.issues for v in validate_terms(t, llm) if not v.agrees})
    out = {}
    for setup in ("plain", "rag", "agent"):
        answers, t0 = [], time.time()
        for it in items:
            if setup == "plain":
                answers.append(run_plain(llm, it["question"]))
            elif setup == "rag":
                answers.append(run_rag(llm, it["question"]))
            else:
                answers.append(run_agent(it["question"], llm, Toolbox(terms, txns, flagged=flagged)).answer)
        out[setup] = {"metrics": score(items, answers), "seconds": round(time.time() - t0, 2),
                      "answers": [{"id": i["id"], "category": i["category"], "correct": correct(i, a),
                                   "abstained": a.abstained, "value": a.value, "expected": i["answer"],
                                   "cited": sorted(cited(a)), "supporting": i["supporting_clauses"]}
                                  for i, a in zip(items, answers)]}
    return out


# ---------------------------------------------------------------- report

def pct(x: float) -> str:
    return f"{100 * x:.0f}%"


def render(llm: LLM, ext: dict, qa: dict, n_items: int) -> str:
    offline = llm.name == "offline-scripted"
    L = ["# perkpilot eval results", ""]
    if offline:
        L += ["> **Offline / scripted mode.** Every number below was produced by `ScriptedLLM`, a deterministic set of",
              "> regex and keyword heuristics standing in for a language model (no API key, no network). The terms text,",
              "> the heuristics and the gold set were written by the same author, so these numbers show that the",
              "> pipeline works end to end and how the three setups differ in *what information they can reach*.",
              "> They are **not** a measurement of any real LLM. Run with `PERKPILOT_LLM_API_KEY` set to get real numbers.", ""]
    L += [f"LLM: `{llm.name}` · Q&A items: {n_items} · extraction checks: {ext['n_checks']}", "",
          "## Q&A: three setups", "",
          "| Metric | Plain LLM | RAG only | Full agent |", "|---|---|---|---|"]
    names = [("answer_correctness", "Answer correctness (answerable)"), ("overall_accuracy", "Overall accuracy (incl. abstain items)"),
             ("citation_accuracy", "Citation accuracy (answered, answerable)"),
             ("abstention_rate_unanswerable", "Abstention rate on unanswerable (higher is better)"),
             ("false_abstention_rate", "False abstention on answerable (lower is better)"),
             ("hallucination_rate", "Hallucination rate (lower is better)")]
    for k, label in names:
        L.append(f"| {label} | " + " | ".join(pct(qa[s]["metrics"][k]) for s in ("plain", "rag", "agent")) + " |")
    L += ["", "Correct answers by question category:", "", "| Category | n | Plain | RAG | Agent |", "|---|---|---|---|---|"]
    cats = sorted({a["category"] for a in qa["agent"]["answers"]})
    for c in cats:
        row = [sum(a["correct"] for a in qa[s]["answers"] if a["category"] == c) for s in ("plain", "rag", "agent")]
        n = sum(a["category"] == c for a in qa["agent"]["answers"])
        L.append(f"| {c} | {n} | " + " | ".join(map(str, row)) + " |")
    L += ["", "Definitions: *answer correctness* = correct machine-readable value on answerable items; *citation accuracy* =",
          "share of answered, answerable items citing at least one gold supporting clause; *hallucination rate* = share of",
          "non-abstained answers that are wrong or answer an unanswerable question.", "",
          "## Extraction: precision / recall per field", "",
          "| Field | n | Correct | Wrong | Missing | Spurious | Precision | Recall |", "|---|---|---|---|---|---|---|---|"]
    for f, r in ext["per_field"].items():
        L.append(f"| {f} | {r['n']} | {r['correct']} | {r['wrong']} | {r['missing']} | {r['spurious']} | "
                 f"{pct(r['precision'])} | {pct(r['recall'])} |")
    o = ext["overall"]
    L.append(f"| **all** | {ext['n_checks']} | {o['correct']} | {o['wrong']} | {o['missing']} | {o['spurious']} | "
             f"**{pct(o['precision'])}** | **{pct(o['recall'])}** |")
    L += ["", "Extraction check failures:", ""]
    L += [f"- `{f['id']}` {f['card']} {f['key']}.{f['field']}: expected `{f['expected']}`, got `{f['got']}`"
          for f in ext["failures"]] or ["- none"]
    L += ["", f"Fields rejected for lack of a supporting span: {len(ext['rejections'])}", "",
          "Validator disagreements (logged for review, not auto-fixed):", ""]
    L += [f"- `{d['rule_id']}`: {'; '.join(d['issues'])}" for d in ext["validator_disagreements"]] or ["- none"]
    L += ["", "## Agent errors", ""]
    wrong = [a for a in qa["agent"]["answers"] if not a["correct"]]
    L += [f"- `{a['id']}` ({a['category']}): expected `{a['expected']}`, got `{a['value']}`"
          + (" (abstained)" if a["abstained"] else "") for a in wrong] or ["- none"]
    return "\n".join(L) + "\n"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true", help="small subset, print only (used in CI)")
    ap.add_argument("--mode", choices=["offline", "openai"])
    args = ap.parse_args()
    llm = get_llm(args.mode)
    items = json.loads((HERE / "gold_qa.json").read_text())["items"]
    if args.smoke:
        items = items[::4]
    ext = eval_extraction(llm)
    qa = eval_qa(llm, items, ext["terms"])
    md = render(llm, ext, qa, len(items))
    if args.smoke:
        print(md)
        assert qa["agent"]["metrics"]["answer_correctness"] >= 0.5, "agent smoke correctness regressed"
        return
    (HERE / "results.md").write_text(md)
    ext.pop("terms")
    (HERE / "results.json").write_text(json.dumps({"llm": llm.name, "extraction": ext, "qa": qa}, indent=1, default=str))
    print(md)


if __name__ == "__main__":
    main()
