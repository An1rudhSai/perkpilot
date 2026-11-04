"""Build the SQLite store: extract + validate every card, then load synthetic transactions."""
from __future__ import annotations

from .config import DATA_DIR, DB_PATH
from .db import connect, save_card
from .extract import extract_card
from .llm import get_llm
from .terms import available_cards


def build(db_path=None, mode: str | None = None, with_transactions: bool = True) -> dict:
    from .validate import validate_terms

    llm = get_llm(mode)
    (DATA_DIR / "validation_disagreements.jsonl").unlink(missing_ok=True)
    summary = {}
    with connect(db_path or DB_PATH) as con:
        for card in available_cards():
            res = extract_card(card, llm)
            vals = validate_terms(res.terms, llm)
            save_card(con, res.terms, res.rejections, vals, llm.name)
            summary[card] = {"rules": len(res.terms.earning_rules), "credits": len(res.terms.credits),
                             "rejected_fields": len(res.rejections),
                             "validator_disagreements": sum(not v.agrees for v in vals)}
        if with_transactions:
            from .transactions import generate
            from .db import save_transactions

            txns = generate()
            save_transactions(con, txns)
            summary["transactions"] = len(txns)
    return summary
