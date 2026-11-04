"""SQLite storage for clauses, extracted rules, rejections, validation logs and transactions."""
from __future__ import annotations

import json
import sqlite3
from contextlib import contextmanager
from pathlib import Path

from .config import DATA_DIR, DB_PATH
from .schema import CardTerms, Rejection, Transaction, ValidationRecord
from .terms import load_terms

SCHEMA = """
CREATE TABLE IF NOT EXISTS clauses (clause_id TEXT PRIMARY KEY, card TEXT, title TEXT, start INT, "end" INT, text TEXT);
CREATE TABLE IF NOT EXISTS card_terms (card TEXT PRIMARY KEY, json TEXT, extractor TEXT);
CREATE TABLE IF NOT EXISTS rejections (id INTEGER PRIMARY KEY, card TEXT, clause_id TEXT, field TEXT, value TEXT, quote TEXT, reason TEXT);
CREATE TABLE IF NOT EXISTS validation_log (id INTEGER PRIMARY KEY, card TEXT, rule_id TEXT, kind TEXT, agrees INT, issues TEXT, validator TEXT);
CREATE TABLE IF NOT EXISTS transactions (id TEXT PRIMARY KEY, date TEXT, card TEXT, merchant TEXT, category TEXT, amount REAL, payment_method TEXT, foreign_txn INT);
CREATE INDEX IF NOT EXISTS tx_card_date ON transactions(card, date);
"""


@contextmanager
def connect(path: Path | None = None):
    path = Path(path or DB_PATH)
    path.parent.mkdir(parents=True, exist_ok=True)
    con = sqlite3.connect(path)
    con.row_factory = sqlite3.Row
    con.executescript(SCHEMA)
    try:
        yield con
        con.commit()
    finally:
        con.close()


def save_card(con, terms: CardTerms, rejections: list[Rejection], validations: list[ValidationRecord], extractor: str):
    card = terms.card
    _, clauses = load_terms(card)
    for t in ("clauses", "card_terms", "rejections", "validation_log"):
        con.execute(f"DELETE FROM {t} WHERE card = ?", (card,))
    con.executemany("INSERT INTO clauses VALUES (?,?,?,?,?,?)",
                    [(c.clause_id, card, c.title, c.start, c.end, c.text) for c in clauses])
    con.execute("INSERT INTO card_terms VALUES (?,?,?)", (card, terms.model_dump_json(), extractor))
    con.executemany("INSERT INTO rejections (card, clause_id, field, value, quote, reason) VALUES (?,?,?,?,?,?)",
                    [(r.card, r.clause_id, r.field, r.value, r.quote, r.reason) for r in rejections])
    con.executemany("INSERT INTO validation_log (card, rule_id, kind, agrees, issues, validator) VALUES (?,?,?,?,?,?)",
                    [(v.card, v.rule_id, v.kind, int(v.agrees), json.dumps(v.issues), v.validator) for v in validations])
    disagreements = [v for v in validations if not v.agrees]
    if disagreements:
        DATA_DIR.mkdir(parents=True, exist_ok=True)
        with open(DATA_DIR / "validation_disagreements.jsonl", "a") as f:
            for v in disagreements:
                f.write(v.model_dump_json() + "\n")


def load_terms_db(con) -> dict[str, CardTerms]:
    return {r["card"]: CardTerms.model_validate_json(r["json"]) for r in con.execute("SELECT card, json FROM card_terms")}


def flagged_rules(con) -> dict[str, list[str]]:
    return {r["rule_id"]: json.loads(r["issues"])
            for r in con.execute("SELECT rule_id, issues FROM validation_log WHERE agrees = 0")}


def save_transactions(con, txns: list[Transaction]):
    con.execute("DELETE FROM transactions")
    con.executemany("INSERT INTO transactions VALUES (?,?,?,?,?,?,?,?)",
                    [(t.id, t.date.isoformat(), t.card, t.merchant, t.category, t.amount, t.payment_method,
                      int(t.foreign)) for t in txns])


def query_transactions(con, card: str | None = None, start: str | None = None, end: str | None = None,
                       category: str | None = None, merchant: str | None = None, limit: int | None = None
                       ) -> list[Transaction]:
    sql, args = "SELECT * FROM transactions WHERE 1=1", []
    for col, val, op in (("card", card, "="), ("date", start, ">="), ("date", end, "<="), ("category", category, "=")):
        if val:
            sql += f" AND {col} {op} ?"
            args.append(val)
    if merchant:
        sql += " AND lower(merchant) LIKE ?"
        args.append(f"%{merchant.lower()}%")
    sql += " ORDER BY date, id"
    if limit:
        sql += f" LIMIT {int(limit)}"
    return [Transaction(id=r["id"], date=r["date"], card=r["card"], merchant=r["merchant"], category=r["category"],
                        amount=r["amount"], payment_method=r["payment_method"], foreign=bool(r["foreign_txn"]))
            for r in con.execute(sql, args)]
