"""FastAPI backend for the dashboard."""
from __future__ import annotations

from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from pydantic import BaseModel

from .agent import run_agent
from .config import AS_OF, CARDS, DB_PATH
from .llm import get_llm
from .terms import clause_by_id
from .tools import Toolbox

STATIC = Path(__file__).parent / "static"
app = FastAPI(title="perkpilot")


@app.on_event("startup")
def _ensure_db() -> None:
    if not DB_PATH.exists():
        from .build import build

        build()


def _tb() -> Toolbox:
    return Toolbox.from_db()


class SimRequest(BaseModel):
    merchant: str = "Merchant"
    amount: float = 100
    merchant_category: str = "dining"
    payment_method: str = "card_present"
    foreign: bool = False


class AskRequest(BaseModel):
    question: str


@app.get("/")
def index():
    return FileResponse(STATIC / "index.html")


@app.get("/api/meta")
def meta():
    llm = get_llm()
    return {"as_of": AS_OF.isoformat(), "llm": llm.name, "cards": {k: v[0] for k, v in CARDS.items()}}


@app.get("/api/credits")
def credits(month: str | None = None):
    return _tb().credit_status("all", month or AS_OF.strftime("%Y-%m"))


@app.post("/api/simulate")
def simulate(req: SimRequest):
    return _tb().simulate(**req.model_dump())


@app.post("/api/ask")
def ask(req: AskRequest):
    run = run_agent(req.question, get_llm(), _tb())
    return {**run.answer.model_dump(), "tool_calls": run.tool_calls}


@app.get("/api/fee/{card}")
def fee(card: str):
    if card not in CARDS:
        raise HTTPException(404)
    return _tb().fee_value(card)


@app.get("/api/clause/{clause_id}")
def clause(clause_id: str):
    c = clause_by_id(clause_id)
    if c is None:
        raise HTTPException(404, "unknown clause")
    return {"clause_id": c.clause_id, "card": c.card, "title": c.title, "start": c.start, "end": c.end, "text": c.text}
