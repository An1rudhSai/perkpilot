"""CLI: python -m perkpilot {build,ask,serve,review}"""
from __future__ import annotations

import argparse
import json


def main() -> None:
    p = argparse.ArgumentParser(prog="perkpilot")
    sub = p.add_subparsers(dest="cmd", required=True)
    b = sub.add_parser("build", help="extract + validate all cards and load synthetic transactions")
    b.add_argument("--mode", choices=["offline", "openai"])
    a = sub.add_parser("ask", help="ask the agent a question")
    a.add_argument("question")
    a.add_argument("--mode", choices=["offline", "openai"])
    s = sub.add_parser("serve", help="run the web UI")
    s.add_argument("--port", type=int, default=8000)
    sub.add_parser("review", help="print validator disagreements and rejected fields")
    args = p.parse_args()

    if args.cmd == "build":
        from .build import build

        print(json.dumps(build(mode=args.mode), indent=2))
    elif args.cmd == "ask":
        from .agent import run_agent
        from .llm import get_llm
        from .tools import Toolbox

        run = run_agent(args.question, get_llm(args.mode), Toolbox.from_db())
        print(run.answer.model_dump_json(indent=2))
    elif args.cmd == "serve":
        import uvicorn

        uvicorn.run("perkpilot.api:app", port=args.port)
    elif args.cmd == "review":
        from .db import connect

        with connect() as con:
            for r in con.execute("SELECT rule_id, issues, validator FROM validation_log WHERE agrees = 0"):
                print(f"DISAGREE {r['rule_id']} ({r['validator']}): {r['issues']}")
            for r in con.execute("SELECT clause_id, field, quote, reason FROM rejections"):
                print(f"REJECTED {r['clause_id']} {r['field']}: {r['reason']} (quote={r['quote']!r})")


if __name__ == "__main__":
    main()
