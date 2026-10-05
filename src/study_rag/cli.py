from __future__ import annotations

import argparse
import json
import sys

from study_rag.config import Settings
from study_rag.eval.run import evaluate, format_report, load_golden
from study_rag.pipeline import StudyRAG


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="study-rag", description=__doc__)
    sub = parser.add_subparsers(dest="cmd", required=True)

    p_ingest = sub.add_parser("ingest", help="index a folder of notes / slides / PDFs")
    p_ingest.add_argument("path")

    p_ask = sub.add_parser("ask", help="ask a question about your course materials")
    p_ask.add_argument("question")

    p_eval = sub.add_parser("eval", help="retrieval metrics against a golden set")
    p_eval.add_argument("golden")
    p_eval.add_argument("-k", type=int, default=5)
    p_eval.add_argument("--min-recall", type=float, default=0.0, help="fail below this (CI gate)")
    p_eval.add_argument("--json", action="store_true")

    args = parser.parse_args(argv)
    rag = StudyRAG.from_settings(Settings())

    if args.cmd == "ingest":
        n = rag.ingest(args.path)
        print(f"Indexed {n} chunks from {args.path}")
        return 0

    if args.cmd == "ask":
        answer = rag.ask(args.question)
        print(answer.text, "\n\nSources:")
        for i, sc in enumerate(answer.sources, start=1):
            section = f" — {sc.chunk.section}" if sc.chunk.section else ""
            print(f"  [{i}] {sc.chunk.source}{section}")
        return 0

    if args.cmd == "eval":
        report = evaluate(rag, load_golden(args.golden), args.k)
        print(json.dumps(report, indent=2) if args.json else format_report(report))
        hybrid_recall = report["hybrid"][f"recall@{args.k}"]
        if hybrid_recall < args.min_recall:
            print(f"FAIL: hybrid recall@{args.k}={hybrid_recall:.3f} < {args.min_recall}",
                  file=sys.stderr)
            return 1
        return 0
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
