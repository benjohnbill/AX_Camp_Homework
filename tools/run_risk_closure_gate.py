#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os
import subprocess
import sys
from datetime import datetime, timezone
from typing import List, Tuple


PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run risk-closure gate: docs contract + gateway E2E + Korean retrieval metrics."
    )
    parser.add_argument(
        "--python",
        default=sys.executable,
        help="Python executable path for child checks (default: current interpreter).",
    )
    parser.add_argument(
        "--korean-min-recall",
        type=float,
        default=0.60,
        help="Minimum recall@k for Korean eval strict check.",
    )
    parser.add_argument(
        "--korean-min-precision",
        type=float,
        default=0.20,
        help="Minimum precision@k for Korean eval strict check.",
    )
    parser.add_argument(
        "--korean-min-acceptance",
        type=float,
        default=0.35,
        help="Minimum top1 acceptance for Korean eval strict check.",
    )
    parser.add_argument(
        "--korean-max-regression",
        type=float,
        default=0.02,
        help="Allowed rewritten-vs-baseline regression for Korean eval strict check.",
    )
    return parser.parse_args()


def _run_step(step_name: str, cmd: List[str]) -> Tuple[int, str]:
    rendered = " ".join(cmd)
    lines = [f"\n=== {step_name} ===", f"$ {rendered}"]
    proc = subprocess.run(
        cmd,
        cwd=PROJECT_ROOT,
        text=True,
        capture_output=True,
    )
    if proc.stdout:
        lines.append(proc.stdout.rstrip())
    if proc.stderr:
        lines.append(proc.stderr.rstrip())
    lines.append(f"[exit={proc.returncode}]")
    return proc.returncode, "\n".join(lines)


def main() -> int:
    args = parse_args()
    print(f"[INFO] risk_gate_started_utc={datetime.now(timezone.utc).isoformat()}")

    steps = [
        (
            "R-1 Docs Contract",
            [args.python, "tools/check_docs_contract.py", "--mode", "warn", "--json-report", "data/docs_contract_report.json"],
        ),
        (
            "R-1.5 Orchestration Contract",
            [args.python, "tools/validate_contracts.py"],
        ),
        (
            "R-2 Gateway Local E2E",
            [args.python, "tools/check_gateway_e2e.py", "--json-report", "data/gateway_e2e_latest.json"],
        ),
        (
            "R-3 Korean Retrieval Eval",
            [
                args.python,
                "tools/eval_korean_retrieval.py",
                "--strict",
                "--min-recall",
                str(args.korean_min_recall),
                "--min-precision",
                str(args.korean_min_precision),
                "--min-acceptance",
                str(args.korean_min_acceptance),
                "--max-regression",
                str(args.korean_max_regression),
                "--json-report",
                "data/korean_retrieval_eval_latest.json",
            ],
        ),
    ]

    for step_name, cmd in steps:
        code, output = _run_step(step_name, cmd)
        print(output)
        if code != 0:
            print(f"\n[FAIL] Risk closure gate stopped at '{step_name}'.")
            return code

    print(f"\n[PASS] risk closure gate passed at {datetime.now(timezone.utc).isoformat()}.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
