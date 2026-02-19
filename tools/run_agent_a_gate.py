#!/usr/bin/env python3
import argparse
import os
import subprocess
import sys
from datetime import datetime, timezone
from typing import List, Tuple

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from tools._env_utils import load_runtime_env_from_secrets


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run agent.md goal A gate checks in one command.")
    parser.add_argument(
        "--python",
        default=sys.executable,
        help="Python executable path for child checks (default: current interpreter).",
    )
    parser.add_argument(
        "--report-json",
        default="data/integrity_latest.json",
        help="Output path for integrity JSON report.",
    )
    parser.add_argument(
        "--max-dup-chat",
        type=int,
        default=0,
        help="Allowed duplicate source_chat_id groups in integrity check.",
    )
    parser.add_argument(
        "--from-secrets",
        dest="from_secrets",
        action="store_true",
        default=True,
        help="Load missing runtime env keys from .streamlit/secrets.toml (default: enabled).",
    )
    parser.add_argument(
        "--no-from-secrets",
        dest="from_secrets",
        action="store_false",
        help="Disable secrets fallback and only use current process env.",
    )
    return parser.parse_args()


def _run_step(step_name: str, cmd: List[str], env: dict) -> Tuple[int, str]:
    rendered = " ".join(cmd)
    lines = [f"\n=== {step_name} ===", f"$ {rendered}"]
    proc = subprocess.run(
        cmd,
        cwd=PROJECT_ROOT,
        env=env,
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
    env = dict(os.environ)

    if args.from_secrets:
        info = load_runtime_env_from_secrets(PROJECT_ROOT)
        if info.get("loaded_keys"):
            print(f"[INFO] loaded from secrets: {', '.join(info['loaded_keys'])}")

    env["DATASTORE"] = os.getenv("DATASTORE", "")
    env["DATABASE_URL"] = os.getenv("DATABASE_URL", "")
    env["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY", "")

    steps = [
        (
            "A-0 Preflight Auth",
            [args.python, "tools/preflight_postgres_auth.py"] + ([] if args.from_secrets else ["--no-from-secrets"]),
        ),
        ("A-2 Bootstrap Check", [args.python, "tools/check_supabase_phase1.py"]),
        ("A-3 Smoke Check", [args.python, "tools/check_postdeploy_smoke.py", "--strict-postgres"]),
        (
            "A-4 Integrity Check",
            [
                args.python,
                "tools/check_data_integrity.py",
                "--expect-postgres",
                "--max-dup-chat",
                str(max(0, args.max_dup_chat)),
                "--report-json",
                args.report_json,
            ],
        ),
    ]

    print(f"[INFO] gate_started_utc={datetime.now(timezone.utc).isoformat()}")
    for step_name, cmd in steps:
        code, output = _run_step(step_name, cmd, env=env)
        print(output)
        if code != 0:
            print(f"\n[FAIL] Gate stopped at '{step_name}'.")
            return code

    print(f"\n[PASS] agent.md goal A gate passed at {datetime.now(timezone.utc).isoformat()}.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
