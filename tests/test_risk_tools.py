import json
import subprocess
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent.parent


def _run(cmd):
    proc = subprocess.run(
        cmd,
        cwd=PROJECT_ROOT,
        text=True,
        capture_output=True,
    )
    assert proc.returncode == 0, f"cmd failed: {' '.join(cmd)}\nSTDOUT:\n{proc.stdout}\nSTDERR:\n{proc.stderr}"
    return proc


def _report_path(name: str) -> Path:
    path = PROJECT_ROOT / "data" / name
    if path.exists():
        path.unlink()
    return path


def test_check_gateway_e2e_generates_report():
    report_path = _report_path("gateway_e2e_test.json")
    _run([sys.executable, "tools/check_gateway_e2e.py", "--json-report", str(report_path)])
    payload = json.loads(report_path.read_text(encoding="utf-8"))
    assert payload["status"] == "pass"
    assert payload["mode"] == "local-testclient"
    scenarios = {item["scenario"] for item in payload.get("results", [])}
    assert {"first_bearer", "cookie_follow_up", "missing_auth", "forbidden_audience"} <= scenarios


def test_eval_korean_retrieval_generates_report():
    report_path = _report_path("korean_eval_test.json")
    _run(
        [
            sys.executable,
            "tools/eval_korean_retrieval.py",
            "--strict",
            "--json-report",
            str(report_path),
        ]
    )
    payload = json.loads(report_path.read_text(encoding="utf-8"))
    assert "baseline" in payload and "rewritten" in payload and "delta" in payload
    assert payload["rewritten"]["recall_at_k"] >= 0.60
    assert payload["rewritten"]["precision_at_k"] >= 0.20
    assert payload["rewritten"]["acceptance_top1"] >= 0.35
