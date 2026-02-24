#!/usr/bin/env python3
"""Validate skill supply-chain registry for controlled external skill intake."""

from __future__ import annotations

import argparse
import json
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

PROJECT_ROOT = Path(__file__).resolve().parent.parent
CHECKSUM_RE = re.compile(r"^[a-fA-F0-9]{64}$")
VALID_STATUS = {"candidate", "pilot", "core", "blocked"}


@dataclass
class Finding:
    severity: str
    code: str
    message: str

    def render(self) -> str:
        return f"[{self.severity}] {self.code} - {self.message}"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Check approved_skills_registry.json integrity.")
    parser.add_argument(
        "--registry",
        default="skills/approved_skills_registry.json",
        help="Path to skill registry JSON (relative to project root by default).",
    )
    parser.add_argument(
        "--mode",
        choices=("warn", "strict"),
        default="warn",
        help="warn: non-blocking, strict: return non-zero on FAIL findings.",
    )
    return parser.parse_args()


def load_registry(path: Path) -> Dict[str, object]:
    return json.loads(path.read_text(encoding="utf-8"))


def validate_registry(payload: Dict[str, object]) -> List[Finding]:
    findings: List[Finding] = []

    if not isinstance(payload.get("version"), int):
        findings.append(Finding("FAIL", "REG_VERSION_INVALID", "Top-level 'version' must be an integer."))

    allowed_sources = payload.get("allowed_sources")
    if not isinstance(allowed_sources, list) or not all(isinstance(v, str) and v.strip() for v in allowed_sources):
        findings.append(Finding("FAIL", "REG_ALLOWED_SOURCES_INVALID", "'allowed_sources' must be a non-empty string list."))
        allowed_sources_set = set()
    else:
        allowed_sources_set = set(allowed_sources)

    skills = payload.get("skills")
    if not isinstance(skills, list):
        findings.append(Finding("FAIL", "REG_SKILLS_INVALID", "Top-level 'skills' must be a list."))
        return findings

    seen_names = set()
    for idx, item in enumerate(skills):
        prefix = f"skills[{idx}]"
        if not isinstance(item, dict):
            findings.append(Finding("FAIL", "REG_ITEM_INVALID", f"{prefix} must be an object."))
            continue

        name = item.get("name")
        source = item.get("source")
        status = item.get("status")
        checksum = item.get("checksum_sha256")

        if not isinstance(name, str) or not name.strip():
            findings.append(Finding("FAIL", "REG_NAME_INVALID", f"{prefix}.name must be a non-empty string."))
            continue

        if name in seen_names:
            findings.append(Finding("FAIL", "REG_DUPLICATE_NAME", f"Duplicate skill name found: '{name}'."))
        seen_names.add(name)

        if not isinstance(source, str) or not source.strip():
            findings.append(Finding("FAIL", "REG_SOURCE_INVALID", f"{name}: source must be a non-empty string."))
        elif allowed_sources_set and source not in allowed_sources_set:
            findings.append(Finding("FAIL", "REG_SOURCE_NOT_ALLOWED", f"{name}: source '{source}' is not in allowed_sources."))

        if not isinstance(status, str) or status not in VALID_STATUS:
            findings.append(Finding("FAIL", "REG_STATUS_INVALID", f"{name}: status must be one of {sorted(VALID_STATUS)}."))
            continue

        if status in {"pilot", "core"}:
            if not isinstance(checksum, str) or not CHECKSUM_RE.match(checksum):
                findings.append(
                    Finding(
                        "FAIL",
                        "REG_CHECKSUM_REQUIRED",
                        f"{name}: pilot/core skill must include a 64-char sha256 checksum.",
                    )
                )
        else:
            if checksum is not None and (not isinstance(checksum, str) or not CHECKSUM_RE.match(checksum)):
                findings.append(
                    Finding(
                        "WARN",
                        "REG_CHECKSUM_FORMAT_WARN",
                        f"{name}: checksum exists but is not a valid 64-char sha256.",
                    )
                )

    return findings


def summarize(findings: List[Finding]) -> Tuple[int, int]:
    fail = sum(1 for f in findings if f.severity == "FAIL")
    warn = sum(1 for f in findings if f.severity == "WARN")
    return fail, warn


def main() -> int:
    args = parse_args()
    registry_path = Path(args.registry)
    if not registry_path.is_absolute():
        registry_path = PROJECT_ROOT / registry_path

    if not registry_path.exists():
        print(f"[FAIL] REG_FILE_MISSING - Registry not found: {registry_path}")
        return 1

    try:
        payload = load_registry(registry_path)
    except Exception as exc:
        print(f"[FAIL] REG_JSON_INVALID - Could not parse registry JSON: {exc}")
        return 1

    findings = validate_registry(payload)
    fail_count, warn_count = summarize(findings)

    for finding in findings:
        print(finding.render())
    print(f"[SUMMARY] FAIL={fail_count} WARN={warn_count}")

    if args.mode == "strict" and fail_count > 0:
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
