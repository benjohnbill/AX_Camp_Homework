#!/usr/bin/env python3
import argparse
import json
import os
import re
import subprocess
import sys
from dataclasses import dataclass, asdict
from datetime import UTC, date, datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

PROJECT_ROOT = Path(__file__).resolve().parent.parent

REQUIRED_DOCS = {
    "Agent.md",
    "Harness_Policy.md",
    "BACKEND_HYBRID_CONTEXT_PLAYBOOK.md",
    "DEBUG_TOKEN_GOVERNANCE.md",
    "SKILL_PROMOTION_POLICY.md",
    "MCP_USAGE_POLICY.md",
    "Antigravity_Agent.md",
    "Android_Studio_Agent.md",
    "Gemini-3.1-Pro_Agent.md",
    "integration_status.md",
}

SYNC_RULES = {
    "Agent.md": {
        "Antigravity_Agent.md",
        "Android_Studio_Agent.md",
        "Gemini-3.1-Pro_Agent.md",
    },
    "Harness_Policy.md": {"Agent.md"},
    "BACKEND_HYBRID_CONTEXT_PLAYBOOK.md": {"Agent.md", "Antigravity_Agent.md"},
    "DEBUG_TOKEN_GOVERNANCE.md": {"Agent.md", "integration_status.md"},
    "SKILL_PROMOTION_POLICY.md": {"Agent.md"},
    "MCP_USAGE_POLICY.md": {"Agent.md", "Harness_Policy.md"},
}

LEGACY_FRONTMATTER_ALLOWLIST = {
    "Agent.md",
    "Android_Studio_Agent.md",
    "Antigravity_Agent.md",
    "BACKEND_HYBRID_CONTEXT_PLAYBOOK.md",
    "DEBUG_TOKEN_GOVERNANCE.md",
    "Gemini-3.1-Pro_Agent.md",
    "Harness_Policy.md",
    "integration_status.md",
    "oneoff_diagnostic_2026-02-20.md",
    "SKILL_PROMOTION_POLICY.md",
}
LEGACY_FRONTMATTER_ALLOWLIST_LOWER = {
    item.lower() for item in LEGACY_FRONTMATTER_ALLOWLIST
}.union(
    {
        "agent.md",
        "agents.md",
        "android_studio_agent.md",
        "antigravity_agent.md",
        "gemini-3.1-pro_agent.md",
        "android/narrativeloopmobile/agent.md",
        "android/narrativeloopmobile/android_studio_agent.md",
        "android/narrativeloopmobile/android_report.md",
        "android/narrativeloopmobile/debug_token_governance.md",
        "android/narrativeloopmobile/integration_status.md",
        "android/narrativeloopmobile/temp_android_progress_report.md",
        "android/readme.md",
    }
)

STATUS_DOCS = {"integration_status.md"}
REQUIRED_METADATA_KEYS = {
    "doc_type",
    "owner",
    "authority_level",
    "last_updated",
    "sync_with",
    "change_triggers",
    "sunset_condition",
}

MD_REF_RE = re.compile(
    r"`([^`\n]+\.md)`|\[[^\]]*\]\(([^)\s]+\.md)\)|(?<![/\w.-])([A-Za-z][A-Za-z0-9_./-]*\.md)\b"
)
IGNORED_DOC_REFS = {
    "SKILL.md",
    "integration_handover_v1.md",
    "SYSTEM_BLUEPRINT.md",
    "SYSTEM_AGENT_POLICY.md",
    "SYSTEM_HANDOFF_CONSTITUTION.md",
    "SYSTEM_HANDOFF_MIGRATION_POLICY.md",
    "SYSTEM_SKILL_GOVERNANCE_POLICY.md",
    "SYSTEM_MCP_POLICY.md",
    "SYSTEM_REMOTE_POLICY.md",
    "INBOX.md",
}
IGNORED_MD_DIR_FRAGMENTS = {
    "/.git/",
    "/.pytest_cache/",
    "/__pycache__/",
    "/venv/",
    "/venv_new/",
    "/venv_backup_",
    "/.gradle-user/",
    "/.pre-commit-cache/",
    "/data/evidence/",
    "/orchestration/results/",
    "/android/NarrativeLoopMobile/evidence/",
    "/android/NarrativeLoopMobile/orchestration/results/",
}
SECRET_PATTERNS = [
    ("OPENAI_KEY", re.compile(r"\bsk-[A-Za-z0-9]{20,}\b")),
    ("GEMINI_KEY", re.compile(r"\bAIza[0-9A-Za-z\-_]{35}\b")),
    ("POSTGRES_DSN", re.compile(r"postgres(?:ql)?://[^\s`]+", re.IGNORECASE)),
    ("PRIVATE_KEY_BLOCK", re.compile(r"-----BEGIN (?:RSA |EC )?PRIVATE KEY-----")),
    ("SLACK_TOKEN", re.compile(r"\bxox[baprs]-[A-Za-z0-9-]{10,}\b")),
]
STATUS_POLICY_MARKERS = [
    "Authority Levels",
    "Conflict Rule",
    "Required Front-Matter",
    "doc_type:",
    "authority_level:",
    "sync_with:",
    "Adoption Stages",
]
STRICT_WARN_CODES = {
    "DOC_SYNC_DRIFT",
    "STATUS_POLICY_LIKE_CONTENT",
    "STATUS_NO_POLICY_ANCHOR",
    "TEMP_REVIEW_EXPIRED",
    "TEMP_TTL_EXCEEDED",
}
TEMP_DOC_RE = re.compile(r"^oneoff_.*?(\d{4}-\d{2}-\d{2})\.md$")
DATE_RE = re.compile(r"\b(\d{4}-\d{2}-\d{2})\b")


@dataclass
class Finding:
    severity: str
    code: str
    file: str
    message: str
    line: Optional[int] = None

    def render(self) -> str:
        suffix = f":{self.line}" if self.line else ""
        return f"[{self.severity}] {self.code} {self.file}{suffix} - {self.message}"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Check markdown documentation contract policy.")
    parser.add_argument(
        "--mode",
        choices=("warn", "strict"),
        default="warn",
        help="warn: non-blocking (default), strict: return non-zero on FAIL and critical WARN findings.",
    )
    parser.add_argument(
        "--changed",
        nargs="*",
        default=None,
        help="Optional changed files list for sync-rule checks (relative paths).",
    )
    parser.add_argument(
        "--temp-ttl-days",
        type=int,
        default=30,
        help="Fallback TTL for oneoff docs when explicit review date is missing.",
    )
    parser.add_argument(
        "--json-report",
        default="",
        help="Optional JSON report output path.",
    )
    return parser.parse_args()


def read_text(path: Path) -> str:
    return path.read_text(encoding="utf-8", errors="replace")


def parse_frontmatter(text: str) -> Tuple[Optional[Dict[str, object]], str]:
    if not text.startswith("---\n"):
        return None, text
    end = text.find("\n---\n", 4)
    if end == -1:
        return None, text
    raw = text[4:end]
    body = text[end + 5 :]
    data: Dict[str, object] = {}
    current_key: Optional[str] = None
    for line in raw.splitlines():
        if not line.strip():
            continue
        if line.startswith("  - ") and current_key:
            data.setdefault(current_key, [])
            if isinstance(data[current_key], list):
                data[current_key].append(line[4:].strip())
            continue
        if ":" in line:
            key, value = line.split(":", 1)
            current_key = key.strip()
            value = value.strip()
            if value:
                data[current_key] = value
            else:
                data[current_key] = []
    return data, body


def collect_md_files(root: Path) -> List[Path]:
    files: List[Path] = []
    for path in root.rglob("*.md"):
        rel = f"/{normalize_rel(path, root)}/"
        if any(fragment in rel for fragment in IGNORED_MD_DIR_FRAGMENTS):
            continue
        files.append(path)
    return sorted(files)


def normalize_rel(path: Path, root: Path) -> str:
    return str(path.relative_to(root)).replace("\\", "/")


def resolve_ref(
    base: Path, ref: str, root: Path, basename_index: Optional[Dict[str, List[Path]]] = None
) -> Optional[Path]:
    clean_ref = ref.split("#", 1)[0].split("?", 1)[0].strip()
    if not clean_ref:
        return None
    direct = root / clean_ref
    if direct.exists():
        return direct
    local = base.parent / clean_ref
    if local.exists():
        return local
    if basename_index is not None and "/" not in clean_ref and "\\" not in clean_ref:
        candidates = basename_index.get(clean_ref.lower(), [])
        if len(candidates) == 1:
            return candidates[0]
    return None


def line_of_offset(text: str, offset: int) -> int:
    return text.count("\n", 0, offset) + 1


def should_ignore_secret(snippet: str) -> bool:
    lowered = snippet.lower()
    return (
        "***" in snippet
        or "<" in snippet
        or "example" in lowered
        or "placeholder" in lowered
        or "token" in lowered and ":" in lowered and "<" in snippet
    )


def gather_changed_files(root: Path, changed_arg: Optional[List[str]]) -> Tuple[Set[str], Optional[str]]:
    if changed_arg is not None and len(changed_arg) > 0:
        cleaned = {str(Path(p)).replace("\\", "/") for p in changed_arg}
        return cleaned, None

    changed: Set[str] = set()
    any_success = False

    diff_cmd = ["git", "-C", str(root), "diff", "--name-only", "--diff-filter=ACMRTUXB", "HEAD"]
    diff_proc = subprocess.run(diff_cmd, capture_output=True, text=True, encoding="utf-8", errors="replace")
    if diff_proc.returncode == 0:
        any_success = True
        for ln in diff_proc.stdout.splitlines():
            ln = ln.strip()
            if ln:
                changed.add(ln.replace("\\", "/"))

    status_cmd = ["git", "-C", str(root), "status", "--porcelain"]
    status_proc = subprocess.run(status_cmd, capture_output=True, text=True, encoding="utf-8", errors="replace")
    if status_proc.returncode == 0:
        any_success = True
        for ln in status_proc.stdout.splitlines():
            ln = ln.rstrip()
            if len(ln) < 4:
                continue
            path_part = ln[3:].strip()
            if " -> " in path_part:
                path_part = path_part.split(" -> ", 1)[1].strip()
            if path_part:
                changed.add(path_part.replace("\\", "/"))

    if not any_success:
        return set(), "git_unavailable"
    return changed, None


def check_required_docs(root: Path, findings: List[Finding]) -> None:
    for doc in sorted(REQUIRED_DOCS):
        if not (root / doc).exists():
            findings.append(Finding("FAIL", "DOC_REQUIRED_MISSING", doc, "Required document is missing."))


def check_md_references(root: Path, md_files: List[Path], findings: List[Finding]) -> None:
    basename_index: Dict[str, List[Path]] = {}
    for item in md_files:
        basename_index.setdefault(item.name.lower(), []).append(item)

    for path in md_files:
        text = read_text(path)
        rel = normalize_rel(path, root)
        for match in MD_REF_RE.finditer(text):
            ref = match.group(1) or match.group(2) or match.group(3)
            if not ref:
                continue
            ref_base = ref.split("#", 1)[0].split("?", 1)[0]
            if ref_base in IGNORED_DOC_REFS:
                continue
            if ref.startswith("http"):
                continue
            target = resolve_ref(path, ref, root, basename_index=basename_index)
            if target is None:
                findings.append(
                    Finding(
                        "WARN",
                        "DOC_LINK_MISSING",
                        rel,
                        f"Referenced markdown not found: {ref}",
                        line=line_of_offset(text, match.start()),
                    )
                )


def check_frontmatter(root: Path, md_files: List[Path], findings: List[Finding]) -> None:
    for path in md_files:
        rel = normalize_rel(path, root)
        rel_lower = rel.lower()
        text = read_text(path)
        fm, _ = parse_frontmatter(text)

        if rel.endswith("SKILL.md") and "/skills/" in f"/{rel}":
            if fm is None:
                findings.append(Finding("WARN", "SKILL_FRONTMATTER_MISSING", rel, "Skill file missing frontmatter."))
                continue
            missing = [k for k in ("name", "description") if k not in fm]
            if missing:
                findings.append(
                    Finding(
                        "WARN",
                        "SKILL_FRONTMATTER_INVALID",
                        rel,
                        f"Skill frontmatter missing keys: {', '.join(missing)}",
                    )
                )
            continue

        if rel_lower in LEGACY_FRONTMATTER_ALLOWLIST_LOWER:
            continue
        if fm is None:
            findings.append(Finding("WARN", "DOC_FRONTMATTER_MISSING", rel, "Missing metadata frontmatter."))
            continue
        missing = sorted(REQUIRED_METADATA_KEYS - set(fm.keys()))
        if missing:
            findings.append(
                Finding(
                    "WARN",
                    "DOC_FRONTMATTER_INVALID",
                    rel,
                    f"Missing frontmatter keys: {', '.join(missing)}",
                )
            )


def check_sync_rules(root: Path, changed_files: Set[str], findings: List[Finding]) -> None:
    if not changed_files:
        findings.append(
            Finding("INFO", "DOC_SYNC_SKIPPED", "workspace", "No changed file set detected; sync matrix check skipped.")
        )
        return

    names_lower = {Path(p).name.lower() for p in changed_files}
    for trigger, required in SYNC_RULES.items():
        if trigger.lower() not in names_lower:
            continue
        missing = sorted(req for req in required if req.lower() not in names_lower)
        for miss in missing:
            findings.append(
                Finding(
                    "WARN",
                    "DOC_SYNC_DRIFT",
                    trigger,
                    f"Policy sync target not changed in same batch: {miss}",
                )
            )


def check_status_guard(root: Path, findings: List[Finding]) -> None:
    for doc in sorted(STATUS_DOCS):
        path = root / doc
        if not path.exists():
            continue
        text = read_text(path)
        if "Harness_Policy.md" not in text:
            findings.append(
                Finding(
                    "WARN",
                    "STATUS_NO_POLICY_ANCHOR",
                    doc,
                    "Status document should reference Harness_Policy.md.",
                )
            )
        for marker in STATUS_POLICY_MARKERS:
            idx = text.find(marker)
            if idx == -1:
                continue
            findings.append(
                Finding(
                    "WARN",
                    "STATUS_POLICY_LIKE_CONTENT",
                    doc,
                    f"Policy-like marker found in status doc: {marker}",
                    line=line_of_offset(text, idx),
                )
            )


def parse_date_from_text(value: str) -> Optional[date]:
    match = DATE_RE.search(value or "")
    if not match:
        return None
    try:
        return datetime.strptime(match.group(1), "%Y-%m-%d").date()
    except ValueError:
        return None


def check_temporary_ttl(root: Path, md_files: List[Path], findings: List[Finding], ttl_days: int) -> None:
    today = date.today()
    ttl = timedelta(days=max(1, ttl_days))
    for path in md_files:
        rel = normalize_rel(path, root)
        name = path.name
        text = read_text(path)
        fm, _ = parse_frontmatter(text)
        is_temp = False
        file_date: Optional[date] = None

        match = TEMP_DOC_RE.match(name)
        if match:
            is_temp = True
            file_date = parse_date_from_text(match.group(1))
        if isinstance(fm, dict) and str(fm.get("doc_type", "")).strip().lower() == "temporary":
            is_temp = True

        if not is_temp:
            continue
        if fm is None:
            findings.append(
                Finding(
                    "WARN",
                    "TEMP_FRONTMATTER_MISSING",
                    rel,
                    "Temporary doc should include owner/sunset metadata frontmatter.",
                )
            )
        review_date = None
        if isinstance(fm, dict):
            review_date = parse_date_from_text(str(fm.get("review_by", "")))
            if review_date is None:
                review_date = parse_date_from_text(str(fm.get("sunset_condition", "")))

        if review_date:
            if review_date < today:
                findings.append(
                    Finding(
                        "WARN",
                        "TEMP_REVIEW_EXPIRED",
                        rel,
                        f"Temporary doc review date expired: {review_date.isoformat()}",
                    )
                )
            continue

        if file_date and (today - file_date) > ttl:
            findings.append(
                Finding(
                    "WARN",
                    "TEMP_TTL_EXCEEDED",
                    rel,
                    f"Temporary doc exceeded TTL ({ttl_days} days) without explicit review date.",
                )
            )


def check_secrets(root: Path, md_files: List[Path], findings: List[Finding]) -> None:
    for path in md_files:
        rel = normalize_rel(path, root)
        text = read_text(path)
        for code, pattern in SECRET_PATTERNS:
            for match in pattern.finditer(text):
                snippet = match.group(0)
                if should_ignore_secret(snippet):
                    continue
                findings.append(
                    Finding(
                        "FAIL",
                        f"SECRET_{code}",
                        rel,
                        "Potential secret leakage pattern in markdown.",
                        line=line_of_offset(text, match.start()),
                    )
                )


def summarize(findings: List[Finding]) -> Dict[str, int]:
    result = {"FAIL": 0, "WARN": 0, "INFO": 0}
    for finding in findings:
        if finding.severity in result:
            result[finding.severity] += 1
    return result


def write_json_report(path: str, findings: List[Finding], counts: Dict[str, int]) -> None:
    report_path = Path(path)
    if not report_path.is_absolute():
        report_path = PROJECT_ROOT / report_path
    report_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "generated_at": datetime.now(UTC).isoformat(),
        "counts": counts,
        "findings": [asdict(item) for item in findings],
    }
    report_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def main() -> int:
    args = parse_args()
    root = PROJECT_ROOT
    md_files = collect_md_files(root)
    findings: List[Finding] = []

    changed_files, changed_err = gather_changed_files(root, args.changed)
    if changed_err:
        findings.append(
            Finding(
                "INFO",
                "DOC_SYNC_GIT_UNAVAILABLE",
                "workspace",
                "Git changed-file discovery unavailable; pass --changed for deterministic sync checks.",
            )
        )

    check_required_docs(root, findings)
    check_md_references(root, md_files, findings)
    check_frontmatter(root, md_files, findings)
    check_sync_rules(root, changed_files, findings)
    check_status_guard(root, findings)
    check_temporary_ttl(root, md_files, findings, ttl_days=args.temp_ttl_days)
    check_secrets(root, md_files, findings)

    counts = summarize(findings)
    strict_warn_count = sum(
        1 for item in findings if item.severity == "WARN" and item.code in STRICT_WARN_CODES
    )
    for item in findings:
        print(item.render())
    print(
        f"[SUMMARY] FAIL={counts['FAIL']} WARN={counts['WARN']} "
        f"STRICT_WARN={strict_warn_count} INFO={counts['INFO']}"
    )

    if args.json_report:
        write_json_report(args.json_report, findings, counts)
        print(f"[INFO] JSON report written: {args.json_report}")

    if args.mode == "strict" and (counts["FAIL"] > 0 or strict_warn_count > 0):
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
