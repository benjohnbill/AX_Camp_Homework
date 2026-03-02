from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Dict, List, Set


_TOKEN_RE = re.compile(r"[0-9A-Za-z가-힣]+")

# Rule-based synonym groups (free, deterministic).
# Keep this list intentionally short and maintainable.
_SYNONYM_GROUPS = [
    {"회사", "직장", "업무", "일", "출근"},
    {"상사", "팀장", "리더", "매니저"},
    {"힘들다", "버겁다", "지치다", "스트레스", "피곤하다"},
    {"불안", "불안하다", "초조", "긴장", "걱정", "걱정된다"},
    {"우울", "우울하다", "무기력", "무기력하다", "가라앉다", "침체"},
    {"기쁨", "행복", "행복했다", "즐겁다", "뿌듯", "좋아졌다", "기분"},
    {"운동", "헬스", "러닝", "달리기", "조깅", "산책"},
    {"카페", "커피", "카페인", "쉬다"},
    {"연애", "관계", "애인", "애인과", "남친", "여친", "갈등"},
    {"공부", "학습", "학원", "시험"},
    {"잠", "수면", "불면", "불면증", "지친다"},
    {"돈", "지출", "소비", "예산", "재정", "걱정"},
]


def _build_synonym_map(groups: List[Set[str]]) -> Dict[str, Set[str]]:
    out: Dict[str, Set[str]] = {}
    for group in groups:
        normalized = {term.strip().lower() for term in group if term and term.strip()}
        for term in normalized:
            out[term] = set(normalized)
    return out


_SYNONYM_MAP = _build_synonym_map(_SYNONYM_GROUPS)


@dataclass(frozen=True)
class RewriteResult:
    original_query: str
    rewritten_query: str
    expanded_terms: List[str]
    is_rewritten: bool


def tokenize(text: str) -> List[str]:
    return [tok.lower() for tok in _TOKEN_RE.findall(text or "")]


def _normalize_token(token: str) -> str:
    out = (token or "").lower().strip()
    if not out:
        return ""
    # Trim common postpositions to improve synonym-map hit rate for Korean queries.
    for suffix in ("으로", "에서", "에게", "께서", "보다", "처럼", "하고", "과", "와", "은", "는", "이", "가", "을", "를"):
        if out.endswith(suffix) and len(out) > len(suffix) + 1:
            out = out[: -len(suffix)]
            break
    # Trim frequent sentence endings.
    for ending in ("했다", "한다", "된다", "하다", "한다", "했다", "다"):
        if out.endswith(ending) and len(out) > len(ending) + 1:
            out = out[: -len(ending)]
            break
    return out


def rewrite_query_for_hybrid(query_text: str, max_expansions: int = 8) -> RewriteResult:
    original = (query_text or "").strip()
    if not original:
        return RewriteResult("", "", [], False)

    raw_tokens = tokenize(original)
    tokens: List[str] = []
    for tok in raw_tokens:
        if tok:
            tokens.append(tok)
        norm = _normalize_token(tok)
        if norm and norm != tok:
            tokens.append(norm)
    seen = set()
    base_terms: List[str] = []
    for token in tokens:
        if token in seen:
            continue
        seen.add(token)
        base_terms.append(token)

    expanded: List[str] = []
    expanded_seen = set(base_terms)
    for token in base_terms:
        syns = _SYNONYM_MAP.get(token, set())
        for syn in sorted(syns):
            if syn in expanded_seen:
                continue
            expanded.append(syn)
            expanded_seen.add(syn)
            if len(expanded) >= max(0, int(max_expansions)):
                break
        if len(expanded) >= max(0, int(max_expansions)):
            break

    if not base_terms:
        return RewriteResult(original, original, [], False)

    rewritten_terms = base_terms + expanded
    rewritten = " ".join(rewritten_terms).strip()
    is_rewritten = rewritten != " ".join(base_terms)
    return RewriteResult(
        original_query=original,
        rewritten_query=rewritten if rewritten else original,
        expanded_terms=expanded,
        is_rewritten=is_rewritten,
    )
