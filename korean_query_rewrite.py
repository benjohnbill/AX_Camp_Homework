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
    {"불안", "초조", "긴장", "걱정"},
    {"우울", "무기력", "가라앉다", "침체"},
    {"기쁨", "행복", "즐겁다", "뿌듯"},
    {"운동", "헬스", "러닝", "달리기", "조깅"},
    {"카페", "커피", "카페인"},
    {"연애", "관계", "애인", "남친", "여친"},
    {"공부", "학습", "학원", "시험"},
    {"잠", "수면", "불면"},
    {"돈", "지출", "소비", "예산", "재정"},
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


def rewrite_query_for_hybrid(query_text: str, max_expansions: int = 8) -> RewriteResult:
    original = (query_text or "").strip()
    if not original:
        return RewriteResult("", "", [], False)

    tokens = tokenize(original)
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
