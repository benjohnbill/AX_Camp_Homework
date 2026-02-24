from korean_query_rewrite import rewrite_query_for_hybrid


def test_rewrite_expands_work_stress_query():
    res = rewrite_query_for_hybrid("직장 일 때문에 버겁다")
    assert res.rewritten_query
    assert res.is_rewritten is True
    # Expansion can vary by ordering; validate presence rather than exact string.
    expanded = set(res.expanded_terms)
    assert "회사" in expanded or "업무" in expanded
    assert "스트레스" in expanded or "지치다" in expanded


def test_rewrite_keeps_unknown_query_stable():
    res = rewrite_query_for_hybrid("별자리 메타 인사이트")
    assert res.rewritten_query == "별자리 메타 인사이트"
    assert res.expanded_terms == []
    assert res.is_rewritten is False
