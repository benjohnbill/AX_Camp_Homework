from search_pg import rank_hybrid_rows


def test_rank_hybrid_rows_alpha_07_prefers_combined_score():
    rows = [
        {"id": "a", "content": "A", "vec_score": 0.90, "text_score": 0.10},
        {"id": "b", "content": "B", "vec_score": 0.70, "text_score": 0.70},
        {"id": "c", "content": "C", "vec_score": 0.40, "text_score": 1.00},
    ]

    ranked = rank_hybrid_rows(rows, alpha=0.7, top_k=3)

    # combined:
    # a = 0.7*0.90 + 0.3*0.10 = 0.66
    # b = 0.7*0.70 + 0.3*0.70 = 0.70
    # c = 0.7*0.40 + 0.3*1.00 = 0.58
    assert [r["id"] for r in ranked] == ["b", "a", "c"]
    assert ranked[0]["combined_score"] > ranked[1]["combined_score"] > ranked[2]["combined_score"]
