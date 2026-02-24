#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import os
import re
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from korean_query_rewrite import rewrite_query_for_hybrid
from search_pg import rank_hybrid_rows

TOKEN_RE = re.compile(r"[0-9A-Za-z가-힣]+")


@dataclass
class QueryEval:
    query: str
    expected_ids: List[str]
    top_ids: List[str]
    hit_count: int
    top1_hit: bool
    precision_at_k: float
    recall_at_k: float


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate Korean retrieval quality with a local dataset.")
    parser.add_argument(
        "--dataset",
        default="data/korean_retrieval_eval.json",
        help="Path to evaluation dataset JSON.",
    )
    parser.add_argument("--top-k", type=int, default=3, help="Top-k for recall/precision.")
    parser.add_argument("--alpha", type=float, default=0.7, help="Hybrid ranking alpha.")
    parser.add_argument(
        "--min-recall",
        type=float,
        default=0.60,
        help="Minimum recall@k for rewritten run in strict mode.",
    )
    parser.add_argument(
        "--min-precision",
        type=float,
        default=0.20,
        help="Minimum precision@k for rewritten run in strict mode.",
    )
    parser.add_argument(
        "--min-acceptance",
        type=float,
        default=0.35,
        help="Minimum top1 acceptance for rewritten run in strict mode.",
    )
    parser.add_argument(
        "--max-regression",
        type=float,
        default=0.02,
        help="Allowed regression margin between rewritten and baseline metrics in strict mode.",
    )
    parser.add_argument(
        "--strict",
        action="store_true",
        help="Return non-zero when rewritten metrics do not meet thresholds.",
    )
    parser.add_argument(
        "--json-report",
        default="data/korean_retrieval_eval_latest.json",
        help="Output JSON report path.",
    )
    return parser.parse_args()


def _tokenize(text: str) -> List[str]:
    return [tok.lower() for tok in TOKEN_RE.findall(text or "")]


def _token_overlap_score(query_text: str, doc_text: str) -> float:
    q_tokens = set(_tokenize(query_text))
    if not q_tokens:
        return 0.0
    d_tokens = set(_tokenize(doc_text))
    return len(q_tokens & d_tokens) / max(1, len(q_tokens))


def _char_trigram_set(text: str) -> set[str]:
    normalized = (text or "").lower().strip()
    if len(normalized) < 3:
        return {normalized} if normalized else set()
    return {normalized[i : i + 3] for i in range(len(normalized) - 2)}


def _trigram_jaccard(a: str, b: str) -> float:
    sa = _char_trigram_set(a)
    sb = _char_trigram_set(b)
    if not sa or not sb:
        return 0.0
    inter = len(sa & sb)
    union = len(sa | sb)
    return inter / max(1, union)


def _evaluate_queries(
    queries: List[Dict],
    docs: List[Dict],
    top_k: int,
    alpha: float,
    use_rewrite: bool,
) -> List[QueryEval]:
    results: List[QueryEval] = []
    for item in queries:
        raw_query = str(item.get("query", ""))
        expected = [str(x) for x in item.get("expected_ids", [])]
        rewrite = rewrite_query_for_hybrid(raw_query) if use_rewrite else None
        effective_query = rewrite.rewritten_query if rewrite else raw_query

        rows = []
        for doc in docs:
            doc_id = str(doc.get("id", ""))
            content = str(doc.get("content", ""))
            vec_score = _trigram_jaccard(raw_query, content)
            text_score = _token_overlap_score(effective_query, content)
            rows.append(
                {
                    "id": doc_id,
                    "content": content,
                    "vec_score": vec_score,
                    "text_score": text_score,
                }
            )

        ranked = rank_hybrid_rows(rows, alpha=alpha, top_k=top_k)
        top_ids = [str(r["id"]) for r in ranked]
        expected_set = set(expected)
        hits = [doc_id for doc_id in top_ids if doc_id in expected_set]
        hit_count = len(hits)
        precision = hit_count / max(1, len(top_ids))
        recall = hit_count / max(1, len(expected_set))
        top1_hit = bool(top_ids) and top_ids[0] in expected_set

        results.append(
            QueryEval(
                query=raw_query,
                expected_ids=expected,
                top_ids=top_ids,
                hit_count=hit_count,
                top1_hit=top1_hit,
                precision_at_k=precision,
                recall_at_k=recall,
            )
        )
    return results


def _aggregate(results: List[QueryEval]) -> Dict[str, float]:
    if not results:
        return {"recall_at_k": 0.0, "precision_at_k": 0.0, "acceptance_top1": 0.0}
    recall = sum(r.recall_at_k for r in results) / len(results)
    precision = sum(r.precision_at_k for r in results) / len(results)
    acceptance = sum(1.0 if r.top1_hit else 0.0 for r in results) / len(results)
    return {
        "recall_at_k": recall,
        "precision_at_k": precision,
        "acceptance_top1": acceptance,
    }


def _round_metrics(metrics: Dict[str, float]) -> Dict[str, float]:
    return {k: round(v, 4) for k, v in metrics.items()}


def _load_dataset(path: Path) -> Dict:
    return json.loads(path.read_text(encoding="utf-8"))


def main() -> int:
    args = parse_args()
    dataset_path = Path(args.dataset)
    if not dataset_path.is_absolute():
        dataset_path = PROJECT_ROOT / dataset_path
    if not dataset_path.exists():
        print(f"[FAIL] dataset not found: {dataset_path}")
        return 1

    raw = _load_dataset(dataset_path)
    docs = list(raw.get("documents", []))
    queries = list(raw.get("queries", []))
    if not docs or not queries:
        print("[FAIL] dataset must include non-empty `documents` and `queries`.")
        return 1

    top_k = max(1, int(args.top_k))
    alpha = float(args.alpha)

    baseline_results = _evaluate_queries(queries, docs, top_k=top_k, alpha=alpha, use_rewrite=False)
    rewritten_results = _evaluate_queries(queries, docs, top_k=top_k, alpha=alpha, use_rewrite=True)

    baseline_metrics = _aggregate(baseline_results)
    rewritten_metrics = _aggregate(rewritten_results)
    delta = {
        "recall_at_k": rewritten_metrics["recall_at_k"] - baseline_metrics["recall_at_k"],
        "precision_at_k": rewritten_metrics["precision_at_k"] - baseline_metrics["precision_at_k"],
        "acceptance_top1": rewritten_metrics["acceptance_top1"] - baseline_metrics["acceptance_top1"],
    }

    print("[BASELINE]", _round_metrics(baseline_metrics))
    print("[REWRITTEN]", _round_metrics(rewritten_metrics))
    print("[DELTA]", _round_metrics(delta))

    report = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "dataset": str(dataset_path),
        "top_k": top_k,
        "alpha": alpha,
        "baseline": _round_metrics(baseline_metrics),
        "rewritten": _round_metrics(rewritten_metrics),
        "delta": _round_metrics(delta),
        "thresholds": {
            "min_recall": args.min_recall,
            "min_precision": args.min_precision,
            "min_acceptance": args.min_acceptance,
            "max_regression": args.max_regression,
        },
        "rewritten_query_examples": [
            {
                "query": q.get("query", ""),
                "rewritten_query": rewrite_query_for_hybrid(str(q.get("query", ""))).rewritten_query,
                "expanded_terms": rewrite_query_for_hybrid(str(q.get("query", ""))).expanded_terms,
            }
            for q in queries[:5]
        ],
    }
    report_path = Path(args.json_report)
    if not report_path.is_absolute():
        report_path = PROJECT_ROOT / report_path
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[INFO] report saved: {report_path}")

    if args.strict:
        rec_ok = rewritten_metrics["recall_at_k"] >= args.min_recall
        pre_ok = rewritten_metrics["precision_at_k"] >= args.min_precision
        acc_ok = rewritten_metrics["acceptance_top1"] >= args.min_acceptance
        regression_ok = (
            delta["recall_at_k"] >= (-1.0 * args.max_regression)
            and delta["precision_at_k"] >= (-1.0 * args.max_regression)
            and delta["acceptance_top1"] >= (-1.0 * args.max_regression)
        )
        if not (rec_ok and pre_ok and acc_ok and regression_ok):
            print(
                "[FAIL] thresholds not met: "
                f"recall({rewritten_metrics['recall_at_k']:.4f}/{args.min_recall}), "
                f"precision({rewritten_metrics['precision_at_k']:.4f}/{args.min_precision}), "
                f"acceptance({rewritten_metrics['acceptance_top1']:.4f}/{args.min_acceptance}), "
                f"delta_floor(-{args.max_regression})={_round_metrics(delta)}"
            )
            return 1

    print("[PASS] korean retrieval evaluation completed.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
