# Backend Hybrid Search and Context Storage Playbook

## 1. Purpose

This document records backend decisions for:
- Hybrid search behavior
- Context-enriched log storage
- Korean query handling roadmap
- Runtime path separation for Streamlit UX stability

Primary readers:
- Project owner
- Control-tower planning agent
- Antigravity backend implementation agent

Scope note:
- `Agent.md` remains the project SSOT.
- This file is backend-focused execution guidance aligned with that SSOT.

## 2. Final Decisions (Locked)

1. Hybrid ranking method: `RRF` (Reciprocal Rank Fusion)
2. Context storage shape: split storage (`content` and `context_text` are separate)
3. Context generation policy: mixed mode
- Short logs: synchronous context generation
- Long logs: asynchronous context generation
4. Runtime strategy: split execution paths
- UI path (Streamlit) is lightweight
- Context generation and retries run outside the UI request cycle

## 3. Why These Decisions

1. RRF is more stable than weighted score sum when vector and keyword score scales drift.
2. Split context storage preserves original user text and keeps future experiments safe.
3. Mixed sync/async balances immediate quality for short notes and lower latency for long writing.
4. Execution path split prevents Streamlit rerun loops from carrying heavy backend work.

## 4. Backend Design Baseline

### 4.1 Data Model Additions

Add backend fields for context lifecycle:
- `context_text`
- `context_status` (`pending|running|succeeded|failed|stale`)
- `context_version`
- `context_source_hash`
- `context_generated_at`
- `context_attempts`
- `context_last_error`
- `context_next_retry_at`

Time semantics:
- `created_at`: original log write time
- `context_generated_at`: context completion time

### 4.2 Write Flow

1. Save original text first (`content`).
2. Compute `context_source_hash` from the source text version.
3. Branch by threshold:
- Short text: generate context synchronously.
- Long text: enqueue async context job and return immediately.
4. On sync failure:
- Do not fail the user write.
- Mark as `pending` and move to async retry path.

### 4.3 Async Worker Rules

1. Claim pending jobs with lock-safe strategy.
2. Use idempotent updates keyed by `(log_id, context_version)` or equivalent guard.
3. Before writing context, compare source hash:
- Mismatch => mark old job `stale`, enqueue fresh version.
4. Retry with bounded backoff and jitter.
5. Stop after max attempts and keep actionable error metadata.

## 5. Hybrid Search Implementation Rules

### 5.1 Candidate Retrieval

Use two channels:
- Vector candidates
- Keyword candidates

Keyword path should include:
- trigram similarity
- ILIKE fallback
- current FTS path where useful

### 5.2 Rank Fusion

Use RRF on ranked candidate lists:
- `rrf_score(doc) = sum(1 / (k + rank_i(doc)))`
- Tune `k` conservatively first (for example, 60) and adjust only with evaluation data.

### 5.3 Safety Improvements

1. Add `exclude_ids` support to avoid immediate self-match after new writes.
2. If `context_text` exists, include it in retrieval signals.
3. If context is missing (`pending/failed`), fall back to original content path without breaking query flow.

## 6. Korean Handling Roadmap

## 6.1 Short Term (current realistic scope)

1. Keep trigram + vector + context_text combined retrieval.
2. Introduce small domain synonym dictionary (emotion and daily-life phrases).
3. Add lightweight query rewrite rules (rule-based first, not always LLM-based).

Expected result:
- Noticeable improvement for short Korean notes without immediate infra migration.

## 6.2 Mid Term

1. Evaluate Korean tokenizer stack (for example, OpenSearch with nori).
2. Expand synonym and rewrite dictionary with real query logs.
3. Build Korean evaluation set and quality gates (`recall@k`, `precision@k`, user acceptance).

## 6.3 Long Term

1. Add Korean reranker (cross-encoder) on top of hybrid candidates.
2. Move to two-stage retrieval:
- stage 1: broad hybrid candidate fetch
- stage 2: precision rerank

Use only if metrics justify added latency and operating cost.

## 7. Cost and UX Tradeoff Guidance

### 7.1 Cost Reality

OpenSearch and advanced rewriting do not always require paid API calls, but always add operating cost:
- self-hosting: infra + maintenance
- managed service: service bill + traffic + storage
- LLM rewrite: token cost if enabled

### 7.2 UX Impact in Streamlit Context

Without path separation, heavy backend logic will degrade UX via rerun delay.
With path separation:
- write latency stays controlled
- background enrichment does not block user input
- search quality improves progressively as context jobs complete

## 8. Rollout Sequence

1. Phase A: lock RRF + exclude_ids + context fallback behavior
2. Phase B: add context lifecycle fields and mixed sync/async path
3. Phase C: deploy worker retries, stale handling, and observability
4. Phase D: Korean short-term boosts (synonyms + rule rewrite)
5. Phase E: mid-term tokenizer PoC decision gate
6. Phase F: long-term reranker decision gate

## 9. Acceptance Criteria

### 9.1 Functional

1. New writes always persist even when context generation fails.
2. Search works with and without context presence.
3. Duplicate async jobs do not corrupt log state.

### 9.2 Quality

1. Short Korean note retrieval improves on evaluation set versus baseline.
2. Immediate self-retrieval is reduced by `exclude_ids`.

### 9.3 UX and Runtime

1. User write path is not blocked by long context processing.
2. Context job backlog, failure rate, and retry success are observable.

## 10. Out-of-Scope for This Document

1. Frontend redesign details
2. Android camera/OCR client UX
3. Full infrastructure migration plan

This document is only for backend retrieval and context-enrichment logic.
