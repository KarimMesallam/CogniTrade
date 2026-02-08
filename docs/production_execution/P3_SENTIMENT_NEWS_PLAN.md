# P3 Sentiment + News Execution Plan

Last updated: 2026-02-08  
Scope: `P3-01` to `P3-05` in `IMPLEMENTATION_TRACKER.md`

## 1) Goal

Add a production-safe sentiment and news decision layer that improves risk-adjusted returns without bypassing existing hard risk controls.

Design principle:
1. Deterministic features first.
2. LLM only for structured reasoning and tool orchestration.
3. Every sentiment-driven change must pass walk-forward validation, then shadow/canary promotion gates.

## 2) Research Findings (What We Will Use)

## LLM orchestration

1. OpenAI function calling and Structured Outputs support strict schema-driven tool calls.  
Decision: use schema-validated tool calls for sentiment agent output contracts.
2. OpenAI `gpt-5-mini` is currently positioned as a fast/cost-efficient reasoning model option.  
Decision: use `gpt-5-mini` as default sentiment tool-calling orchestrator.
3. DeepSeek docs are currently mixed: one reasoning-model page states function calling is not supported, while newer DeepSeek tool-calls/thinking-mode/pricing docs describe tool calling for V3.2 models.  
Decision: keep DeepSeek tool-calling path optional/experimental and protect with integration tests plus safe fallback.

## Sentiment/news data providers

1. Alpha Vantage provides a `NEWS_SENTIMENT` endpoint with ticker/topic/time filters and sentiment metadata.  
Decision: primary structured sentiment/news adapter.
2. NewsAPI provides broad article search (`/v2/everything`) with filtering/sorting/pagination.  
Decision: secondary adapter for redundancy and broader source coverage.
3. Alternative.me provides Crypto Fear & Greed Index API and requires attribution near display.  
Decision: ingest as a market-regime sentiment factor; include attribution in reporting surfaces.
4. GDELT DOC 2.0 API provides large-scale multilingual news querying with filters useful for event clustering and narrative dispersion.  
Decision: optional tertiary adapter for broader event-context enrichment.

## Model for deterministic sentiment scoring

1. FinBERT is finance-domain sentiment NLP with labels `positive/neutral/negative` and is widely used for financial text scoring.  
Decision: deterministic headline/body scoring uses FinBERT before any LLM summarization.

## Backtesting/validation method

1. Freqtrade docs and community practice emphasize lookahead-bias checks and realistic strategy validation workflows.  
Decision: preserve current engine path for continuity, but require no-lookahead checks and walk-forward ablations in `P3-05`.

## 3) Target Architecture

## 3.1 Pipeline stages

1. Ingestion stage:
`providers -> normalized news_event rows`
2. Enrichment stage:
deduplication, symbol mapping, source quality tags, language normalization.
3. Feature stage:
point-in-time sentiment features per `(symbol, timeframe_bucket)`.
4. Decision-agent stage:
tool-calling model produces structured sentiment intent.
5. Policy/risk stage:
intent influences weights/sizing only inside bounded caps.
6. Audit/monitoring stage:
every input/feature/decision stored with timestamps and model/tool metadata.

## 3.2 Runtime integration points (current codebase)

1. Decision entry point: `bot/main.py` (LLM decision currently called in loop).
2. LLM orchestration: `bot/llm_manager.py`.
3. Policy/routing target: `bot/policy.py` (planned by `P1-09`).
4. Risk hard-stop layer: `bot/risk_engine.py` (planned by `P2-05`).
5. Persistence layer: `bot/database.py`.

## 4) Data Contracts (Mandatory)

## 4.1 `news_events`

Required columns:
1. `event_id` (ULID/UUID, immutable)
2. `provider`
3. `provider_event_id`
4. `symbol_candidates` (array/json)
5. `published_at_utc`
6. `ingested_at_utc`
7. `headline`
8. `body_excerpt`
9. `url`
10. `source_name`
11. `source_reliability_score` (config-driven)
12. `provider_sentiment_label` (nullable)
13. `provider_sentiment_score` (nullable)
14. `raw_payload_hash`

Constraints:
1. unique index on `(provider, provider_event_id)` where available.
2. never overwrite original text fields; append corrected parses in side fields.

## 4.2 `sentiment_features`

Required columns:
1. `feature_id`
2. `symbol`
3. `time_bucket_start_utc`
4. `time_bucket_end_utc`
5. `event_count`
6. `sentiment_score_mean`
7. `sentiment_score_weighted`
8. `sentiment_velocity`
9. `sentiment_dispersion`
10. `fear_greed_value`
11. `fear_greed_classification`
12. `computed_at_utc`
13. `feature_version`

Constraint:
1. features use only events with `published_at_utc <= decision_time`.

## 4.3 `sentiment_decision_audit`

Required columns:
1. `decision_id`
2. `decision_time_utc`
3. `symbol`
4. `agent_model`
5. `tool_trace` (tool names + input hashes)
6. `agent_output_json`
7. `schema_valid` (bool)
8. `fallback_reason` (nullable)
9. `applied_policy_modifier`
10. `risk_override_reason` (nullable)

## 5) Sentiment Agent Contract

The agent does not place orders. It returns a constrained intent payload.

Schema (v1):
```json
{
  "direction": "LONG|SHORT|NEUTRAL",
  "confidence": 0.0,
  "position_multiplier": 1.0,
  "time_horizon_minutes": 60,
  "thesis": "short text",
  "risk_flags": ["string"],
  "invalidation_conditions": ["string"]
}
```

Enforcement rules:
1. `confidence` must be `0.0..1.0`.
2. `position_multiplier` clipped to configured max.
3. invalid schema -> fallback to `NEUTRAL`, multiplier `1.0`.
4. missing/failed tools -> fallback to `NEUTRAL`.
5. risk engine can only reduce exposure, never increase beyond caps.

## 6) Env Configuration Additions (P3)

Add to `.env.example` and config parser:
1. `ENABLE_SENTIMENT_DECISIONS=False`
2. `SENTIMENT_PRIMARY_PROVIDER=alphavantage`
3. `SENTIMENT_SECONDARY_PROVIDER=newsapi`
4. `ALPHAVANTAGE_API_KEY=`
5. `NEWSAPI_API_KEY=`
6. `ENABLE_GDELT_PROVIDER=False`
7. `ENABLE_FEAR_GREED=True`
8. `SENTIMENT_FETCH_INTERVAL_SECONDS=300`
9. `SENTIMENT_TIME_BUCKET_MINUTES=15`
10. `SENTIMENT_MIN_EVENT_COUNT=3`
11. `SENTIMENT_MIN_CONFIDENCE=0.65`
12. `SENTIMENT_MAX_POSITION_MULTIPLIER=1.20`
13. `SENTIMENT_MAX_NEGATIVE_MULTIPLIER_REDUCTION=0.50`
14. `SENTIMENT_AGENT_MODEL=gpt-5-mini`
15. `SENTIMENT_AGENT_PROVIDER=openai`
16. `SENTIMENT_AGENT_TIMEOUT_SECONDS=12`
17. `SENTIMENT_SHADOW_MODE=True`

## 7) Detailed Task Plan

## P3-01: News/Sentiment Ingestion

Deliverables:
1. Provider adapter interfaces.
2. Alpha Vantage adapter implementation.
3. NewsAPI adapter implementation.
4. Fear & Greed adapter implementation.
5. Optional GDELT adapter behind feature flag.

Acceptance additions:
1. idempotent ingestion on repeated fetch windows.
2. timestamp parsing to UTC only.
3. provider outage isolation (one provider failure does not stop pipeline).

## P3-02: Feature Engine

Deliverables:
1. FinBERT scoring module.
2. deterministic aggregation by bucket.
3. symbol mapping rules for BTC/ETH majors first.
4. persistence + versioned feature computation.

Acceptance additions:
1. no future-leakage joins.
2. deterministic output for fixed inputs.
3. missing-news behavior returns neutral/default features.

## P3-03: Tool-Calling Agent

Deliverables:
1. tool registry: `get_sentiment_snapshot`, `get_recent_news`, `get_regime_state`, `get_risk_state`.
2. strict schema validator and sanitizer.
3. fallback circuit for timeout/tool/model failure.
4. full decision audit row write.

Acceptance additions:
1. malformed model output cannot reach execution path unvalidated.
2. agent cannot bypass risk engine or direct order manager.
3. all decisions include trace IDs.

## P3-04: Policy + Risk Integration

Deliverables:
1. sentiment modifier in policy engine.
2. bounded multipliers by regime.
3. integration with short-capable mode (`P1-06/P1-07`) once available.
4. kill-switch override when sentiment feed stale or failing.

Acceptance additions:
1. stale sentiment automatically disables sentiment modifier.
2. multiplier cap enforcement validated before order submission.
3. risk overrides logged and queryable.

## P3-05: Validation + Rollout

Deliverables:
1. ablation framework:
baseline vs baseline+sentiment vs baseline+sentiment+regime.
2. walk-forward evaluation per regime slice.
3. mandatory shadow run report template.
4. canary promotion checklist with automatic fail thresholds.

Acceptance additions:
1. promotion blocked if uplift is not robust across windows/regimes.
2. promotion blocked if turnover/slippage worsens beyond thresholds.
3. promotion blocked without shadow/canary evidence artifacts.

## 8) Validation Metrics (Promotion Gates)

Hard must-pass:
1. max drawdown not worse than baseline by configured tolerance.
2. no statistically fragile uplift (single-window-only gains rejected).
3. latency budget met for decision cycle.
4. no critical schema or fallback errors in shadow period.

Track and report:
1. Sharpe, Sortino, Calmar.
2. Profit factor and expectancy.
3. Regime-specific hit rate.
4. Turnover and fee drag.
5. Sentiment signal coverage and staleness.

## 9) Known Risks + Mitigations

1. News latency and revision risk.
Mitigation: TTL/staleness gates and delayed confirmation windows.
2. Source manipulation/noise risk.
Mitigation: source quality weighting + dispersion feature + outlier clipping.
3. LLM hallucination risk.
Mitigation: deterministic features first, strict schema, fail-closed fallback.
4. Regime-shift decay.
Mitigation: drift monitors and automatic de-risk policy (`P2-07` dependency).
5. Provider contract drift.
Mitigation: contract tests on adapter payloads and schema-version alerts.

## 10) Execution Order Recommendation

1. Build `P3-01` and `P3-02` first (no live trading impact).
2. Run backtest ablations on sentiment features before agent wiring.
3. Implement `P3-03` in shadow-only mode.
4. Enable `P3-04` bounded policy integration.
5. Complete `P3-05` and enforce rollout gates.

## 11) Research References

1. OpenAI function calling guide: https://platform.openai.com/docs/guides/function-calling
2. OpenAI Structured Outputs guide: https://platform.openai.com/docs/guides/structured-outputs
3. OpenAI model docs (`gpt-5-mini`): https://platform.openai.com/docs/models/gpt-5-mini
4. OpenAI pricing page: https://openai.com/api/pricing/
5. DeepSeek tool calls: https://api-docs.deepseek.com/guides/tool_calls
6. DeepSeek thinking mode: https://api-docs.deepseek.com/guides/thinking_mode
7. DeepSeek reasoning model page: https://api-docs.deepseek.com/guides/reasoning_model
8. DeepSeek pricing/feature table: https://api-docs.deepseek.com/quick_start/pricing/
9. Alpha Vantage docs (`NEWS_SENTIMENT`): https://www.alphavantage.co/documentation/
10. NewsAPI endpoints docs: https://newsapi.org/docs/endpoints
11. NewsAPI `/everything`: https://newsapi.org/docs/endpoints/everything
12. Alternative.me Crypto Fear & Greed API: https://alternative.me/crypto/fear-and-greed-index/
13. GDELT DOC 2.0 API docs: https://api.gdeltproject.org/api/v2/doc/doc
14. FinBERT model card: https://huggingface.co/ProsusAI/finbert
15. FinBERT paper (ArXiv): https://arxiv.org/abs/1908.10063
16. Freqtrade strategy docs: https://www.freqtrade.io/en/stable/strategy-customization/
17. Freqtrade lookahead analysis docs: https://www.freqtrade.io/en/stable/lookahead-analysis/
