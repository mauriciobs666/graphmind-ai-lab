# Provenance — join-in-judge-synthetic

> **Pack version:** v1 · **Imported:** 2026-11-12T17:58:21Z · **Dataset version:** v2

Written by `model-bench/scripts/import_decision_pack.py` (typed-decision-models v1.3 §3.4; plan4 §4.2 U3d): the dataset's eval split only, copied one-way from decision-models. `prompts/` is committed pack data the importer never writes. Re-importing needs a new `--pack-version`; an unchanged one is refused.

Items: 200 in stratum `eval`, 0 in stratum `side-check`.

| Origin | Destination | Source git SHA | Source SHA-256 |
|---|---|---|---|
| `decision-models/data/join-in-judge/v2/items.jsonl` | `items.jsonl`, stratum `eval` (200 items) | `ee643501c419c596cb0d4cccf60a924abec29e64` | `04c6c5da211d34f6a60ecb82deedb333db2a71f9939b69e0250245e24b8639de` |
| `decision-models/data/join-in-judge/v2/splits.json` | eval membership | `ee643501c419c596cb0d4cccf60a924abec29e64` | `2788edb3b9a765a83366a6431bb524460df8d82ffcf3e4828ecea7df626f42b7` |
| `decision-models/data/join-in-judge/v2/generation.json` | `pack.json` `labels`; this file | `ee643501c419c596cb0d4cccf60a924abec29e64` | `5920b356b0e57c2b240b92f14238c7211a9ef676e5bfce1b5d74af65790bbe98` |

## Labeler (`labels.labeler`)

- **provider:** `anthropic`
- **model:** `claude-sonnet-5-5`
- **surface:** `claude-code-subagent`
- **effort:** `high`

## Generator (`labels.generator`)

- **provider:** `anthropic`
- **model:** `claude-opus-4-8`
- **surface:** `claude-code-subagent`
- **effort:** `medium`

## Labelling

- **labelBatchCap:** `50`
