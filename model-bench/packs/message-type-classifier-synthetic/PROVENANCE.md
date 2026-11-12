# Provenance — message-type-classifier-synthetic

> **Pack version:** v1 · **Imported:** 2026-11-12T18:34:43Z · **Dataset version:** v2

Written by `model-bench/scripts/import_decision_pack.py` (typed-decision-models v1.3 §3.4; plan4 §4.2 U3d): the dataset's eval split only, copied one-way from decision-models. `prompts/` is committed pack data the importer never writes. Re-importing needs a new `--pack-version`; an unchanged one is refused.

Items: 290 in stratum `eval`, 0 in stratum `side-check`.

| Origin | Destination | Source git SHA | Source SHA-256 |
|---|---|---|---|
| `decision-models/data/message-type-classifier/v2/items.jsonl` | `items.jsonl`, stratum `eval` (290 items) | `967692e6b832ed1cf8769a310ccd358729e4f21b` | `0e336e3664c23e08795fc537d56d5afc3cc6ab9de29029751a569224b666b1f0` |
| `decision-models/data/message-type-classifier/v2/splits.json` | eval membership | `967692e6b832ed1cf8769a310ccd358729e4f21b` | `eed30c1376e23f68f7f65333dee51d790eb5a1a9ffa3cf3f2a651954a660cf8a` |
| `decision-models/data/message-type-classifier/v2/generation.json` | `pack.json` `labels`; this file | `967692e6b832ed1cf8769a310ccd358729e4f21b` | `011584f33c13c862d8bb13d6f70df1a580d8743ce714dffebaeeaf45eade11b4` |

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
