# Provenance — guard-judge-synthetic

> **Pack version:** v1 · **Imported:** 2026-11-12T17:11:58Z · **Dataset version:** v1

Written by `model-bench/scripts/import_decision_pack.py` (typed-decision-models v1.3 §3.4; plan4 §4.2 U3d): the dataset's eval split only, copied one-way from decision-models. `prompts/` is committed pack data the importer never writes. Re-importing needs a new `--pack-version`; an unchanged one is refused.

Items: 317 in stratum `eval`, 85 in stratum `side-check`.

| Origin | Destination | Source git SHA | Source SHA-256 |
|---|---|---|---|
| `decision-models/data/guard-judge/v1/items.jsonl` | `items.jsonl`, stratum `eval` (317 items) | `7a1efb177dd3a2fc2da12e5963ea472727ed686e` | `bf2679150745ed768f2952284102f47bed3574717e377e7ace062d438f06ba7a` |
| `decision-models/data/guard-judge/v1/splits.json` | eval membership | `7a1efb177dd3a2fc2da12e5963ea472727ed686e` | `fed2e6eb75c55703c0bf5d0f4cf5645a4615c4e2a30dfe7192f31517d2141310` |
| `decision-models/data/guard-judge/v1/generation.json` | `pack.json` `labels`; this file | `7a1efb177dd3a2fc2da12e5963ea472727ed686e` | `7db8b2dd73087feb2d3f4256daa8cb3f4eb86be1c3995583615f3d471696bdd2` |
| `model-bench/packs/guard-judge-understanding/items.jsonl` | `items.jsonl`, stratum `side-check` (85 items) | `— (model-bench)` | `9648831cda0fa4954e9260ad8c0603e0d8ed463296a135828ab7ee90d3ff4749` |

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
