# Provenance — nlq-structured-query

> **Pack version:** 1.0.0 · **Generated:** 2026-09-12T01:12:50Z

Data files below are copied one-way from `falkor-chat/` by `model-bench/scripts/refresh_golden.py` (plan §3.1 point 3, D1: "copy the data, clean-build the code"). Re-running the importer requires bumping `pack.json`'s `packVersion` by hand first — a content-hash-changing edit under an unchanged version number is refused.

| Origin | Destination | Source git SHA | Source SHA-256 | Copied at |
|---|---|---|---|---|
| `falkor-chat/server/tests/eval/nlq_golden_set.jsonl` | `items.jsonl` | `d58125ad3164c4f589dcc135ee53c89daa3b2d93` | `f198290dab95ce8fd2f95204b6d24b195846e078d4c08fd0a959e81d7b02e0da` | 2026-09-12T01:12:50Z |
| `falkor-chat/scripts/seed_catalog.sh` | `tables.json#catalog` | `14891c94f9d4cefeec4adbd873697dc5b11b1fb7` | `58d80fcc93cd20db49cc66e28bcf090d92771e0916285b4356124bbbe18045b7` | 2026-09-12T01:12:50Z |
| `falkor-chat/server/falkorchat/querygen.py` | `schema.json` | `3d014975aca6244ab78c589caadedd39db5a6392` | `6276c28a69c5d8ed469efb3730478d6769fd9d0a4f5f2d6d71b9f911e90297b6` | 2026-09-12T01:12:50Z |

**Pending (Step 4, `docs/plans/small-model-benchmarking-s4-spec.md` §7):** `tables.json`'s
`"knowledge_base"` half (a human-in-the-loop `ws:nlq-eval` live snapshot, §2.6) has not been
written yet, so it carries no row in the table above — `refresh_golden.py --pack
packs/nlq-structured-query --check-tables-shape --source-git-sha <sha>` adds it once that
snapshot is taken. `items.jsonl`'s `"answerable"` stamp (`--stamp-answerability`, over
`reference_specs.json`, §6) is pending for the same reason: it needs the live snapshot data to
execute against. `reference_specs.json` itself (40 hand-authored entries) is this step's own
deliverable and does not get a row here — it is source content this pack ships, not an origin
copied from `falkor-chat/`. **Note for whoever runs Step 4:** verifying the 40 reference specs
against the real `ws:nlq-eval` snapshot (this session used the one already taken by `teco` this
same day) confirms `nlq-34`/`nlq-35`/`nlq-36`/`nlq-37` (relationship-traversal) AND
`nlq-38`/`nlq-39` (conflicting-facts) all fail Layer B (`SchemaViolationError`) — **six** items
stamp unanswerable, not the four the plan's own text names. See this unit's delivery report for
the full finding; `--stamp-answerability`'s own live run should reproduce the same six against
the official `tables.json`, not silently reconcile down to four.
