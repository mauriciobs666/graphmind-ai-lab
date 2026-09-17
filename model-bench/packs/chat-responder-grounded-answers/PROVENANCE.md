# Provenance — chat-responder-grounded-answers

> **Pack version:** 0.1.0 · **Generated:** 2026-09-17T17:34:53Z

`items.jsonl`'s 30 items are **authored content, not copied**: each `question` is a hand-written
paraphrase (never a verbatim corpus message, FR-19) drawn from one or more threads of the
121-message corpus at `packs/embedder-graphrag-retrieval/corpus.jsonl` (itself copied one-way from
`falkor-chat` at S3), and each item's `context` list carries passage text copied — verbatim or
near-verbatim — directly into the item's own JSON row, never a live `docId`/path reference back
into that other pack's directory (`docs/plans/small-model-benchmarking-s7-spec.md` §2.6: a live
reference would be unhashed, unversioned, and would violate this component's own standalone rule,
FR-23). This pack is therefore fully standalone: nothing in `modelbench` resolves an item's
`context` against `embedder-graphrag-retrieval` at run time.

| itemId | draftedBy | basedOn | verifiedBy | verifiedAt |
|---|---|---|---|---|
| cr-01 | coder | eval-payment-timeout-incident-001, eval-payment-timeout-incident-003 | Mauricio Stefani | 2026-09-17 |
| cr-02 | coder | eval-payment-timeout-incident-006, eval-payment-timeout-incident-007 | Mauricio Stefani | 2026-09-17 |
| cr-03 | coder | eval-payment-timeout-incident-008, eval-payment-timeout-incident-009 | Mauricio Stefani | 2026-09-17 |
| cr-04 | coder | eval-search-latency-incident-001, eval-search-latency-incident-005 | Mauricio Stefani | 2026-09-17 |
| cr-05 | coder | eval-search-latency-incident-007 | Mauricio Stefani | 2026-09-17 |
| cr-06 | coder | eval-search-latency-incident-001, eval-search-latency-incident-008 | Mauricio Stefani | 2026-09-17 |
| cr-07 | coder | eval-notification-service-database-choice-009, eval-notification-service-database-choice-010 | Mauricio Stefani | 2026-09-17 |
| cr-08 | coder | eval-notification-service-database-choice-011 | Mauricio Stefani | 2026-09-17 |
| cr-09 | coder | eval-notification-service-database-choice-004, eval-notification-service-database-choice-009 | Mauricio Stefani | 2026-09-17 |
| cr-10 | coder | eval-notification-service-retry-strategy-003 | Mauricio Stefani | 2026-09-17 |
| cr-11 | coder | eval-notification-service-retry-strategy-008, eval-notification-service-retry-strategy-009 | Mauricio Stefani | 2026-09-17 |
| cr-12 | coder | eval-notification-service-retry-strategy-009 | Mauricio Stefani | 2026-09-17 |
| cr-13 | coder | eval-oauth-token-refresh-bug-003 | Mauricio Stefani | 2026-09-17 |
| cr-14 | coder | eval-oauth-token-refresh-bug-008, eval-oauth-token-refresh-bug-007 | Mauricio Stefani | 2026-09-17 |
| cr-15 | coder | eval-oauth-token-refresh-bug-001, eval-oauth-token-refresh-bug-008 | Mauricio Stefani | 2026-09-17 |
| cr-16 | coder | eval-session-timeout-policy-001 | Mauricio Stefani | 2026-09-17 |
| cr-17 | coder | eval-session-timeout-policy-007, eval-session-timeout-policy-008 | Mauricio Stefani | 2026-09-17 |
| cr-18 | coder | eval-session-timeout-policy-005 | Mauricio Stefani | 2026-09-17 |
| cr-19 | coder | eval-hr-onboarding-checklist-002, eval-hr-onboarding-checklist-003 | Mauricio Stefani | 2026-09-17 |
| cr-20 | coder | eval-hr-onboarding-checklist-005, eval-hr-onboarding-checklist-008 | Mauricio Stefani | 2026-09-17 |
| cr-21 | coder | eval-q3-product-roadmap-planning-007 | Mauricio Stefani | 2026-09-17 |
| cr-22 | coder | eval-q3-product-roadmap-planning-005 | Mauricio Stefani | 2026-09-17 |
| cr-23 | coder | eval-office-relocation-logistics-003, eval-office-relocation-logistics-007 | Mauricio Stefani | 2026-09-17 |
| cr-24 | coder | eval-office-relocation-logistics-008 | Mauricio Stefani | 2026-09-17 |
| cr-25 | coder | eval-database-backup-policy-003, eval-database-backup-policy-004 | Mauricio Stefani | 2026-09-17 |
| cr-26 | coder | eval-database-backup-policy-008, eval-database-backup-policy-010 | Mauricio Stefani | 2026-09-17 |
| cr-27 | coder | eval-customer-support-escalation-process-007 | Mauricio Stefani | 2026-09-17 |
| cr-28 | coder | eval-customer-support-escalation-process-001, eval-customer-support-escalation-process-009 | Mauricio Stefani | 2026-09-17 |
| cr-29 | coder | eval-on-call-rotation-schedule-001, eval-on-call-rotation-schedule-003 | Mauricio Stefani | 2026-09-17 |
| cr-30 | coder | eval-on-call-rotation-schedule-008, eval-on-call-rotation-schedule-009 | Mauricio Stefani | 2026-09-17 |
