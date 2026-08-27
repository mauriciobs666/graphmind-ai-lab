# LLM Provider & Model Configuration — Feature Requirements (correction)

> **Status:** archived · **Owner:** `tico` · **Tracks:** K-042 (M4), K-045 · **Last updated:** 2026-08-26
> **Supersedes:** [`llm-provider-config.md`](./llm-provider-config.md) (FR-10/AC-8 wording only)

## Why this document exists

`llm-provider-config.md` is `archived` — K-042 was built and shipped against it, and per root
`AGENTS.md`'s collision rules the only edit an archived document permits is a header pointer, not
a body correction. Two of its clauses have since drifted from the shipped, QA-confirmed behavior:
**FR-10** and **AC-8** both describe an unresolvable/failing use-time model as making "the run
suspend[s]." The actual shipped behavior (confirmed by K-042 Landing 2's live QA acceptance pass,
AC-8, `docs/test-reports/llm-provider-config2-report.md`, and the D-2 fix, `docs/HISTORY.md`
2026-08-11) is that the run **fails**, with the cause recorded (`status: 'failed'`, an `error`
field) — the same terminal-failure vocabulary the executor uses for every other drive-time fault,
not the `human`/`wait` "suspend and wait for a signal" semantics "suspends" evokes. `tico` flagged
the drift while archiving the original document (`docs/HISTORY.md` 2026-08-11); filed as K-045 and
closed here, 2026-08-26.

This is **not** a new requirements interview — nothing about the feature's intent, scope, or
behavior changed. It is a factual correction against already-shipped, already-accepted behavior,
so it carries no open questions and is filed `archived` on creation. Everything else about the
feature — intent, user stories, FR-1..FR-9 and FR-11..FR-20, out-of-scope, AC-1..AC-7 and
AC-9..AC-13, the decision log — is unchanged and stays authoritative in the original document.

## Corrected text

**FR-10 (was):** "An unresolvable model encountered at use time fails loudly — the run suspends
(or the reply fails) with an error stating what could not be resolved. It never silently falls
back to another model."

**FR-10 (corrected):** An unresolvable model encountered at **use time** fails loudly — the run
**fails, with the cause recorded** (or the reply fails) with an error stating what could not be
resolved. It never silently falls back to another model.

**AC-8 (was):** "Given a model that resolves at publish but fails at call time, when a run reaches
it and no fallback chain applies, then the run suspends with an error naming what failed — and no
other model is used in its place."

**AC-8 (corrected):** Given a model that resolves at publish but fails at call time, when a run
reaches it and no fallback chain applies, then the run **fails, with the cause recorded in its
error field** — and no other model is used in its place.

## Decision log

2026-08-26 — `tico` closed K-045: corrected FR-10 and AC-8's "the run suspends" wording (both
carried the identical stale claim) to match the shipped `failed`-with-cause behavior, via this
successor document per the archived-original collision rule rather than an in-place edit.
