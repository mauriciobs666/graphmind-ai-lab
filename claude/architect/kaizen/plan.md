# Kaizen — Improvement Plan: architect

> Forward-looking backlog for the `architect` agent.
> Status: 🔵 proposed · 🟡 in-progress · ✅ done (then moved to history.md) · ⚪ rejected/deferred
> Last reviewed: 2026-08-25

## Active

*(no active items)*

## Parking lot / ideas
- **Live-probe seam check (parked, 2026-08-09 — ex-inbox entry 4).** Before scheduling a plan step that runs a live probe/experiment, check the target actually has a graph/tenancy/environment seam that makes the probe isolated (e.g. a throwaway `ws:<probe>` workspace) — surfaced designing K-031 (falkor-chat's `publish_def` writes to a hardcoded `reference` graph with no per-workspace override; `materialize_snapshot`'s shared query constant against a throwaway `ws:` graph was the workaround). Judged narrow/single-occurrence, not promoted to Guardrails — revisit if a second instance turns up.
- A short self-review checklist before delivering a plan (every step concrete & file-specific, alternatives recorded, risks listed, handoff summary present) — and, since 2026-07-27, the canonical header block present and its `Status:`/`Owner:`/`Tracks:` filled.
- **`architect` owns one recurring flip it isn't told about yet (noted 2026-07-27).** Root `AGENTS.md`'s routing table makes the architect the performer of the `Status: archived` flip on `plans/<slug>.md` at milestone close, on `teco`'s coordination. Today that reaches the agent only through the closing unit's brief; if a close ever ships with plans left `active`, the fix is one line in this prompt.
- Optionally delegate wide codebase sweeps to the Explore agent by default for large repos.
- Extend `hooks/guard-plan-doc-writes.sh` to cover Bash write patterns (`sed -i`, `>` redirects, `git commit`, package installs) **only if** the prompt-guarded Bash ever proves leaky in practice — deliberately left out on 2026-07-08 (see history).
