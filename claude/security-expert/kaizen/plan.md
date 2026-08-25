# Kaizen — Improvement Plan: security-expert

> Forward-looking backlog for the `security-expert` agent.
> Status: 🔵 proposed · 🟡 in-progress · ✅ done (then moved to history.md) · ⚪ rejected/deferred
> Last reviewed: 2026-08-20

## Active

| ID | Added | Priority | Status | Summary |
|------|------------|----------|--------|---------|
| K-001 | 2026-08-20 | high | 🔵 | First-run shakedown: a real code-security review (with a CPG-backed component), a real agent/prompt-safety review, and — separately, with explicit human sign-off — a first supervised local exploitation attempt to validate the FR-10 approval loop end to end. |
| K-002 | 2026-08-20 | medium | 🔵 | The `guard-exploitation-approval.sh` pattern catalog (tool names + network-client/local-marker check) is a best-effort backstop, not exhaustive — revisit it after the first few real exploitation-shaped invocations to see what it missed or over-flagged, same "grows via ad hoc addition" convention as `guard-destructive-ops.sh`'s `pipeline.sh --reset` case. |
| K-003 | 2026-08-20 | low | 🔵 | If a second offensive-security-shaped agent is ever added to this team, extract `guard-exploitation-approval.sh`'s logic into a shared `scripts/guard-exploitation.sh` core (mirroring the 2026-07-11 doc-write-guard consolidation) rather than duplicating it — not before, one consumer doesn't earn the indirection yet. |
| K-004 | 2026-08-20 | medium | 🔵 | Extend `guard-exploitation-approval.sh`'s `matcher` (or add a second hook) to also watch `WebFetch` calls, not just `Bash` — the prompt-level FR-10 ritual now explicitly covers `WebFetch` (2026-08-20 fix pass on `analyst`'s review), but the harness backstop still doesn't. Needs a pattern for "does this WebFetch URL look like a live local/dev target with an exploitation-shaped query string" — genuinely harder to pattern-match reliably than a Bash command, so scope it carefully rather than rushing a noisy first cut. |

## Parking lot / ideas

- **Judged and kept, do not re-litigate (2026-08-24, C4 lint).** The "never runs automatically"
  rule is stated twice on purpose — once in the §"Your four review lenses" intro, once as
  §Boundaries' "**No standing gate, ever.**" A class-7 dedup sweep will flag it; both times it
  should be kept. They are two decision points the agent genuinely stands at, per the plan's
  finding 5 ("the test is not *is this said twice* but *is it needed twice*"): the lens-catalogue
  statement blocks **self-triggering** while reading the catalogue ("I'm doing a code review, lens 1
  applies, I'll add a security section"), and its distinctive payload is the anti-volunteering rule
  plus the four enumerated workflows; the Boundaries statement blocks **claiming standing
  authority** ("my review gates cobb's promotion"), and its distinctive payload is the enumerated
  callers plus its structural parallelism with the three sibling boundary bullets. Same fact, two
  inferences.

- **Judgment calls made at creation, worth a second look on first real use (2026-08-20, `cobb`
  design pass, `claude/docs/requirements/security-expert.md`):**
  - Chose **not** to layer FR-10's approval gate on the existing shared
    `scripts/guard-destructive-ops.sh` core — built a standalone
    `guard-exploitation-approval.sh` instead, reasoning that shared-state-destruction literals
    (`GRAPH.DELETE`, `FLUSHALL`, volume wipes) and offensive-security-tool/network-exploitation
    patterns are different hazard classes with different maintainers and different growth rates.
    If `analyst` or the stakeholder judges this wrong (i.e. the two catalogs should have been one
    script from the start), it's a mechanical merge, not a redesign.
  - The exploitation guard's "no visible local marker → ask" rule for `curl`/`wget`/`nc`/`ssh`
    will false-positive on any legitimate non-exploitation network use (e.g. `curl`ing a project
    doc instead of using `WebFetch`) — accepted deliberately, since over-asking is the stated safe
    failure direction for this specific capability, and the prompt already steers the agent to
    prefer `WebFetch`/`WebSearch` for non-exploitation external lookups.
  - Picked a **slug-collision convention** for FR-11's review-doc path (fold a distinguishing word
    into the topic slug, e.g. `executor-security`, rather than inventing a new `-security` role
    suffix) because the doc-reference convention's role set (`(none)` · `-coordination` · `-ml` ·
    `-graph` · `-rca` · `-impl` · `-report`) is stated as **closed** in root `AGENTS.md`, and
    expanding it is a bigger repo-wide governance call than this single-agent design task was
    authorized to make. If security reviews turn out to collide with `analyst` reviews often
    enough to be a real friction point, that's the case for actually proposing a `-security` role
    through a `tico` interview, not silently adding one here.
  - Did **not** add a `security-expert:qa-engineer` boundary pair (only `analyst`/`cobb`/`devops`,
    the three the requirements doc's FRs actually name) — FR-10's exploitation and
    `qa-engineer`'s black-box acceptance testing are adjacent but the requirements doc never draws
    that boundary explicitly; add the pair later if a real handoff pattern between the two
    emerges.
- No worked example of a good findings report yet (all four lenses) — link one here once a first
  real review lands, per K-001.
