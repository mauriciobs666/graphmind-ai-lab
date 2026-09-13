# devops-opencode-headless — implementation review (G1, general correctness/quality)

> **Status:** archived · **Owner:** `analyst` · **Tracks:** — (M0)

## Scope & verdict

Reviewed U2's and U3's delivered artifacts (`cobb`) against the approved plan
`opencode/docs/plans/devops-opencode-headless.md` (v3) — the general correctness/quality/
coherence pass (G1 in the coordination ledger), explicitly **not** the security lens
(`security-expert`'s parallel Pass 3 owns glob-smuggling / permission-table adversarial analysis).
In scope, all read and independently exercised where the brief asked for it:
`claude/devops/{devops-persona.md,devops.md,scripts/sync-persona.sh}`, `claude/README.md`,
`opencode/agents/tank/opencode.json`, the three outer scripts
(`health-check.sh`/`bring-up.sh`/`tear-down.sh`), `opencode/agents/tank/README.md`,
`opencode/{AGENTS.md,CLAUDE.md}`, and root `AGENTS.md`'s plan-directed edits. Also weighed the
open question about `opencode/docs/` having no `HISTORY.md` yet.

**Verdict: approve with suggestions.** No blockers, no majors. Two low-stakes findings below —
neither breaks anything or drifts from the plan's actual requirements; both are polish.

**CPG:** not applicable — no `cpg_opencode`/`cpg_claude` graph exists (confirmed by the plan §1
and the Pass-1/Pass-2 security review; greenfield config/prompt authoring, no call graph to
analyze).

## Verification performed (not just read)

- **Byte-identity of the shared-persona span.** Extracted the text between
  `<!-- SHARED-PERSONA:BEGIN -->`/`<!-- SHARED-PERSONA:END -->` in `claude/devops/devops.md` with
  `awk` and diffed it against `claude/devops/devops-persona.md`: **identical** (`diff` empty,
  matching md5sum `00f8950d33171f99c4d3be255f11816e`). Independent of `teco`'s own prior diff per
  the ledger.
- **`sync-persona.sh` idempotency, on the real files.** Ran it twice against the actual working
  tree: `git diff --stat -- claude/devops/devops.md` reported the identical `10 insertions(+), 3
  deletions(-)` before, after run 1, and after run 2 — a true no-op both times, and re-confirmed
  the marked span was still byte-identical to `devops-persona.md` afterward. The repo is left
  exactly as U2 delivered it (no new dirt introduced).
- **`sync-persona.sh` failure paths, against scratch copies (not the real files).** Missing
  `BEGIN` marker, duplicated `BEGIN` marker, and a missing persona source file all exit `1` with a
  specific, actionable message (`found 0`, `found 2`, `missing persona source file: <path>`) —
  matches plan §3.1/§4 step 3's "fail loudly" requirement exactly.
- **Config-shape spot checks:** `python3 -m json.tool` on `opencode.json` and `environments.json`
  (both valid); `bash -n` on all three outer scripts and `sync-persona.sh` (clean); read the full
  resolved `permission.bash` list in `opencode.json` and hand-checked its tier ordering against
  plan §3.2 — matches exactly (default deny → 7 read-only allows → 3 wrapper-script allows → 5
  destructive-catalog denies → 4 repeated-scoping-flag denies → 6 metacharacter denies), consistent
  with `teco`'s independent `opencode debug agent tank` check already logged in the ledger (not
  re-run here — no reason to duplicate that specific check per the brief).
- **Prompt addendum vs. plan §3.4's five required topics**, read side by side: headless context,
  destructive-ops override, FR-7 ambiguity clause, bring-up/teardown protocol, default
  health-check meaning — all five present in `opencode.json`'s `agent.tank.prompt` field, worded
  consistently with the plan's own language.
- **Model/provider block vs. `deprecated/opencode/agents/severino/opencode.json`'s precedent** —
  same `provider.lmstudio` shape (`npm`, `name`, `options.baseURL`/`apiKey`, a `models` map), no
  stray `name` field on the agent (correctly avoided per both files' own documented gotcha).

## Findings

### Minor: root `AGENTS.md`'s `deprecated/` bullet still points at the requirements doc for a "new" `tank`, not its now-built home

`AGENTS.md:68-70` — "...retired to clear the way for the new headless `tank` agent
(`opencode/docs/requirements/devops-opencode-headless.md`)." This sentence predates U3 (it was
already there before this coordination started) and wasn't one of plan §4 step 11's two edits, nor
the one extra stale-text fix `cobb` flagged as beyond the plan's literal ask (that fix touched the
"Working in this repo" bullet a few lines down, confirmed correct). It isn't factually wrong — the
requirements doc still exists and still accurately describes what motivated the retirement — but
now that `tank` is built and has its own `opencode/agents/tank/README.md`, citing only the
requirements doc here reads as slightly dated next to the freshly-updated `opencode/` bullet three
lines above it (which now correctly points at `opencode/agents/tank/README.md`). Low stakes: no
reader is misled, nothing breaks. Suggested improvement: when this file is next touched, swap or
add the citation to `opencode/agents/tank/README.md` (or drop the parenthetical entirely, since the
`opencode/` bullet above it now already covers this) — not worth a dedicated edit on its own.

### Nit: the "hook parenthetical folded back into the Guarded-ops bullet" judgment call resolved well, but is worth a one-line self-note

Plan §3.1 literally says devops.md's Claude-only tail should have "the hook parenthetical folded
back into the Guarded-ops bullet" — read most literally, that would mean re-inserting the
parenthetical into the *same* Guarded-ops bullet inside the marked span, which is impossible
without breaking the span's required byte-identity to `devops-persona.md`. `cobb` resolved this the
only way that satisfies both constraints: a new bullet under the new `## Claude Code specifics`
heading, labeled "Guarded ops enforcement (extends the persona's Guarded-ops bullet above)",
carrying the exact same parenthetical text verbatim (confirmed identical against the pre-split
`devops.md`, `git diff -- claude/devops/devops.md`). No information lost, reads coherently, heading
level consistent with the rest of the document. This is genuinely the flagged judgment call working
out fine — noted here only because the ledger asked for a coherence read on exactly this point, not
because anything needs fixing.

## What's solid

- **The persona split mechanism is correct end-to-end**, not just claimed: byte-identity holds,
  `sync-persona.sh` is genuinely idempotent and genuinely fails loudly on all three specified
  failure shapes, and `devops.md`'s frontmatter is untouched (confirmed via diff — no lines above
  the body changed).
- **The tank config and outer scripts match the plan precisely** — tool flags, permission-table
  tier structure and ordering, the five-topic prompt addendum, the `environments.json` shape, the
  `state/` gitignore pattern, and the outer-script contract (thin `cd` + fixed message template,
  `exec opencode run --agent tank "..."`) all line up with plan §3.2/§3.4/§4 step 9 with no drift
  found.
- **`opencode/agents/tank/README.md` is complete against plan §4 step 8's checklist** —
  prerequisites, all three invocation shapes, adding a second environment, the state marker and its
  freshness override, and an explicit persona-editing warning are all present and accurately
  described (each claim in it was checked against the actual `opencode.json`/scripts, not just read
  at face value).
- **`opencode/AGENTS.md`/`CLAUDE.md` and `claude/README.md`'s new sentence** are accurate, cite
  rather than restate, and are well within the context-file size bar (148 words, longest line 97
  chars — nowhere near the ~2,500-word/~700-char smells).
- **Root `AGENTS.md`'s two plan-directed edits plus the one extra fix** are all accurate against
  the repo's actual current state — every path they now cite (`opencode/AGENTS.md`,
  `opencode/agents/tank/README.md`) exists and is current.

## Open question — the missing `opencode/docs/HISTORY.md`

No problem seen with `cobb`/`teco`'s reasoning. Root `AGENTS.md`'s own documentation convention
treats `HISTORY.md` as a lookup-only document that "may grow without bound" and is never revised in
place the way a living, read-whole document is — so an entry written before G1/G2/U4 land would
either need a correction appended later (if any gate finds something) or would misleadingly read as
"delivered and settled" before it actually is. Waiting to write one accurate entry covering U1–U4
once the chain closes avoids both problems and costs nothing (nothing currently depends on
`opencode/docs/HISTORY.md` existing). This is consistent with how the convention treats `HISTORY.md`
generally, not a deviation from it.
