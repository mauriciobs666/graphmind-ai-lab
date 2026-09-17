# Kaizen-team distillation — U5 (`qa-engineer`) gate review

> **Status:** active · **Owner:** `analyst` · **Tracks:** — (no backlog id; `docs/plans/kaizen-team-distillation-coordination.md`, unit U5)

## Scope & verdict

Reviewed the uncommitted working-tree diff produced by `cobb`'s standing kaizen-distillation pass
over the shared `kaizen_team` FalkorDB graph's `qa-engineer`-produced raw `:KaizenEntry` nodes
(`skills/agent-maintenance/SKILL.md` §5), unit U5 of
`docs/plans/kaizen-team-distillation-coordination.md`. Two files: `claude/qa-engineer/
qa-testing-techniques.md` (3 new sections) and `claude/qa-engineer/kaizen/history.md` (new dated
entry). Baseline is the real codebase and documents each promoted/discarded claim cites, not the
report's own narrative — every technical claim below was independently traced to current source,
not taken on cobb's word, per U4's lesson. I did not re-derive the disposition calls from the raw
graph entries themselves (they are already cleared, by design, before this gate — same limitation
U1–U4 operated under); I verified what remains: the promoted/discarded text against its cited
source, and the graph's post-clear state.

**Verdict: approve.** No blocker, no major. Nothing below rises even to a suggestion-worthy nit
that would change the text — this section exists to record what was checked and how.

**CPG:** not applicable — this is a documentation-only kaizen-distillation task (prose edits to
two Markdown files), with no code-level component to load a CPG for.

## Findings

None. Every checked item held up:

1. **Promotion 1 (`/v1/models` vs `/api/v0/models`)** — verified accurate.
   `opencode/docs/test-reports/devops-opencode-headless-report.md` §2 (DEF-1) documents the exact
   incident with matching figures (`loaded_context_length: 8192`, `max_context_length: 262144`,
   ~11,381-token overhead) and a not-yet-applied recommendation (report line 594-599: "Worth
   considering for the README's Prerequisites section"). `opencode/agents/tank/README.md:13-20`
   confirms the Prerequisites section still only names `curl http://localhost:1234/v1/models` —
   the `/api/v0/models`/`loaded_context_length` check was never added. The 16384-token minimum the
   new KB section cites matches `opencode/agents/tank/README.md:20` and
   `opencode/docs/manuals/local-llm.md:86` verbatim. Promoting the general technique rather than
   editing that README is correct: `qa-engineer`'s write guard (`claude/AGENTS.md`,
   `guard-doc-writes.sh` wrappers) scopes it to `docs/test-plans/*`/`docs/test-reports/*`, and
   `cobb`'s own wrapper doesn't cover an `opencode/` README either.
2. **Promotion 2 (`attest`/`run` staleness, merged)** — both merged mechanisms verified directly
   against current source, and more deeply than the diff itself claims:
   - `modelbench/cli.py:274-302` (`_gather_attested_fields`) confirms the `input()`-per-unset-field
     fallback exactly as described.
   - `modelbench/hostinfo.py:274-306` (`check_attestation_staleness`, `"compared"` branch)
     confirms the comparison is `runtimeName`/`runtimeVersion`/`residencySource` only — none of
     the four operator-attested fields (`ATTESTED_FIELD_NAMES`, `hostinfo.py:53-58`, which
     includes `lmStudioAppVersion`) enter that comparison.
   - I additionally traced that the staleness check is actually wired into `run` (the raw claim's
     load-bearing premise): `modelbench/runner.py:848-855` calls `check_attestation_staleness` and
     raises `RunRefused` on `stale`; `modelbench/cli.py:404-408` catches `RunRefused` and prints
     `model-bench: {exc}` to stderr with no traceback, returning `exc.exitCode` — confirming "a
     plain message, no traceback" precisely.
   - No CUDA-version or other machine-local figures appear anywhere in the diff — the claimed drop
     is genuinely clean, not smuggled into a different sentence.
3. **Promotion 3 (adversarial totality-contract lesson)** — `packs/tool-caller-shop-assistant/
   tools/sim.py:111-114` currently uses `self._handlers.get(safe_name)`, confirming the bug is
   fixed. `git log -S "self._handlers.get(name)" -- packs/tool-caller-shop-assistant/tools/sim.py`
   returns commit `3878493` ("fix(model-bench): TD-1 - dispatch() looks up sanitized tool name
   (U149)"), confirming this is a real, already-shipped fix, not an invented anecdote — the KB
   entry promotes only the reusable test-design lesson and correctly states the bug is fixed.
4. **The 3 discards** — all re-verified against their cited sources:
   - `falkor-chat/docs/test-reports/document-ingestion2-report.md:204-217` documents the boot
     `MemberIdCollisionError`, the `u1`/`u2` workaround, and the "plausibly intentional, not filed
     as a defect" judgment near-verbatim.
   - `falkor-chat/server/falkorchat/repository.py:164-180`'s `_escape_fuzzy_token` docstring cites
     "document-ingestion2 QA Defect 1" by name and describes the exact RediSearch metacharacter
     class this discard describes.
   - `claude/data-scientist/lm-studio-model-notes.md:45-68` ("A chat template can reject a SECOND
     system-role message outright") cites "unit U159" (line 68) and `convo.py:379-396`, matching
     U4's promotion target exactly; `modelbench/convo.py:379-396` (`_prologue_system_message`)
     confirms the collapse-to-one-system-message fix is real and matches the docstring's own
     framing. Genuine duplicate, correctly discarded rather than re-promoted.
5. **`kaizen/history.md`'s new entry** — inserted immediately above the existing 2026-09-10 entry
   (`history.md:5`, before line-original `## 2026-09-10`), no duplication. Arithmetic checks out:
   header claims 7 total / 4 promoted / 3 discarded; body lists 3 promoted bullets where the
   second bullet merges 2 raw entry IDs (1 + 2 + 1 = 4 promoted) and 3 discarded bullets (4 + 3 =
   7), matching the coordination doc's own dispatch-time count.
6. **Graph state** — independently re-ran both reads against `kaizen_team`:
   `MATCH (a:Agent {agentId:'qa-engineer'})-[:PRODUCED]->(k:KaizenEntry) RETURN count(k)` → `0`,
   and the legacy `MATCH (k:KaizenEntry {author:'qa-engineer'}) RETURN count(k)` → `0`. Matches the
   report exactly.

## What's solid

- Every promoted technique's technical claim is not just consistent with its cited source but
  independently reproducible from reading that source cold (the `/v1/models` distinction, the
  `attest`/`run` mechanics, the dispatch fix) — this pass shows no sign of U4's "unsupported
  anecdote carried verbatim" failure mode.
- The dropped-CUDA-figures judgment call (promoting only the code-verified general mechanism,
  discarding the machine-local specific figure) is exactly right and, unlike U4, was applied
  proactively rather than needing a gate round to catch it.
- Section style/structure (H2 lesson-as-heading, evidence paragraph, bolded `Technique:` callout)
  matches the file's existing four sections cleanly — no drift in convention.
- `qa-engineer.md` itself correctly left untouched (`git status` confirms only the two KB/history
  files changed) — none of the 4 promotions crossed the always-loaded-prompt bar, a reasonable
  call for environment/tooling techniques.

## Open questions

None.
