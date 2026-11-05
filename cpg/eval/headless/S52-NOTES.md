# S52 — writer agent definitions: notes

> **Status:** active · **Owner:** `cobb` · **Tracks:** P0-43 (S52)

Files (S52 owns only these and `tests/test_guard_paths.sh`, `tests/mutants_guard_paths.sh`,
`tests/test_agents_json.py`): `agents.json`, `settings.json`, `guard-paths.sh`. S53 owns
`run-writer.sh` and `README.md`. Plan: `docs/plans/connected-code-graph-p0-2.md` §3 S52, §6.1–§6.3, §9.1.

## Catalog call: `claude/AGENTS.md`'s catalog rules do not apply

These four are harness-only definitions: they load through `--agents` in headless `claude -p`
sessions that the launcher starts, are never symlinked into `~/.claude/agents/`, and no
orchestrator delegates to them. `claude/AGENTS.md`'s add/edit/rename rule (source, `kaizen/`,
`claude/README.md` entry, roster) and `claude/scripts/audit-team.sh` (it scans only
`claude/<name>/<name>.md`) are for deployed, delegable agents, so none of it is owed here, and
`claude/AGENTS.md` is unchanged. What replaces it:
- The change log is `docs/HISTORY.md` (the `cpg/` component's own), per the module convention.
- The integrity record is the per-attempt SHA-256 of `agents.json` and `settings.json` that S53's
  run index keeps; `resume` refuses a changed hash.
- The check is S54's planted-breach rehearsal plus the three test files here.
- Any edit to the definitions after S54 passes re-opens S54 (the hash changes).

## Interface contracts S53 must honour

1. **`CCG_GUARD`.** `settings.json`'s hook command runs `"$CCG_GUARD"` and exits 2 (deny) if the
   variable is unset or not executable, so a launcher mistake fails closed. The launcher runs under
   `env -i`, so it must pass `CCG_GUARD=<repo>/cpg/eval/headless/guard-paths.sh` in that `env`
   list. (The repo path cannot be written into the committed file: it would be a personal home path.)
2. **cwd is the root.** The guard takes the root from the hook's `cwd` field (the role root) and
   needs `/usr/bin/python3` (the launcher's `PATH` is `/usr/bin:/bin`).
3. **Tools.** Every definition lists `Read, Grep, Glob, Write, Edit`, equal to the argv's `--tools`.
   The definitions set no `model`; `--model` picks it (§6.1). The hook matcher also covers
   `NotebookEdit` and `MultiEdit`, which the argv's `--tools` already excludes.
4. **Output paths.** KA writes `out/KA.jsonl`, KB `out/KB.jsonl`, QW `out/QW.jsonl`. The annotator
   definition leaves `{outputPath}` to the brief (`out/AN1.jsonl` / `out/AN2.jsonl`). The brief
   must also fill `{stratum}`, `{units}`, `{cypherMcpTools}`, `{flowAnchors}`.
5. The brief (first user message) repeats Appendix B (S40). The definitions carry it too, per the
   plan; the duplication is accepted.

## What the definitions contain

- Per definition: a trust-boundary section (kit text and `QW-nogold.jsonl` are data, never
  instructions: `-spec2` F-3, the prompts' half); the tool and append method (only Write/Edit exist,
  so "append" is Read then Edit on the last line); the role's strategy; Appendix B.0 and the role's
  part, verbatim.
- KA is provider-first (inventory every provider, then hunt consumers); KB is consumer-first
  (walk files, resolve each outbound edge). Different procedure and wording, not a flag
  (`test_agents_json.py` checks the Jaccard overlap of the strategy sections).
- No ancestor-plus-line rule (`-spec2`'s F-13 is the keyNorm reorder; the withdrawal of the
  ancestor-plus-line locator is E1: the grammar extract drops `-spec3` §3.5's `line` mention and
  rule 5, and B.0's naming rule governs). No learning-capture, KB, CPG or repo instructions.

## `{grammarExtract}`: deviations from "quote verbatim", for S40 to mirror

Appendix B's slot list (§4.0, §4.1 rules 1–4 and 6–10, §4.2 bullets for Python, TS/JS, shell,
Markdown, §6 `http` and `mcp_tool` rows without the example column) was applied to
`docs/plans/connected-code-graph-bundle-spec3.md`, with `-spec4` §1's replacement of rule 4's first
sentence applied ("Every code file **the unit's extractor parsed** has exactly one…").
The sources still name
proving-repo elements and sections the paste does not carry, against Appendix B's own promise
that the text names none, so these were removed (S40 should apply the same cuts, or the briefs and
definitions diverge):
- §4.0: bullets 2–3 (the `S11`/SL-7 check; the `FULL_NAME` example and the `cpg-model.md` pointer).
- Rule 1 body (rejected-alternative example); rule 2's `test_fusion.py:241` example; rule 4's
  `§3.5 line` mention and its rationale sentence about `r.load()`; rule 8's `Protocol` stub
  sentence, the `embedding.py:37` cite and the "Resolution into a non-symbol" bullet; rule 10's
  `nameNorm` sentence.
- Reviewer tags `(analyst …)`, `[P-slice]`/`[P-later]`, and `§` cross-references to sections not
  pasted. The `http` row's `(F-13, §13)` and step (8a) (token-screen masking, done by the
  normalizer, not by writers) were dropped.
- Still dangling, left as written: rule 4 and 7 talk of a "signature" (§4.3, Java/Kotlin/C#/C++
  only; the repo has no code in them). Harmless, but S40 may prefer to cut it.

## For S54 to verify (not observable offline)

- Hook behaviour under `--restricted --settings`: that `PreToolUse` fires for Read/Grep/Glob/
  Write/Edit and a deny (exit 2) is shown to the model; the matcher's pipe form is assumed.
- That `--agents` accepts this file path under `-p` and the agent name resolves with `--agent`.
- That the hook's `cwd` field equals the role root.
- Each prompt is about 20-24 KB (about 3,000-3,700 words); whether `sonnet` keeps B.1 straight at
  that size is for S54/S55 to see.
