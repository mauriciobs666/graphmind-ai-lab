# S52 — writer agent definitions: notes

> **Status:** active · **Owner:** `cobb` · **Tracks:** P0-43 (S52)

Files (S52 owns only these and `tests/test_guard_paths.sh`, `tests/mutants_guard_paths.sh`,
`tests/test_agents_json.py`, `tests/mutants_agents_json.sh`): `agents.json`, `settings.json`, `guard-paths.sh`. S53 owns
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

1. **`CCG_GUARD`.** `settings.json`'s hook command runs `"$CCG_GUARD"` (no `exec`) and maps every
   exit code other than 0 to exit 2, and denies if the variable is unset or not executable, so a
   launcher mistake or a crashing guard fails closed (only exit 2 blocks a hook). The launcher runs under
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

## `{grammarExtract}`: the cuts (S40 must mirror them; the extract is byte-identical in all four)

Source: `docs/plans/connected-code-graph-bundle-spec3.md` §4, §6, cut per the Appendix B "Rules for
the pasters" bullets (data-scientist's M-1 ruling). The extract is 12,145 bytes, sha256
`5202c983c7ced1bc2b1ef166fd2b8a77bfbf3d444f1880e3a5afdd06e009f3ff`, pinned in `test_agents_json.py`.
- **Ruling cuts (M-1):** rule 7 deleted whole; the line reads "(Rules 5 and 7 are not used in this
  brief.)"; `http` step (7) reads "becomes `{}`. The syntaxes include `{id}`, `{id:int}`, `:id`,
  `<id>`, `<int:id>`, `[id]`." (no catch-all clause). The grammar block's `dup` line and `<anon>` base stay.
- **Name/section cuts (Appendix B: no proving-repo element, no unpasted section):** §4.0 bullets 2-3;
  rule 1 body; rule 2's `test_fusion.py:241` example; rule 4's `§3.5 line` mention and its `r.load()`
  rationale; rule 8's `Protocol` sentence, the `embedding.py:37` cite and the "Resolution into a
  non-symbol" bullet; rule 10's `nameNorm` sentence; `(analyst …)`, `[P-slice]`/`[P-later]` tags and
  `§` references to unpasted sections; the `http` row's `(F-13, §13)` and step (8a).
- Left as written: the "signature" mention in rule 4's contract bullet and rule 8 (Java/Kotlin/C#/C++ only).

## Gate Pass 1 dispositions

- **B-1 fixed.** `guard-paths.sh` runs its body in `main()` under `except Exception` that denies
  (exit 2), and `deny` is itself surrogate-safe. `settings.json` runs the guard without `exec` and
  turns any non-zero exit into 2 (the reviewer's form, executed against stub exits 1/3/126/139 and a
  lone-surrogate input: all 2; exit 0 stays 0). Tests: an exit-code probe over 8 tools x 5 keys x 13
  value classes plus 5 cwd classes (525 cells; red before the fix with 48 rc=1 cells, green after);
  four stub-exit tests of the settings command. Mutants added: guard `except` narrowed (killed);
  settings command reverted to `exec` (killed); settings command ignoring the exit code (killed).
- **M-1 resolved** by the ruling above (no Appendix B edit by S52).
- **m-1 fixed:** rule 4's first sentence is back to spec3's "Every code file" (it lay in the extract,
  outside Appendix B's bytes); spec4 §1's "extractor parsed" wording is the normalizer's, not a writer's.
- **m-2 fixed:** the extract's length and sha256 are pinned and the four copies must be byte-identical.
  S40's brief should be checked against the same hash.
- **m-3 fixed:** `tests/mutants_agents_json.sh` (11 mutants: edited B.0 word, edited B.1 word, foreign
  part pasted, fence sentence dropped, ancestor rule added, extract rule edited, rule 7 reinstated,
  extracts diverging, KB instruction added, extra tool, identical strategies).
- **m-4 fixed:** the checklist is "in your context only (never in a file)"; no scratch file in `out/`.
- **Residual, for S54:** the gate's list of ten harness observations (hook fires, deny reaches the
  model, non-0/2 exits are denied, `cwd`, key names, exact tool list, glob forms, no `$VAR`
  expansion, `--agents` path, `CCG_GUARD` inheritance) stays S54's.

## For S54 to verify (not observable offline)

- Hook behaviour under `--restricted --settings`: that `PreToolUse` fires for Read/Grep/Glob/
  Write/Edit and a deny (exit 2) is shown to the model; the matcher's pipe form is assumed.
- That `--agents` accepts this file path under `-p` and the agent name resolves with `--agent`.
- That the hook's `cwd` field equals the role root.
- Each prompt is about 20-24 KB (about 3,000-3,700 words); whether `sonnet` keeps B.1 straight at
  that size is for S54/S55 to see.
