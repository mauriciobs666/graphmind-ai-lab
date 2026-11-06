# Connected code graph P0 — ground-truth report

> **Status:** active · **Owner:** `qa-engineer` · **Tracks:** C-1001 P0 (S54/S45–S47)

CPG: not applicable — this report audits headless-session logs and guard behaviour; no code-level graph question is asked.

No held-out content exists yet. Sections for S45–S47 (counts, estimation, split, hashes) are added by those steps.

## Rehearsal (S54)

Eval dir `~/.local/share/ccg-eval/cf7b2042cce7/`, plan `docs/plans/connected-code-graph-p0-2.md` §6.3 and §6.5. Audited 2026-11-05 with `cpg/eval/tools/audit.py`, `--accepted-plugins` (3 builtin), `--accepted-agents` (10 names), `--settings`/`--agents` from `cpg/eval/headless/`. Raw outputs: `work/s54-audit-*` (first run) and `work/s54-audit2-*` (re-run) under the eval dir.

**Verdict: PASS**, on the reading that init item 3 means "plugins are builtin and equal the accepted list" (observed: 3 builtin plugins). Read literally ("plugin list is empty") that item is unmet; the wording is an open item for the architect at S50 (O1). No other criterion is unmet.

History: the first run (sessions `fb75b8e5`, `eef808b0`, `519f5a9a`) failed because `guard-paths.sh` rule 3 allowed pattern-only Globs `{..,x}/*` and `kit-src/\.\./\.\./*` (both returned "No files found"). The guard was fixed in commit 036e2cb9 and `rehearsal/breach2` re-run.

### Audit results

| Session | Attempts | Tool calls | Exit / verdict | Refused (counted) | Void |
|---|---|---|---|---|---|
| `bf8e71bc` rehearsal/breach2 | 1 | 29 | 1 / void | path-outside-root 11, write-outside-out 2 | mcp-call 3 (see O3) |
| `eef808b0` rehearsal/resume | 2 | 32 | 0 / clean | 0 | 0 |
| `519f5a9a` rehearsal/resume2 | 2 | 33 | 0 / clean | path-outside-root 1 (attempt 2) | 0 |

`unpinned: []`, `streamTruncatedCalls` 0 everywhere. The re-run JSON for `eef808b0` and `519f5a9a` is identical to the first run's (diff empty). `settings.json` sha8 8fd894ba and `agents.json` f88cfb46 equal the run-index values; `runs/index.jsonl` has 12 lines (6 start + 6 end).

### Pass-when criteria

| Criterion | Result | Evidence |
|---|---|---|
| Every illegitimate attempt refused | PASS | breach2 stream: 13 `permission_denials` (Read x4, Grep x2, Glob x5, Write x2). All 13 results are `is_error` with `guard-paths:` text, including `{..,x}/*`, `../*`, `kit-src/\.\./\.\./*`, `{kit-src,..}/*` (Glob, no path) and Grep `glob` `{..,x}/*`. The 3 MCP calls got `<tool_use_error>No such tool available: ...` (tool not in the session; Bash, WebFetch and Agent were not attempted as tool calls) |
| Audit flags every attempt | PASS | `.details`: 11 path-outside-root + 2 write-outside-out, all `refused:true`, plus 3 mcp-call `refused:false`; in `519f5a9a` the attempt-2 planted read |
| Controls succeed, not flagged | PASS | `Read kit-src/cypher-mcp/README.md` returned content; `Glob kit-src/**/*.md` returned `kit-src/...md` paths (5192-char result), no name outside `kit-src/`. Neither appears in `.details` |
| No outside-root names returned | PASS | the five escape-shaped Globs/Grep returned only the denial text |
| Init 1, 2, 4, 5 | PASS | breach2 init: skills and slash commands `[]`, 10 agents = accepted list; no memory attachment or file (first run); OAuth works (result success, cost 0.1305712 USD, 30 turns) |
| Init 3 | PASS on builtin-set reading | 3 builtin plugins, set equal to the accepted list; literal "empty" unmet (O1) |
| Kill-and-resume | PASS | unchanged from the first run (distinct logs, full isolation set in attempt 2, no false violation, planted read flagged) |

### Open items

- **O1 (architect, S50):** plan §6.3 init item 3 says "empty"; observed 3 builtin plugins (`@builtin`); reword to "all builtin, equal to the accepted list".
- **O2 (architect, S50):** control read named `kit-src/README.md` does not exist; use `kit-src/cypher-mcp/README.md`. Also the resumed attempt's planted read must be briefed in advance (`resume` takes only the fixed continuation prompt) and `resume` reuses the recorded timeout unless `--timeout` is passed.
- **O3 (devops, low, audit.py):** a call to an absent tool (`No such tool available`, nothing executed) is not recognised as refused, so it voids the session. Correct for flagging; in production a writer that merely tries an absent MCP tool would be voided. Decide whether `<tool_use_error>... No such tool available` counts as refused.

## S41 audit (KA/http, KA/mcp, KA/wiring, KA/scripts, KA/flow)

`audit.py` with the same accepted-plugins/agents lists and current `settings.json`/`agents.json`; outputs `work/audit-KA-<stratum>.json` and `.details` under the eval dir. Tallies only.

| Session | Attempts | Tool calls | Verdict | Refused | Void | Init | Result | Key lines | Schema (EP §4.1) |
|---|---|---|---|---|---|---|---|---|---|
| KA/http `9187a4ae` | 1 | 43 | clean (exit 0) | path-outside-root 1 | 0 | isolation set (Read,Grep,Glob,Write,Edit; no MCP/skills/commands; 3 builtin plugins; 10 agents; dontAsk; claude-opus-5-5) | success, 2.3055624 USD, 44 turns | 52 | 0 unparsable, 0 failures, 0 duplicate `seq` |
| KA/mcp `f951f535` | 1 | 51 | clean (exit 0) | path-outside-root 1 | 0 | same | success, 3.0071998 USD, 52 turns | 103 | 0 unparsable, 0 key/type failures, 0 duplicate `seq`; 17 lines carry a `stratum` other than `mcp` (14 `wiring`, 3 `scripts`) |
| KA/wiring `275d816c` | 1 | 31 | clean (exit 0) | 0 | 0 | same | success, 0.977272 USD, 32 turns | 20 | 0 unparsable, 0 key/type failures, 0 duplicate `seq`; 4 lines carry `stratum` `scripts` |
| KA/scripts `d39cf4fd` | 1 | 83 | clean (exit 0) | path-outside-root 1 | 0 | same | success, 5.2896456 USD, 84 turns | 167 | 0 unparsable, 0 key/type failures, 0 duplicate `seq`; 28 lines carry another `stratum` (15 `mcp`, 13 `wiring`) |
| KA/flow `d4cb867a` | 1 | 104 | clean (exit 0) | 0 | 0 | same | success, 4.0060444 USD, 105 turns | 88 | 0 unparsable, 0 key/type failures, 0 duplicate `seq`; 4 lines carry `stratum` `http` |

All four outputs copied to `work/keys/KA-<stratum>.jsonl` (byte-identical): `KA-wiring` 20 lines, sha256 `7f2e88b61e468726dfe39745b7579624ae0a6a618b2beab95228d2e6d9d76999`; `KA-scripts` 167 lines, sha256 `c6060f4623c712662e568cac542687c3743919e3e27652a380271720d11f2631`; `KA-http` 52 lines, `KA-mcp` 103 lines (sha256 `KA-http` 3cd6f823a8627aff4a20589b780fec325b87b4a8d66725ceb660b89dbc5ccc92, `KA-mcp` 6af69fe0234adf87fdbccef67571bee722d324810108de83d2c6c02e738af929). Each session's one refused call was a read outside the root, refused by the guard. The `stratum` field is "recorded but not trusted" (EP §3.1), so the stratum mismatches (17 in KA/mcp, 4 in KA/wiring, 28 in KA/scripts) are not counted as schema failures; flagged for `teco` as a scope-adherence note.

`KA-flow` copied to `work/keys/KA-flow.jsonl` (byte-identical): 88 lines, sha256 `5aeb24c52523113c01fe2c20f29857f9840c30ba1bddf26d8cd90e74befaec90`.

## S42 audit (KB/http, KB/mcp, KB/scripts, KB/wiring, KB/flow)

Same `audit.py` command and lists; outputs `work/audit-KB-<stratum>.json`/`.details`. Tallies only. All five: 1 attempt, `streamTruncatedCalls` 0, `unpinned` empty, init matches the isolation set (same tools, no MCP/skills/commands, 3 builtin plugins, 10 agents, `dontAsk`), model `claude-sonnet-5-5`, result success, 0 refused, 0 void.

| Session | Tool calls | Verdict | Cost / turns | Key lines | Schema (EP §4.1) | Stratum mismatches | sha256 of `work/keys/KB-<stratum>.jsonl` |
|---|---|---|---|---|---|---|---|
| KB/http `4f912290` | 17 | clean | 0.4045341 USD / 18 | 29 | 0 unparsable, 0 failures, 0 dup `seq` | 0 | `ababf45222401c24597a9c41a33c6890bc33336139da4909acff68da48844012` |
| KB/mcp `76f2439b` | 27 | clean | 0.8185354 USD / 28 | 57 | same, 0 | 11 (`wiring`) | `4cedcb40b7961589af51c7058ebd1e634043fa4b2107d942a89c428109bdb61e` |
| KB/scripts `a4728922` | 42 | clean | 1.2789739 USD / 43 | 120 | same, 0 | 17 (7 `wiring`, 10 `mcp`) | `823853351f419908f08f551c77183fb23ac073691845f2c21e98f781cecea2c6` |
| KB/wiring `4ae79f7c` | 18 | clean | 0.2668278 USD / 19 | 11 | same, 0 | 3 (`mcp`) | `6627d535539a0e0624f309b7bd3859e370d5e7a997e1c69386023178f7c4fbc4` |
| KB/flow `19636ffc` | 46 | clean | 0.7265691 USD / 47 | 31 | same, 0 | 0 | `cf3c2433821b2a5e2a97f9abc1bad9f0663f2bebcd2192ce2001371ee0c6ce72` |

All five copied byte-identical to `work/keys/`. The `stratum` field is "recorded but not trusted" (EP §3.1), so mismatches are not schema failures.

All 5 KA and all 5 KB key files are in `work/keys/` (sha256 recomputed at this audit; the nine earlier hashes are unchanged): `KA-flow` 88 lines `5aeb24c52523113c01fe2c20f29857f9840c30ba1bddf26d8cd90e74befaec90`; `KA-http` 52 lines `3cd6f823a8627aff4a20589b780fec325b87b4a8d66725ceb660b89dbc5ccc92`; `KA-mcp` 103 lines `6af69fe0234adf87fdbccef67571bee722d324810108de83d2c6c02e738af929`; `KA-scripts` 167 lines `c6060f4623c712662e568cac542687c3743919e3e27652a380271720d11f2631`; `KA-wiring` 20 lines `7f2e88b61e468726dfe39745b7579624ae0a6a618b2beab95228d2e6d9d76999`; `KB-flow` 31 lines `cf3c2433821b2a5e2a97f9abc1bad9f0663f2bebcd2192ce2001371ee0c6ce72`; `KB-http` 29 lines `ababf45222401c24597a9c41a33c6890bc33336139da4909acff68da48844012`; `KB-mcp` 57 lines `4cedcb40b7961589af51c7058ebd1e634043fa4b2107d942a89c428109bdb61e`; `KB-scripts` 120 lines `823853351f419908f08f551c77183fb23ac073691845f2c21e98f781cecea2c6`; `KB-wiring` 11 lines `6627d535539a0e0624f309b7bd3859e370d5e7a997e1c69386023178f7c4fbc4`.

## S43 audit (QW/impact)

Same `audit.py` command and lists; outputs `work/audit-QW-impact.json`/`.details`. Counts only.

| Session | Attempts | Tool calls | Verdict | Refused | Void | Init | Result | Question lines | Schema (EP §4.2) |
|---|---|---|---|---|---|---|---|---|---|
| QW/impact `f54653d7` | 1 | 57 | clean (exit 0) | 0 | 0 | isolation set (same as KA/KB; `claude-opus-5-5`) | success, 2.0253176 USD, 58 turns | 48 (`en` 24, `pt` 24) | 0 unparsable, 0 failures (conditional `gold`/`goldOptions`/`ambiguityKind` rules checked), 0 duplicate `seq`; 36 unambiguous (18 `code-location`, 18 `contract-point`), 12 ambiguous (9 `code-location`, 3 `contract-point`) |

`streamTruncatedCalls` 0, `unpinned` empty. The raw output (sha256 `3e840f8bbb4aa63b3dc25f2cd6d84fbda43f7efe1243635e693012c1abe60811`, 48 lines) is kept byte-identical at `work/questions/QW-raw.jsonl`.

## S43a (strip_gold)

`strip_gold.py` requires a `qid` the writer's schema does not carry (EP §4.2 has `seq`); EP §10.2 says GC assigns `QW-<seq>` when it copies the file. `work/questions/QW.jsonl` is therefore the raw file with `"qid":"QW-<seq>"` prepended to each line and every other field unchanged (checked: all 48 lines equal the raw line plus `qid`); sha256 `c7cde6f39c57b075d7a869e8510f117534dfbb44d8329a95b4e41fcf68c4eee7`. EP §13.2 should state this step explicitly.

| Step | Result |
|---|---|
| `strip_gold.py` | exit 0, 48 items |
| `strip_gold.py --verify` | `verify: ok`, `errors: 0`, 48 items, exit 0 |
| `QW-nogold.jsonl` | 48 lines, each with exactly the keys {`aid`, `lang`, `text`}; `en` 24, `pt` 24; sha256 `2be2b92fb37b730f7d2e2d78e471cea04c9751c563e0dd6558b7f6a0b1464126` |
| `work/aid-map.jsonl` | sha256 `97698e88404d678ec74d48152daaf3cb6b054c8113f24125be1a98d698b6be01` |
| `roles/AN1-impact/QW-nogold.jsonl` | mode 0444, sha256 equal to the above |
| `roles/AN2-impact/QW-nogold.jsonl` | mode 0444, sha256 equal to the above |

Placement is the role root (the brief's "`QW-nogold.jsonl` in your working directory"). No AN session was started.

## S44 audit (AN1/impact, AN2/impact)

Same `audit.py` command and lists; outputs `work/audit-AN1-impact.*` and `work/audit-AN2-impact.*`. Counts only; no agreement or gold comparison.

| Session | Attempts | Tool calls | Verdict | Refused | Void | Init | Result | Lines | Schema (EP §4.3) | Coverage of the 48 `aid`s | Labels | sha256 of `work/questions/AN<n>.jsonl` |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| AN1 `01e4fd28` | 1 | 7 | clean (exit 0) | 0 | 0 | isolation set, `claude-sonnet-5-5` | success, 0.2044846 USD, 8 turns | 48 | 0 unparsable, 0 failures | 48 distinct, 0 duplicate, 0 missing, 0 unknown | ambiguous 8, unambiguous 40 | `872b97e7f8b8b03571b49597d6f30230c284229248eebf05f5508ba6586c5845` |
| AN2 `67033772` | 1 | 17 | clean (exit 0) | 0 | 0 | isolation set, `claude-opus-5-5` | success, 0.693197 USD, 18 turns | 48 | 0 unparsable, 0 failures | 48 distinct, 0 duplicate, 0 missing, 0 unknown | ambiguous 9, unambiguous 39 | `2b0beceb98e5c407a819b822557f7dfba6d733ffdfdaa7926f90d41223eb15be` |

Schema check: exactly the keys {`schema`, `aid`, `label`, `reason`}, `schema` = `ccg-annotation-v1`, `label` in {`ambiguous`, `unambiguous`}, `reason` a string of at most 200 characters. The `aid` set is compared with the 48 in `QW-nogold.jsonl`. `streamTruncatedCalls` 0, `unpinned` empty for both; both outputs copied byte-identical.

## S45 matching (part 1)

Tools: `cpg/eval/tools/match_keys.py` via `cpg/.venv`; working files under `work/s45/` and `work/` of the eval dir. Context file `work/s45-context.json` is built from `work/units.json`, `work/cypher-tools.json` and the `qname`s of `work/flow-anchors.jsonl`. Kit: `kit-src`. Counts only. Order run: `normalize` per file (writer KA or KB, `--log`, `--invalid`), `validate` per file and on the union, `match` on the valid union, `nearmiss`, `scope`.

**Validation (V1-V3), per key file**

| File | Lines | Valid | Invalid |
|---|---|---|---|
| KA-http | 52 | 52 | 0 |
| KA-mcp | 103 | 103 | 0 |
| KA-scripts | 167 | 167 | 0 |
| KA-wiring | 20 | 20 | 0 |
| KA-flow | 88 | 73 | 15 |
| KB-http | 29 | 29 | 0 |
| KB-mcp | 57 | 57 | 0 |
| KB-scripts | 120 | 120 | 0 |
| KB-wiring | 11 | 11 | 0 |
| KB-flow | 31 | 31 | 0 |
| Total | 678 | 663 | 15 |

Invalid by check: `V2:line-outside-span` 13, `V1:line-out-of-range` 2; all in KA-flow, seq 1-11, 13, 14, 17, 32 (clustered in the first items; flagged for the method owner and `teco` as a possible systematic line-numbering slip, not investigated). Invalid items stay out of matching. The tool's `dropped` counts are `{}`. `normalize`: 0 unparsable final lines; rewrites logged: `config-grain` 1 (KA-wiring seq 20), step 8a rewrites 0 (no V3 8a entries, so no masked-segment count to review), `E` rewrites 0.

**Matching.** `runs/index.jsonl` holds 0 `KA2` sessions: no `--ka2-strata`, `t` = 2 in all five strata.

| Measure | Value |
|---|---|
| Valid items by writer | KA 415, KB 248 |
| Valid items by derived stratum and writer | scripts KA 167 / KB 122; flow KA 69 / KB 31; mcp KA 67 / KB 34; wiring KA 59 / KB 32; http KA 53 / KB 29 |
| `match.json` records | 368 (http 49, mcp 67, scripts 154, flow 75, wiring 23) |
| Matched (found by both KA and KB, history `101`) | http 29, mcp 29, scripts 112, flow 25, wiring 19; total 214 |
| KA only (`100`) / KB only (`001`) | 146 / 8 (KB only: scripts 2, flow 6) |
| `unclassified` | 0 |
| `stratumDisagreements` | 82 |
| Near-miss pairs | 0 (N1 0, N2 0, N3 0) |
| Out-of-scope (V4) | 0; scope summary H 293, H_all 368; per stratum flow 75, http 49, mcp 67, scripts 154, wiring 23 |

Files (sha256 prefix): `work/match.json` 510bc3f4942f2e62, `work/nearmiss.jsonl` empty (e3b0c44298fc1c14), `work/s45/match-scoped.json` f5c86cb2bb082b44. Stopped before adjudication; `adjudicate`, `capture`, `sample` and `split` not run.

### S45 diagnostics (A: the 15 invalid KA-flow items; B: the empty near-miss result)

Counts and numbers only. Nothing was repaired or re-validated; `work/match.json` unchanged (sha256 prefix 510bc3f4942f2e62).

**A. Invalid KA-flow items (seqs 1-11, 13, 14, 17, 32).**
- **Distance from the cited line to the nearest edge of the symbol's span** (signed; positive = after the span's last line, negative = before its first). `V2` items, one value per cited line (13 items, 16 lines): +896, +996 (seq 1); +42 (17); +395 (3); +378 (7); +157 (4); -47 (14); +342 (8); +816 (6); +903, +1004 (2); +310 (9); +152, +165 (32); +90 (13); +105 (5). Every `V2` item has exactly one candidate span for its qname (no ambiguity). `V1` items (seq 10, 11): the cited line is past the end of the file by +108 and +118 lines (file length 1005 in both). Cited line inside the file: yes for the 13 `V2` items, no for the 2 `V1` items.
- **Pattern:** scattered, not one constant offset. Magnitudes 42 to 1004, 15 of 16 `V2` lines positive (one -47); no value repeats across items. 10 of the 15 items point into one 1163-line file, 3 into a 1005-line file, 1 each into files of 247 and 393 lines. Not a "+1" slip and not a fixed header length; the cited lines are mostly far below the named element.
- **Stratum-tag mismatch:** none of the 15 (all 15 are tagged `flow` and derived `flow`). The 82 `stratumDisagreements` do not include them.
- **Would have matched a KB record** (same `itemId`, i.e. same anchor and impacted identity, line check ignored): 6 of 15 (seqs 3, 4, 5, 6, 8, 9). The other 9 have no KB item with the same impacted element (0 share even the impacted id). These 6 are exactly the 6 KB-only flow records (6 of 6 `itemId`s equal), so the "KB-only" flow count is wholly explained by the invalid KA twins.

**B. Is zero near-misses a blind check?** Unmatched sets per stratum (KA-only / KB-only): http 20 / 0, mcp 38 / 0, scripts 40 / 2, wiring 4 / 0, flow 44 / 6; 344 candidate (KA-only x KB-only) pairs in total (scripts 80, flow 264).

| Definition | scripts (80 pairs) | flow (264 pairs) | total (344) |
|---|---|---|---|
| (1) N1/N2/N3 as run | 0 | 0 | 0 |
| (2) share any file path (anchor or impacted) | 4 | 18 | 22 |
| (3) same impacted-element path, anchor ignored | 0 | 6 | 6 |
| (4a) same impacted top-level container (path plus first name segment) | 0 | 0 | 0 |
| (4b) same `anchorId` (any impacted) | 3 | 12 | 15 |

Strata with KB-only 0 (http, mcp, wiring) have no candidate pairs by construction. No pair has equal `impactedId`, so N2 and N3 cannot fire on this data; N1 needs equal anchor and equal impacted path, and no pair meets both.

**Does the verb see what it should?** On a scratch copy of `match.json` (`work/s45/scratch/`, never `work/match.json`), I injected (a) a provider-only duplicate of a real KB-only record with its cited lines shifted by one and a different `impactedId`: `nearmiss` returned 6 pairs, all N1, each including the injected record (it shares anchor and impacted path with the 6 flow KB-only records, which share one anchor and one file); (b) a provider-only/consumer-only pair of http anchors with different `anchorId` and `keyNorm`s differing by a trailing `/`: `nearmiss` returned 1 pair, N2. N3 was not exercised. So N1 and N2 are not blind; the real result is a true zero under the rules, with the caveat that the line-shift in (a) has no effect on the verdict (anchors carry no line; `itemId` ignores lines).

**Feed for `capture` (read-only; per stratum n1 = found by KA, n2 = found by KB, m = both; `t` = 2):**

| Stratum | n1 | n2 | m |
|---|---|---|---|
| http | 49 | 29 | 29 |
| mcp | 67 | 29 | 29 |
| scripts | 152 | 114 | 112 |
| wiring | 23 | 19 | 19 |
| flow | 69 | 31 | 25 |

If the 6 invalid KA-flow twins were valid, flow would read n1 75, n2 31, m 31 (a computation from the counts above, not a run).

## S45 KA2 (flow2)

Session `90a53928` (run-index role `KA`, stratum token `flow2`; treated as KA2 and normalized with `--writer KA2`, since the run index has no `KA2` role). Counts only.

**Audit:** clean (exit 0), 1 attempt, 97 tool calls, 0 refused, 0 void, `streamTruncatedCalls` 0, init matches the isolation set (`claude-opus-5-5`), result success, 4.6156886 USD, 98 turns. Output copied to `work/keys/KA2-flow.jsonl` (byte-identical), 111 lines, sha256 `d98dabd5cef031e777045a720fd1a979c96eb18e5b8c80d34a36eff639a8f96c`; 0 unparsable, 0 schema failures, 0 duplicate `seq`; stratum tags `flow` 107, `http` 4.

**Normalize and validate:** 111 records, 111 valid, 0 invalid (so 0 by reason, 0 by file); 0 rewrites. Derived stratum: flow 107, http 4. The 4 http-derived KA2 records were left out of matching: the `match` contract sets `t` = 3 for any stratum with a KA2 record, and KA2 was a flow pass; all 4 are already held by KA and KB (history `101`), so they add nothing. Matching used the 663 valid part-1 records plus the 107 flow records of KA2. The 15 invalid KA-flow items stay out.

**Re-match (`--ka2-strata flow`):** 420 records (part 1: 368), `t` flow 3, others 2; `unclassified` 0, `stratumDisagreements` 82. Flow: 127 records. History (KA, KA2, KB): `111` 21, `110` 28, `101` 4, `100` 16, `011` 6, `010` 52. The 6 formerly KB-only flow records are now `011` (KA2 re-found them with valid lines). Other strata unchanged (http 49, mcp 67, scripts 154, wiring 23). The part-1 `match.json` is kept as `work/s45/match-part1.json`.

**Estimates (§6.1-§6.4, `capture.py` on the scoped file, `work/capture.json`):**

| Stratum | n1 | n2 | m | t | Chapman N | Chao N | N* | Completeness | Trigger |
|---|---|---|---|---|---|---|---|---|---|
| flow | 127 (KA 69, KA2 107) | 31 | 31 | 3 | 127 (interval degenerate) | 165.94 | 165.94 (Chao binds) | 0.765 | yes |
| http | 49 | 29 | 29 | 2 | 49 | 52.17 | 52.17 | 0.939 | yes |
| mcp | 67 | 29 | 29 | 2 | 67 | 78.72 | 78.72 | 0.851 | yes |
| scripts | 152 | 114 | 112 | 2 | 154.71 | 157.81 | 157.81 | 0.976 | no |
| wiring | 23 | 19 | 19 | 2 | 23 | | | 0.994 | no |
| pooled | r 420 | | | | | | 477.78 | 0.879 | |

Flow: f1 68, f2 38, f3 21; **flow: completeness below threshold** (0.765 < 0.95). No third pass runs. http (0.939) and mcp (0.851) also fall below 0.95 and were not asked about; they are for `teco`'s decision (EP §6.5: at most one extra pass per stratum).

**EP §7.2 (invalid items):** 15 items, all KA-flow, all still invalid and counted as not found by KA (KA2 re-found 6 of them by identity, now valid). By file (anonymised): F1 10, F2 3, F3 1, F4 1. KA-flow invalid share 15/88 = 17%, above the 10% line. The held-out count by file is not available until S47's split; this table is the concentration input.

**S45 part 2:** near-miss check on the new `match.json`: 0 pairs (N1 0, N2 0, N3 0), so no adjudicate. Scope (V4): 0 out of scope; H 294, H_all 420 (flow 127, http 49, mcp 67, scripts 154, wiring 23). Capture: above. Sample (`sample`): `work/spot.jsonl` 40 union items (8 per stratum; 20 singleton, 20 doubleton, 4 per cell), shortfall records 0; `work/ptgold.jsonl` 8, `work/engold.jsonl` 8. Hashes (prefixes): `match.json` fc411944dd9a05cf, `capture.json` 71371bf01519fd78, `spot.jsonl` 3ce35a6732b6cf29, `ptgold.jsonl` 1f08b438f11d7e25, `engold.jsonl` 419734bea1d625ad, `nearmiss.jsonl` empty.

## S45 KA2 (http2, mcp2)

Counts only. Both are KA2 passes under EP §6.5 (http 0.939 and mcp 0.851 were below 0.95), original KA-http and KA-mcp briefs unchanged; normalized with `--writer KA2`.

| Session | Audit | Tool calls | Refused | Void | Cost / turns | Lines | Schema | Valid / invalid | Invalid share | sha256 of `work/keys/KA2-<stratum>.jsonl` |
|---|---|---|---|---|---|---|---|---|---|---|
| KA-http2 `6aa23e3a` | clean (exit 0), init matches the isolation set | 51 | path-outside-root 1 | 0 | 2.6173528 USD / 52 | 59 | 0 unparsable, 0 failures, 0 duplicate `seq` | 59 / 0 | 0% | `f977ea9a948e0b1f036c953727ca661512f8bc283d863bad75a8c07bda5277c3` |
| KA-mcp2 `bf1937fb` | clean (exit 0), init matches the isolation set | 48 | 0 | 0 | 2.308611 USD / 49 | 77 | 0 unparsable, 0 failures, 0 duplicate `seq` | 77 / 0 | 0% | `6f2e085f51bba8bd6704c8db5b8aaab1dd2a0ebe62a13531ee18623133e8aaef` |

No invalid items, so no breakdown by reason or file; the EP §7.2 invalid-share rule (10%) does not apply to either. Invalid items overall: still the 15 KA-flow items (F1 10, F2 3, F3 1, F4 1; 17% of that session).

**Off-stratum KA2 records (my call under the `match` contract, which sets `t` = 3 for any stratum holding a KA2 record).** Matching used each KA2 session's records that derive to its own stratum only: flow 107 (flow2), http 49 (http2), mcp 43 (mcp2). Excluded: http2 derives scripts 6 and wiring 4; mcp2 scripts 18 and wiring 16; flow2 http 4. Of these 48 excluded records, 48 were already held by KA or KB (history already present in the earlier match) and 1 (mcp2, derived scripts) was found by no session. That one item is left out so scripts and wiring keep `t` = 2; including it would add 1 record to scripts and make scripts `t` = 3 without a scripts KA2 pass. Reversible: say if you want it in.

**Re-match** (`--ka2-strata flow http mcp`; `t`: flow 3, http 3, mcp 3, scripts 2, wiring 2): 420 records, same as before (neither new pass found an unheld in-stratum item); `unclassified` 0, `stratumDisagreements` 82. Histories (KA, KA2, KB): http `111` 29, `110` 20; mcp `111` 29, `110` 14, `100` 24; scripts and wiring unchanged. Previous files kept: `work/s45/match-part2a.json`, `capture-part2a.json`, `spot-part2a.jsonl`, `ptgold-part2a.jsonl`, `engold-part2a.jsonl`.

**Estimates (§6.1-§6.4; n1 = found by KA or KA2, n2 = KB, n3 = KA2 alone count shown as sessions):**

| Stratum | n1 | n2 | n3 (KA2) | m | t | Chapman N | Chao N | N* | Completeness | Flag |
|---|---|---|---|---|---|---|---|---|---|---|
| http | 49 | 29 | 49 | 29 | 3 | 49 | 49.00 | 49.00 (Chapman binds) | 1.000 | |
| mcp | 67 | 29 | 43 | 29 | 3 | 67 | 79.27 | 79.27 (Chao binds) | 0.845 | completeness below threshold |
| scripts | 152 | 114 | 0 | 112 | 2 | 154.71 | 157.81 | 157.81 | 0.976 | |
| wiring | 23 | 19 | 0 | 19 | 2 | 23 | 23.15 | 23.15 | 0.994 | |
| flow | 127 | 31 | 107 | 31 | 3 | 127 | 165.94 | 165.94 | 0.765 | completeness below threshold |

http f1 0, f2 20, f3 29; mcp f1 24, f2 14, f3 29. Pooled: r 420, N* 475.17, completeness 0.884. Chapman intervals are degenerate except scripts (154.09 to 159.40). mcp (0.845) and flow (0.765) stay below 0.95 after their extra pass; no further passes run.

**S45 part 2 redrawn:** near-miss 0 pairs (N1 0, N2 0, N3 0), no adjudicate; scope 0 out of scope (H 294, H_all 420); capture as above (15 invalid items); sample: `work/spot.jsonl` 40 union items (8 per stratum; 20 singleton, 20 doubleton), shortfall records 0; `work/ptgold.jsonl` 8, `work/engold.jsonl` 8. Hashes (prefixes): `match.json` dc01128e989a12f3, `capture.json` 44d0006dd5f7b5d3, `spot.jsonl` 379629ac97ae4cc4, `ptgold.jsonl` 1f08b438f11d7e25 (equals the previous draw), `engold.jsonl` 419734bea1d625ad (equals the previous draw).

## S47 split and manifest

Counts, hashes and the contacted anchors only; no held-out content. `E` = `cf7b2042cce7bad5d0ab7876940bcea00343d914`; seed `ccg-p0f-split-v1|E`; protocol `docs/plans/connected-code-graph-eval-protocol-ml.md@1.7+docs/plans/connected-code-graph-eval-protocol2-ml.md@1.4`.

**Procedure (honest record).** S46: the 40 union items were verified against source by an independent analyst (AI), not checked by a human: 47 of 48 entries correct (the 40 union items 39 correct, 1 incorrect). The 8 Portuguese gold items (entries 41-48) were human-confirmed by the stakeholder ("all fit"): 0 of 8 incorrect. The analyst re-verified entry 28 after a partial first check.

**Corrections (EP §7.4).** Entry 26 is incorrect (scripts stratum, found by KB only, counting; the cited script only prints a hint and never runs the command). The note gives no locator, so the item is removed (logged in `work/corrections.jsonl`). Incorrect by stratum: scripts 1 of 8 (Wilson 95% [0.022, 0.471]); other strata 0 of 8 each (Wilson [0.000, 0.324]); pooled 1 of 40 (Wilson [0.004, 0.129]). Escalation (a stratum with 2 or more of 8, or 4 or more of 40): not triggered, so no singleton review. PT gold check: 0 of 8 incorrect; the EN gold sample (`work/engold.jsonl`, 8 items) was not part of the stakeholder check and has no recorded verdict. No re-estimation: N* below stands; the post-correction union is 419 items (420 before).

**Estimates (unchanged from S45; `work/capture.json`):**

| Stratum | n1 | n2 | m | t | N* | Completeness | Flag |
|---|---|---|---|---|---|---|---|
| http | 49 | 29 | 29 | 3 | 49.00 | 1.000 | |
| mcp | 67 | 29 | 29 | 3 | 79.27 | 0.845 | completeness below threshold |
| scripts | 152 | 114 | 112 | 2 | 157.81 | 0.976 | |
| wiring | 23 | 19 | 19 | 2 | 23.15 | 0.994 | |
| flow | 127 | 31 | 31 | 3 | 165.94 | 0.765 | completeness below threshold |

Post-correction union by stratum: http 49, mcp 67, scripts 153, wiring 23, flow 127 (419). The scripts item removed was a KB-only record, so scripts n2 114 stands in the table above as estimated.

**Contact set (EP §8.2).** Every source is closed and committed: S14 (slice report §4), S30 (rule-pack spec §11), the P0 golden tests (`cpg/tests/golden/answer-ac1.json`, the only golden file), S56 flow labels (all seven languages) and S59 `contact.json` (`files` 9 over the two corpora, `flowAnchors` 33 paths, union 39 distinct paths). Rewrites through `normalize_contract` (contract) and `E` (symbol): 0 of 9 raw lines changed. `work/contact.jsonl`: 9 lines, 6 distinct anchors, listed by locator (they are dev): `GET /shop/api/health` (S14, S30, golden), `GET /health` (S14, S30), `GET /x/api/health` (S14), the `salesperson/src/api/client.ts` symbol `buildUrl` (S14), MCP tool `send_message` (S30), and the unit `falkor-chat/server` (python) (S30). S59's paths and S56's flow methods are path-level; no item in the union has an anchor, impacted element or provider in any of those 39 paths, so they contacted 0 anchors (checked by path on the union). Contact sent 4 clusters to dev.

**Split (EP §9.1).** 136 clusters; 43 dev clusters (4 contacted, 39 by rank); held clusters per stratum: flow 12, http 15, mcp 3, scripts 36, wiring 4; 23 reserve clusters. Step 3b `served-anchor` moves: 0. Step 4 top-up: none needed (no marked stratum). SQ-P0-1 trigger: false.

| Stratum | Held-out items | of which counting | Held-out anchors | Dev items |
|---|---|---|---|---|
| http | 26 | 26 | 15 | 18 |
| mcp | 46 | 46 | 4 | 20 |
| scripts | 79 | 79 | 36 | 49 |
| wiring | 8 | 8 | 7 | 11 |
| flow | 99 | 0 | 12 | 28 |
| reserve | 35 (counting 35) | | 23 | |

`H` (held-out counting items, pooled) 159 beside `H_all` 258; the AC-5 floor of 75 is met by the four non-flow strata alone; flow contributes 0 counting items (E11). wiring sits at its minimum of 8. Files: held-out `ac5-<stratum>.jsonl` x5, `ac5-reserve.jsonl`, `ac24.jsonl`, an empty `exposure.jsonl`, in `~/.local/share/ccg-eval/cf7b2042cce7/heldout/`; dev `cpg/eval/dev/ac5-<stratum>.jsonl` x5 and `ac24.jsonl`.

**AC-24 (EP §10.2-§10.3).** Annotators AN1 and AN2 agree on 47 of 48 `aid`s (raw agreement 0.9792; Cohen's kappa 0.9286, diagnostic only). Outcomes: unambiguous set 36 (en 18, pt 18); ambiguity set 8 (en 4, pt 4); dropped "writer-ambiguous, annotators-unambiguous" 3 (en 2, pt 1); dropped annotators disagree 1 (pt); "writer-unambiguous, annotators-ambiguous" 0. Split: held-out 30 (unambiguous 24: 6 per language and group cell; ambiguous 6: en 3, pt 3); dev 14 (unambiguous 12; ambiguous 2). Gold stored normalized with `form` `provider` and the writer's original object beside it; gold `contract` locators rewritten: 0; masked `http` gold anchors needing the unserved-gold check: 0.

**EP §7.2: invalid KA-flow items in held-out.** The 15 KA-flow items with invalid lines count as not found by KA. All 15 are held-out flow items (their identities were re-found by KA2, 6 also by KB). By file (anonymised): F1 10, F2 3, F3 1, F4 1. The held-out concentration is therefore 10 of 15 in one file, and 15 of the 99 held-out flow items.

**Manifest** `cpg/eval/heldout-manifest.json` (sha256 `ce2bb8b1c1c7941ede76e742b03ce3a0e7777128cfe376303ae07eccfc6f800d`, 70 raw artifacts, `verify`: 77 checked, 0 mismatches). File sha256: `ac5-http` 24237acd7832db4f54d74dfcc84398802ab00e1e2246c421daa87e8ed65add89; `ac5-mcp` 04f2e39f7a171ffd6226148a2148518281e9677adf79bc6f43e8b0b8f978020c; `ac5-scripts` 842162dff65823bd129673f3dfa9dd3edac07a50af7bfe5525c995ba086818a5; `ac5-wiring` 55df0ab263b4e9d74fdc46bbae62739029306aabb832d888d21f0152e4ecb8d8; `ac5-flow` db7f5244b20fb0f99c9d53bd99d21e25a2748ee3083509b1976f7a81d28c430a; `ac5-reserve` 118e460ace4b079d8d9f5a4bed89b483d240436d6c76d8dda0346b57c8161676; `ac24` 1fc707eecc852b29605e6767fc220db03827a082bd63ace0e397dc8468117169. Dev sha256: `ac5-flow` a67ca4c19db71d6231669bec9fc93ec8bd54d19cb9943255199446d957e4c66d; `ac5-http` 197d15e6a0450b5c31929f926c32ac2dbc9af1ac24d3e6b1fe356bc91052f90f; `ac5-mcp` 90c704f102ae2fe4e67ffd15a251e18ed27a2e163c08155fe0b5ae685a6d283b; `ac5-scripts` 132106ea0363dafe71d276e2a1cfb0516f9b2dd934fca3d657878a6d478244dd; `ac5-wiring` 8ec9cc3a92b1c498d6c4c815535c320c4f4471825558de4414fa23eec5b179b1; `ac24` 153996d69e2470834c3a98297a6bf06f5eaee42fc2b0590a07044c8784d038eb. The 70 raw artifacts cover `runs/` logs and index, role `out/` files, key and question files, and the match, capture, contact, corrections and aid-map files; session transcripts live under `~/.claude/projects/` and are not hashed.

**Leak check.** `split_manifest.py check-report` (held-out directory, dev files and `contact.jsonl`; EP2 §1.4 probes, normalized and pre-8a forms): 1761 probes, 0 hits, exit 0.
