# Kaizen — Change History: architect

> Dated log of actual changes to the `architect` agent. Most recent first.

## 2026-09-10 — Kaizen distillation, `architect`'s inbox (2 entries, both captured mid-writing the out-of-scope K-030 plan): 1 promoted to `graph-dba`'s KB, 1 discarded as already captured more deeply in the K-030 plan itself

- **What:** `cobb` distillation pass over `architect`'s produced inbox. Re-queried fresh: exactly 2
  entries, both dated 2026-09-10, both captured while `architect` was writing
  `claude/docs/plans/agent-knowledge-base-strategy.md` (K-030, out-of-scope, a separate concurrent
  session's in-progress plan — read-only, never edited).

- **`b4f0e6a1-9d2c-4a5f-8b31-6e0c9a2d5f18` (FalkorDB single-quoted string literals reject the
  SQL-style doubled-single-quote escape for an apostrophe) — PROMOTED into
  `claude/graph-dba/falkordb-quirks.md`, § *Cypher dialect & query behavior*.** Grepped
  `falkordb-quirks.md`, `skills/agent-maintenance/SKILL.md` and `skills/agent-standards/*.md` for
  "quote"/"apostrophe"/"escape" first — no prior coverage of this exact gotcha anywhere. Re-derived
  live via the `cypher` MCP tool against `kaizen_team`: `RETURN 'it\'s a test' AS a, "it's a test"
  AS b` both return the correct string; a `CREATE` using `''` inside a map-literal string value
  fails with `"Invalid input mismatched quote"` at the second quote. This is a durable, live-verified
  FalkorDB dialect fact any agent writing a free-text Cypher string literal benefits from, not
  something specific to K-030 — added as a new bullet, sited beside the existing "no
  string-repetition operator" entry (same "how string literals actually behave on this build"
  family). Also logged in `claude/graph-dba/kaizen/history.md` per the cross-agent-KB-promotion
  convention.

- **`7c1e2f9a-4b6d-4e2a-9c3f-1a8d6e5b7c02` (falkor-chat's document-ingestion pipeline is
  create-only — no `update_document`/`delete_document`/`list_documents` anywhere) — DISCARDED, no
  edit, already captured more deeply at the document it was written for.** Read
  `claude/docs/plans/agent-knowledge-base-strategy.md` §1 "The substrate fork — resolved, with
  rationale" (`:51-121`) whole: it states the exact same CPG-confirmed absence
  (`update_document`/`delete_document`/`list_documents` missing from
  `falkorchat/{services,repository,mcp,api}.py`), the same non-idempotent-`create_document`
  citation, and reaches the identical conclusion for Open question 1 (the substrate fork) — plus
  goes further than the raw entry (tenancy/side-effect mismatch, the LLM-based entity/relationship
  extraction machinery `ingest_document` would drag in). Same disposition shape U55/U56 already
  established this pass for an entry captured mid-writing K-030 material: the plan doc is the
  deeper, more authoritative treatment, so the raw capture is a compressed duplicate, not a new
  fact. No project-docs edit made.

- **Graph ops (per entry, write→log→clear, never batched):** both re-counted individually
  immediately before clearing (`MATCH (k:KaizenEntry {entryId:'<id>'}) OPTIONAL MATCH
  (:Agent)-[p:PRODUCED]->(k) OPTIONAL MATCH (k)-[m:MENTIONS]->(:Agent) RETURN count(DISTINCT p),
  count(DISTINCT m)`), each returning `producedEdges=1, mentionEdges=0` ⇒ `otherRemaining = 0` ⇒
  full-node `DETACH DELETE` for both. No `MENTIONS` tag added to either: neither is substantively
  about a different agent (entry 1 is a generic FalkorDB dialect fact; entry 2 is about
  falkor-chat, already fully covered by `architect`'s own in-flight K-030 plan).
- **The `tico` orphan (`e1a6c4d2-8b3f-4b1a-9c7e-3f2a6d9b1c4e`, `MENTIONS`→`tico`) was verified
  untouched** — read-only check, not swept.
- **Docs touched:** `claude/graph-dba/falkordb-quirks.md`, `claude/architect/kaizen/history.md`,
  `claude/graph-dba/kaizen/history.md`.
- **Why:** distillation of `architect`'s `kaizen_team` inbox, requested standalone (not part of a
  numbered coordination unit this time). `architect` closes at 0 produced / 0 mentioned.
- **Plan items:** none opened — both entries landed inside `cobb`'s write remit or were discarded;
  nothing kept open.

## 2026-09-10 — Kaizen distillation, `architect`'s inbox (5 entries): 3 promoted (all generalized, 0 model-bench facts recorded), 1 discarded as already published, 1 discarded as superseded (U51)

- **What:** U51 of `claude/docs/plans/kaizen-distillation2-coordination.md`. Re-queried at dispatch:
  exactly the 5 entries the brief named, all dated 2026-09-10. Three read as general
  grep-pin/plan-editing techniques; two as model-bench-specific facts, flagged for the same
  collision risk U50 hit (a concurrent, unrelated model-bench coordination has commits and
  uncommitted work in flight right now) — **no file under `model-bench/` or
  `docs/plans/small-model-benchmarking*.md` was read for writing, staged, or committed.**

- **`a4f1c2e8` (line-break-wrapped grep residual pin) — DISCARDED, already published.** The fact
  ("a line-based grep cannot see a phrase a hard wrap has split") is, word for mechanism, the rule
  `claude/analyst/review-techniques.md`'s section *"A 'this already exists' claim is a grep away
  from confirmation"* already states — including a same-day (2026-09-10) Tombstone correction
  sharpening it further (inline emphasis markers split a phrase the same way a wrap does). The
  entry's addendum — verify the residual's before-count is non-zero rather than assumed — is
  likewise already covered in substance by this file's own Guardrails ("a residual asserted over a
  token the edit does not retire is satisfied by construction") and by `review-techniques.md`'s
  item 4 in the grep-pinned-edit-table section. Re-derivation note: the entry's own worked citation
  (`convo.py:79-80`, "home of the mapping") does not match current `model-bench/modelbench/convo.py`
  at all — that file has moved past the state the citation was taken from — but this is
  irrelevant to the disposition, since the entry states a general technique, not a fact about
  `convo.py`'s current content.

- **`7c1f0a3e` (wrong plan/note mapping cell: delete the plan's column, don't split the row) —
  PROMOTED, new Guardrails bullet in `architect.md`.** Checked `review-techniques.md` and
  `architect.md` for overlap first (grepped "denominator", "second home", "split its rows",
  "single source of truth" — no hits). Genuinely new: distinct from the existing "Compress by
  pointer" bullet's `-graph.md`/`-ml.md`-divergence case (a stale *premise*), this is a
  review-gate-found *wrong cell* in a mapping duplicated across the plan and a note. Added as its
  own short bullet rather than folded into `architect.md:51` (already flagged 1,617 chars / the
  file's longest line across three prior units — U31, U40, U41 — as wanting a compaction pass, not
  a further extension).

- **`3f1c9e42` (self-referential grep pin over-counts by one) — PROMOTED, new item 8 in
  `review-techniques.md`'s *"A grep-pinned edit table is an edit list, not a completeness proof"*
  section.** Checked the existing six-item list plus "Two derived checks" for overlap (grepped
  "self-referential", "matches its own", "inside the document", "pin table", "one high" — no
  hits): none of items 1–7 cover a pin table living *inside* the very document it counts. Added as
  item 8, directly before "Two derived checks" — same section, no new `##`.

- **`a2f6b6b0` (plan Appendix A describes a shape shipped S1 code doesn't implement — `latencyMs`)
  and `a1e6f2b0` (`ToolDispatchFailed` pins a deliberately-open design decision) — both
  RE-DERIVED TRUE at the last **committed** sha, both PROMOTED as generalized rules only; no
  model-bench-specific fact recorded anywhere.** Re-derivation was pinned to the last commit
  (`41d82e9`), not the dirty working tree, per U27's precedent that `model-bench/` is another
  session's live area — and this mattered: the working tree right now carries a large **uncommitted**
  diff (another session's in-flight work) that already converts `latencyMs` to a derived
  `@property` over a new `ItemTiming` and decides the exact `ToolDispatchFailed` question this
  entry calls "deliberately open." At committed HEAD, neither fix has landed: `results.py:164`
  still declares `latencyMs: float | None` as a plain dataclass field (no `ItemTiming` class exists
  in that file at all), and `convo.py`'s `ToolDispatchFailed` docstring still reads *"What is
  deliberately not decided here is whether such a call should abort the conversation at all …
  left open on purpose"* — matching both entries exactly, unlike U50's two casualties. So this is
  **not** the U50 pattern (fixed-in-tree, discard) — both hold, right now, at the pinned baseline.
  But the *specific* facts have an unusually short half-life: the concurrent session is mid-fix on
  both, uncommitted, and either could land before this sentence is read. Recording either as a
  durable project-docs fact would very likely ship something false within hours, and I have no safe
  place to put it anyway (`model-bench/` is off-limits this unit). What survives the transience is
  the **generalized technique** each entry is really an instance of — neither previously stated
  anywhere in `architect.md` or `review-techniques.md` (grepped "deliberately open", "design
  decision", "provisional", "owed to" — no hits) — so both are promoted as new Guardrails bullets
  in `architect.md`, stated as rules with no current-state claim about `model-bench` attached:
  (1) a prior plan's own Appendix/skeleton description of shipped code is a claim, not a fact —
  diff the actual module source before specifying downstream work on a claimed shape; (2) when a
  plan defers a design decision, specify pinning the deferral in the implementation itself (name
  the mechanism up front, state decided-vs-open in its own docstring, add one self-flagging
  "this is the test that changes" test) rather than leaving it as prose an implementer can resolve
  silently.

- **Graph ops (in order, per entry — write history, confirm, then clear; never batched):** all
  five re-queried individually immediately before clearing
  (`MATCH (k:KaizenEntry {entryId:'<id>'}) OPTIONAL MATCH (:Agent)-[p:PRODUCED]->(k) OPTIONAL MATCH
  (k)-[m:MENTIONS]->(:Agent) RETURN count(DISTINCT p), count(DISTINCT m)`), each returning
  `producedEdges=1, mentionEdges=0` ⇒ `otherRemaining = 0` ⇒ full-node `DETACH DELETE` for all
  five. No `MENTIONS` tag added to any: none of the five is substantively about a different agent.
- **Budget:** `architect.md` **1,896 → 2,172 w** (+276, three new Guardrails bullets), longest line
  unchanged at 1,617 chars (`:51`, still the pre-existing compaction candidate, not touched here).
  `claude/analyst/review-techniques.md` **14,129 → 14,285 w** (+156), sections unchanged at **33**
  (item 8 is inside the existing grep-pinned-edit-table section, no new `##`), 0 lines newly over
  700 chars.
- **Docs touched:** `claude/architect/{architect.md,kaizen/history.md}`,
  `claude/analyst/review-techniques.md`.
- **Why:** unit U51 of `claude/docs/plans/kaizen-distillation2-coordination.md`. `architect`'s
  produced inbox is now 0.
- **Plan items:** none opened — every entry landed inside `cobb`'s write remit or was discarded;
  nothing kept open.

## 2026-09-09 — Kaizen distillation, `architect` chunk 2 of 2 (the eight entries dated 2026-09-08/09-09): 6 promoted (2 halves), 2 discarded — **`architect`'s inbox closed at 0/0** (U41)

- **What:** U41 of `claude/docs/plans/kaizen-distillation2-coordination.md`, the closing chunk, run with one `cobb`-produced entry (`2746ee65…`) folded in deliberately because three of the nine land on the **same paragraph** of `skills/agent-standards/claude-code.md` and splitting them across two units a day apart is the concurrent-edit hazard this coordination keeps logging. **Zero new sections in any file** — every promotion is a fold into existing material, except the two new bullets inside `claude-code.md`'s existing `## Bash tool environment` section.
- **Routing-first held for the fourth unit running.** `suggestedHome` predicted **0 of 8** homes (six said `knowledge base`, two `prompt`); nothing landed in `architect`'s own artifacts, and `architect.md` was not opened — its longest line (`:51`, 1,617 chars) is still the compaction item recorded in U31 and U40. Homes were `skills/agent-standards/claude-code.md`, `skills/python-web-quirks/SKILL.md`, `claude/analyst/review-techniques.md` and `claude/tdd-engineer/guard-testing-techniques.md`.
- **Every cell over 300 characters was paged before disposition** (`substring(k.fact, offset, 240)` cross-checked against `size()`): six of the nine facts and eight of the nine evidence cells exceed the tool's `CYPHER_MCP_MAX_CELL` cut, the longest at **554**.

**`0acf4d71-dea8-445c-a0ac-ca2175838231` — HALF promoted (shim's existence discarded as published; the absolute path and the reproducible-count rule promoted).** `claude-code.md`'s `## Bash tool environment` section was read **whole** (`:574–599`, then `:599–655`) before ruling. Its first bullet already documents the shim in the entry's own terms — `grep` as a shell function exec-ing the claude binary with `ARGV0=ugrep`, its flag list including `--ignore-files`, the `export -f` consequence, and the BRE-vs-ERE divergence — so that half is a discard. What it does **not** carry, and what the entry is actually for: `/usr/bin/grep` as the **inline** escape (the section prescribed only the heavier subprocess route), and the rule that a count destined for a document as a reproducible claim must be measured with the binary the reader will run. Re-derived 2026-09-09: `grep --version` → `ugrep 7.8.4`, `/usr/bin/grep --version` → `grep (GNU grep) 3.11`, `type -a grep` shows the function. The entry's own reassurance was verified too and promoted as the bound — the 18 grep-pinned residual commands agreed under both binaries, so divergence is the exception and has to be measured rather than assumed.

**`b3c301c6-608b-49fe-a768-98368381d9e0` — promoted (same edit; it supplies the mechanism the previous entry only names).** Reproduced digit for digit at the entry's own two arms, every measurement stating its binary. Named gitignored path: `grep -cF include-system-site-packages model-bench/.venv/pyvenv.cfg` → **1** under the shim and **1** under `/usr/bin/grep`. Recursive from `model-bench/`, controlled for the shim's own flags: `/usr/bin/grep -rlFI --exclude-dir=.git` → **2** files (`.venv/pyvenv.cfg`, `.venv/lib/python3.12/site-packages/pip/_internal/utils/virtualenv.py`), shim → **0**. Second case, non-hidden ignored directory, reproduced as a mechanism but **not as a figure**: the entry's "84 files under `tests/__pycache__`" states no pattern, and I read 59, 85 or 0 depending on pattern and `-I`, so only the first arm was promoted with numbers. **Mechanism attributed by control, not by assumption** — `.venv` is hidden and the shim passes `--hidden`, `pyvenv.cfg` is ASCII so `-I` is not the cause, and passing **`--no-ignore-files`** to the shim returns both files. That last arm is my own addition, not the entry's: it is a third and lighter escape route than either the subprocess or the absolute path, and it is what proves the attribution.

**`3f1c8a6e-2d47-4b19-9c05-7ae61b0d4f82` — promoted (new bullet in the same section; unled by the brief).** A markdown row of ~35 KB on one line defeats `grep -n`, `sed -n <N>p` and `Read` alike — a harness-tooling fact, which is what `claude-code.md`'s observed-behaviour bullets are for, and it sits naturally beside the backgrounded-pipe and scratchpad bullets already there. Promoted as the rule plus an **operational detector** the entry lacks (`awk '{print length}' <file> | sort -rn | head -1`), so the shape is found before it is paid for rather than after. The instance figures are the entry's own and are pinned to it (34,838 bytes, ~400 wrapped lines at `fold -s -w 110`, an 85 KB `grep -n` result); no claim is made about the document's current state.

- **Graph (this group):** all three current-shape, `1 PRODUCED / 0 MENTIONS` each ⇒ `otherRemaining = 1 + 0 − 1 = 0` ⇒ full-node `DETACH DELETE`.

**`c3f7a1e2-5b64-4d19-9f8a-2e7c61d40b93` — DISCARDED in full, and its one unpublished claim is FALSE.** `skills/python-web-quirks/SKILL.md`'s executor section was read **whole** (`:644–682`), not grepped. The entry's substance is published there in that text's own words: the `put` at `thread.py:178` preceding `_adjust_thread_count` at `:179`, the refusal arriving with the job enqueued, the same reproduction (patch `Thread.start` to raise with one worker busy), and the enumeration of the pre-queue raises — *"two of the **three** raises ahead of the `put`; the third is `BrokenThreadPool` at `:167`."* U39 landed it from `coder`'s near-twin capture. What the entry adds beyond that text is the claim that **all four raise a bare `RuntimeError`, so exception type cannot discriminate** — and the published text already says the opposite, more precisely: `BrokenThreadPool` is separable *by type* and only the shutdown pair is not. Re-derived 2026-09-09, `BrokenThreadPool.__mro__` = `BrokenThreadPool → BrokenExecutor → RuntimeError → Exception`, and a post-shutdown `submit` raises a plain `RuntimeError('cannot schedule new futures after shutdown')`. So the shipped section is not merely equivalent to the entry, it is the **corrected** version of it. The one true residue — that a *later-started* worker also runs the abandoned item — is not lost: it belongs to the cold-pool mechanism and was promoted with `3f5b1c02…` below.

**`3f5b1c02-6a7e-4d1b-9f2a-8c4e77b0d913` — promoted (`skills/python-web-quirks/SKILL.md`, folded onto the existing `qsize`/`_threads` table's two-bounds paragraph; no new section).** The brief predicted a half here and predicted its own discard steer would fail; it was a **full** promotion, and the reason is the one this pass keeps rediscovering — the published table measures a pool that *did* start a thread (`after shutdown(wait=True)` → `qsize 1`, `_threads 1`, explained by the `None` sentinel), while the entry's claim is the **empty-`_threads`** case, a different mechanism with the same reading. No sentence in `:644–682` states it. Re-derived 2026-09-09 on CPython 3.12.3 (system interpreter, not `falkor-chat`'s venv — that tree is another session's and was only read): fresh `max_workers=2` pool, `Thread.start` patched to raise, `submit` raised `RuntimeError("can't start new thread")`, `qsize` **1** / `_threads` **0**, and `shutdown(wait=True)` **returned in 0.0000 s** with the job unrun after a 0.3 s settle. **Re-derivation sharpened the entry twice.** (1) Post-shutdown `qsize()` reads **2**, not 1 — the abandoned item *plus* the shutdown sentinel — which makes the cold case distinguishable from the published warm row by one unit; that diagnostic is the entry's own claim made checkable and the entry does not have it. (2) The "until some later successful submit" clause was confirmed by construction and is stronger than written: the abandoned item ran **ahead** of the item that revived the pool. Both controls ran alongside a warm-pool arm (`qsize 0`, `_threads 1`, item ran) and a re-run of the published table's four rows, which reproduced exactly (`0/0 → 0/1 → 0/1 → 1/1`).

- **Graph (this group):** both current-shape, `1 PRODUCED / 0 MENTIONS` each ⇒ `otherRemaining = 0` ⇒ full-node `DETACH DELETE`.

**`c1f7a6b2-3d54-4e9a-b0c7-9f2e81a4d63b` — promoted (`claude/analyst/review-techniques.md`, folded onto §*"A 'this already exists' claim is a grep away from confirmation"*; unled by the brief).** Two candidate homes were read whole before choosing. §*"Verifying a 'copied verbatim' text-block claim needs a programmatic diff"* (`:190–205`) already prescribes the same instrument — whitespace-normalise, then diff — but for a different question: whether two strings are **identical**, where a wrap causes a dropped character. The entry's question is a **count/existence** one, where a wrap causes a match to vanish entirely. Same instrument, different axis, which is this file's own idiom. It was folded into the grep-confirmation section rather than into item 5 of the residual section because that section's load-bearing sentence is *"One grep settles it"* over a `-rn -i` of a cited term, and this is precisely the case where it does not — and because item 5 is already the file's longest item. **Promoted as the rule with its consequence in both directions:** the entry states the undercount; the sharper half, added here, is that a *negative* grep is the reading this failure inverts, and a negative is exactly what the section warns is least re-checked. Instance figures kept as the entry's own (P12-6: 3 reported, 4 actual), not restated as a current fact about that plan.

**`00c5498f-e89d-4701-bfcf-b806ae5945cc` — promoted (`claude/analyst/review-techniques.md`, closing paragraph of §*"A grep-pinned edit table is an edit list, not a completeness proof"*).** The brief flagged this as a **process** fact and asked whether the coordination doc's own `## Four platform failures, all cleanly recovered` section or `teco.md` already publishes it. Checked with a whitespace-normalised scan across `teco.md`, `architect.md`, the coordination doc and `review-techniques.md`: **it does not.** That section establishes *establish actual state before acting*, *prefer resuming over re-dispatching*, and the append-before-mutate ordering — all about what a coordinator does **after** a kill; nothing about how a delegate should hold a measurement **before** one. Routed to `analyst` rather than to `teco` on the receiving artifact's scope: it is a *measurement-conduct* rule for whoever runs a residual sweep, which is what this section is, and its second payoff is the paired-binary control — the operational form of the `/usr/bin/grep` promotion made in `claude-code.md` this same unit, now cross-referenced from here rather than restated. Both of the entry's other payoffs kept as clauses (self-checking instrument; no placeholders in a document a coordinator commits by explicit path). The `grep -rn` vs `-rEn` figure is the entry's own and is pinned to its date, not asserted about any current tree.

- **Graph (this group):** both current-shape, `1 PRODUCED / 0 MENTIONS` each ⇒ `otherRemaining = 0` ⇒ full-node `DETACH DELETE`.

**`9b2e4d70-51ac-4c38-8f6a-1d3e7c05ab94` — HALF promoted (`claude/tdd-engineer/guard-testing-techniques.md`, folded onto §*"When the docstring states SEMANTIC reach and the body does a SYNTACTIC match"*; unled by the brief).** That section was read **whole** (`:177–230`) and it already carries the entry's mechanical half in its own words: *"an allowlist keyed on **names** rather than **sites** is blind to a second use of a name already in the list"*, the site-qualification remedy, and *"write the exemption as an equality, not a subtraction … it reddens in the other direction too, on an exemption left behind by a raise that is gone."* It even cites the entry's own file and constant. So the *name-set-equality-is-rename-safe* clause is a discard. What no sentence there covers is the entry's second clause — the **reason string beside** the entry is not covered by any of those mechanisms; equality protects the key set, and nothing protects the prose inside a key's value. Routed to `tdd-engineer` rather than `analyst` because the whole family (guard shape, exemption shape, what a guard can and cannot check) lives in that KB and this is its next member. **Re-derived at the entry's own sha, not the working tree** (`git show fc2b43b:…`, `falkor-chat/` read-only per the brief): both cited line numbers are **exact** — `STOREFRONT_RAISES_TODAY` at `:3939`, `NON_FAMILY_RAISES` at `:3995` — and the `RuntimeError` value does explain the name over two legs, `services._dispatch_write` and `Storefront.enqueue_turn`, with the second conceded request-reachable in the comment's own words. Given how often this pass has found a correct claim wearing a wrong citation, the exactness is worth recording. The promoted text states the **rule** (per-site reasons, pinned like the key set) with the instance as its worked case, and closes on the generalisation the entry supplies and the section lacks: a reason a guard cannot check is documentation, and it needs a reviewer rather than a test.

- **Graph (this group):** current-shape, `1 PRODUCED / 0 MENTIONS` ⇒ `otherRemaining = 0` ⇒ full-node `DETACH DELETE`.

- **Unit totals:** **6 promoted (2 of them halves), 2 discarded, 0 kept open**, all 8 `architect` nodes `DETACH DELETE`d (plus `cobb`'s `2746ee65…`, logged in `claude/cobb/kaizen/history.md`). **`architect` closes at 0 produced / 0 mentioned.** No `MENTIONS` tag added — nothing was left open for another agent's pass.
- **Budget**, measured on the shipped files against `a1e2234`: `skills/agent-standards/claude-code.md` **9,831 → 10,269 w** (+438), sections **7 → 7**, three lines over 700 characters at `:260–262` — **all three pre-existing at `a1e2234` at identical lengths** (824/1,168/1,033) and untouched here; `skills/python-web-quirks/SKILL.md` **7,675 → 7,911 w** (+236), sections **23 → 23**, 0 lines over 700; `claude/analyst/review-techniques.md` **12,156 → 12,489 w** (+333), sections **30 → 30**, 0 over 700; `claude/tdd-engineer/guard-testing-techniques.md` **2,545 → 2,794 w** (+249), sections **3 → 3**, 0 over 700. `claude/architect/architect.md` **untouched**; `claude/AGENTS.md` **untouched at 2,491 w**; `claude/README.md` and `skills/README.md` untouched — no catalog line earned, since every promotion is a fold into existing material and no knowledge base was created, renamed or re-scoped (`cobb`'s U39 rule).
- **Docs touched:** `skills/agent-standards/claude-code.md`, `skills/python-web-quirks/SKILL.md`, `claude/analyst/review-techniques.md`, `claude/tdd-engineer/guard-testing-techniques.md`, and the four `kaizen/history.md` files (`architect`, `cobb`, `analyst`, `tdd-engineer`).

## 2026-09-09 — Kaizen distillation, `architect` chunk 1 of 2 (the seven entries dated 2026-09-07): 4 promoted, 3 discarded — **every promotion landed in `analyst`'s KB, none in `architect`'s own artifacts** (U40)

- **What:** U40 of `claude/docs/plans/kaizen-distillation2-coordination.md`, split on the date boundary — the eight entries dated 09-08/09-09 are U41's and were left untouched. **Six of the seven were variations on one theme** (grep-based done-conditions and residuals in plans), and that theme already had a home: `claude/analyst/review-techniques.md`'s section *"A grep-pinned edit table is an edit list, not a completeness proof"*, whose numbered list of six residual failure modes was read **whole** before any disposition. **Zero new items and zero new sections** — every promotion is a fold into an existing item or into the section's "Two derived checks" paragraph.
- **Routing-first held for the third unit running.** `suggestedHome` again predicted **0 of 7** homes: three said `prompt`, three `knowledge base`, one `project docs`; the actual outcome was 4 folds into one *other* agent's knowledge base and 3 discards. The producer knows the fact, not the shelf.

**`f4ee08b9-ae99-48cb-b88c-85b6b49fb065` — promoted (refines item 2).** The entry (grep the attribute a newly added union member *lacks*, not the type name) is item 2's own rule; its `isinstance`-pins-the-variable half and its `"continuous"`-string half are both already in item 2's body, re-derived there at `c523a35`. What is **not** there is a *selection rule*: item 2 said "enumerate by the attribute the ship criterion actually reads", which presumes you already know the ship criterion, where the entry gives an attribute derivable from the type change alone. Folded as one clause; the storage-tag literal was added as an enumeration *target* rather than only as an example of what a name-grep misses.

**`05192600-9a74-4baa-9dc3-08e964cab6b3` — promoted (refines item 5), and it repairs the section's own opening line.** Item 5 was rewritten in U39 and already opens *"A residual counts lines in files, not occurrences in code"*, carrying the self-contradictory done-condition. The entry — captured against the **pre-U39** text — still adds three things U39's rewrite does not have. (1) The **consequence**: the contradiction is not merely a false failure, because the cheap resolution is to skip the test, leaving the behaviour pinned by nothing — evidenced at `docs/reviews/small-model-benchmarking-impl.md:1545` (P5-3), where a mutation carrying a one-name tolerance list passes all **472** tests *because* no test may name the key (suite baseline confirmed at `:1381`). (2) A **third shape**: a residual written as a regex over definitions matches test *function names* — re-derived by construction, `grep -rEn 'def [A-Za-z_]*(percentile|quantile)'` matches `def test_percentile_rejects_a_float_level`, and only under `-E` (the identical pattern as BRE exits 1, which is the section's own dialect trap arriving one item earlier). (3) The **missing plan-side remedy**: path-scope the residual and prescribe in the plan which file the assertion lives in — shipped at `docs/plans/small-model-benchmarking.md:3444`, so the remedy is real rather than hypothetical. Item 5's first shape now covers both directions, which is why the section's opening sentence — flagged in the coordination doc's Follow-ups as saying *"Six ways the residual **passes**"* over an item that is a false *failure* — was corrected here as the one-line fix it was left as, not as a restructuring.

**`f6119439-7a86-438f-9072-1a488b46e137` — DISCARDED: the fact is FALSE, and the corrected fact was promoted in its place.** The entry claims *"GNU grep: `--include` does NOT suppress a file named explicitly on the command line."* Re-derived at four arms in `model-bench/` with real GNU grep 3.11 (`/usr/bin/grep`, not the harness shim): `--include='*.py'` returns the explicitly named `tests/conftest.py` (rc 0), but `--include='*.txt'` returns **nothing, rc 1** — with `-r` and without it. The entry observed a hit only because `conftest.py` **matches the very glob it passed**; it generalised a filter that happened not to bite into a filter that cannot bite. The true behaviour is the opposite and more useful: `--include` filters by **base name regardless of how the file arrived**, so a residual scoped by naming a file explicitly returns a **clean zero for the wrong reason** whenever that file's base name misses the glob. That corrected constraint was folded into item 5, where it is load-bearing for `05192600`'s path-scoping remedy. Verified identical on the harness's `ugrep` shim, so nothing here turns on which grep ran. **Reported, not fixed (outside remit):** `docs/plans/small-model-benchmarking.md:3272` states the false version as a parenthetical justification — harmless today, because that command's explicit file *does* match `*.py`, but it is a correct command shipping with a false mechanism, and it would license an unsafe re-scoping later.

**`b1f2c7a4-3d59-4e18-9f60-7a2c5d8e41bb` — half promoted (diagnosis discarded as published in item 6; plan-side remedy folded into it).** Item 6 (*"Cross-table collisions survive per-table discipline"*, promoted in U23 from `analyst`'s near-twin `b1f0c7a4…`) was read **whole**, not against a summary. It already carries both halves of this entry's diagnosis in its own words — a residual driven to zero by a *different* table's edit on the same line, and a residual whose baseline is an **intermediate** state, passed alike by a faithful implementation and by one that skipped both edits. What the `architect` capture carries that the `analyst` one did not is the **remedy from the plan author's chair**: item 6 stops at the reviewer's move (*judge the residual as a conjunction with the first edit-set's own residual*), which changes how you read a plan and not the plan. The entry's *"the residual property is a property of the ROUND, not of a table"* plus its three concrete requirements — colliding tables name each other on both rows, fix their order, restate the later residual over the surviving post-edit spelling **re-derived, not merely re-scoped** — were absent. The `re-derived, not re-scoped` distinction is the sharp part and guards against exactly the wrong fix (narrow the path, leave the stale pattern). Premise spot-checked rather than restated as a figure: `stats.py:159` at `5878014` is `return _percentile(means, 2.5), _percentile(means, 97.5)` — one shipped line carrying both literals, `git grep -cF` reading `1` for each. Precedent for a plan-side remedy living in a reviewer's KB: item 5 already prescribes one (*"plan-side, restate the behaviour by key-set or complement"*).

**`b7f3a1c2-5d84-4e19-9a06-3c2e8f14d7b0` — DISCARDED: already published, verbatim in substance.** The entry's rule *is* the first of the section's **"Two derived checks"** (read whole at the promotion site): *"A residual command must be re-asked against every implementation the same table authorises — an authorised literal branch can re-add the very string the residual asserts to zero."* Same rule, same plan-gate lineage (Passes 5–6 of the same document). The only clause not already present is the behavioural consequence *"trains the implementer to override the done-condition"* — which the entry supplies **no observation for**: its evidence is two greps re-run at `5878014`, not an observed override. Under §5 step 2 (re-derive the fact yourself) an unevidenced assertion about how an implementer will react is not promotable, so this is a full discard rather than the half it first looked like.

**`0a4b6b2e-8737-4dbd-818a-5e4502674ed6` — promoted (sits OUTSIDE all six items; extends derived check 2).** The odd one out and the only non-grep entry: a specialist note (`-ml.md`/`-graph.md`) owns a function signature, the plan owns a required-with-no-default argument on one of that signature's callees, and a newly note-specified producer inserted between them takes no parameter to carry it. Its nearest kin is derived check 2 — *"a required, no-default parameter added to a public function breaks that function's call sites, which the defining token never reaches"* — of which this is the **cross-document** case, and the reason it is worth its own clause is that **no grep can find it**: neither document names the other's term, so `architect.md`'s existing sweep-by-grep rule is structurally blind to it. **Ruled by whose decision it changes, per the brief.** It reads as `architect`'s (the plan author does the fold), but the rule is phrased as a *check* — "check every plan-mandated required argument on the callees of any newly specified producer" — and a check is the reviewer's currency; the whole point is that it is invisible to the author's own sweep. Two further reasons it did not go to `architect.md`: that file has **no knowledge base**, so the only landing site is the always-loaded prompt (highest bar, and this fires in a small minority of sessions); and the one bullet it would fold onto, `architect.md:51`, is **1,617 characters** — the file's longest line, already recorded in this log (U31) as wanting a compaction pass, the same defect shape as the `tdd-engineer.md:40` follow-up the brief bars adding to. **Promoted as the rule, not the instance:** the plan's own v1.13 changelog already records the instance (*"Rule 8's parameter list carries no `support`, so `continuous_verdict()` has nothing to forward as Table E's required `clamp`"*), and records it as **non-blocking and bounded** — at `designEffect == 1.00`, `clamp=None` and `(-1.0, 1.0)` return the same interval. So the entry's *"the requirement becomes unsatisfiable"* is true in form and harmless in effect here; the promoted text says so, because a check whose only live instance is harmless is exactly the check nobody runs by accident.

**`73df8a0d-f63d-4a3a-b7ce-ac75b5607536` — DISCARDED: already published, in the entry's own words, in the docstring of the very function it is about.** The only `falkor-chat` entry in the chunk, and the brief's steer to check before assuming it needed a home was right. Both halves verify: `services.py:2085` raises `WorkflowRunNotFoundError` with the message *"cannot start run: snapshot … has no START or trigger message … is missing"*, and an AST attribution of every raise site of both classes puts `WorkflowDefNotFoundError`'s three at `materialize_def` (`:1752`), `get_workflow_def_structure` (`:1814`) and `diff_def_snapshot` (`:1869`) — exactly the three functions the entry names, none of them on a run-start path. But `start_workflow_run`'s **own docstring** (`services.py:2007–2031`) already states it: *"Raises … `WorkflowRunNotFoundError` when the snapshot (or trigger message) anchor misses — nothing is started in either case."* And the ruling is recorded a second time at `docs/plans/salesperson-ui-coordination.md:445` (that coordination's U30, committed `a3f681e`), which reached the same conclusion and cites the same docstring. **A derived promotion was drafted and then withdrawn.** Both *class* docstrings in `repository.py` are narrower than reality — `WorkflowDefNotFoundError`'s names 1 of its 3 raise sites, `WorkflowRunNotFoundError`'s omits the start case entirely — which looked like a clean reviewer's check ("enumerate raise sites, don't trust the class docstring"). It does not survive: the docstring a reader actually consults is the **function's**, and that one is complete and correct. The motivating premise was false, so nothing was promoted.

- **Graph:** all seven current-shape, `1 PRODUCED / 0 MENTIONS` each ⇒ `otherRemaining = 1 + 0 − 1 = 0` ⇒ full-node `DETACH DELETE` for every one. No `MENTIONS` tag was added: nothing was left for another agent's future pass, and the coordination's own Follow-ups record `MENTIONS` as a queue with no consumer.
- **Budget:** `claude/analyst/review-techniques.md` **11,568 → 12,045 w** (+477), sections **30 → 30** (no new section), **0 lines over 700 characters**. `claude/architect/architect.md` **untouched**. `claude/AGENTS.md` **untouched at 2,491 w**. No catalog line earned: no knowledge base created, renamed or re-scoped, and every promotion is a fold into existing material (`cobb`'s U39 rule).
- **Docs touched:** `claude/analyst/review-techniques.md`, `claude/analyst/kaizen/history.md`, `claude/architect/kaizen/history.md`, `claude/cobb/kaizen/history.md`.

## 2026-09-09 — `architect.md`: a step row's SCOPE column is a build instruction, swept separately from its done-condition (inbound `MENTIONS` promotion, U31)

- **What:** U31 of `claude/docs/plans/kaizen-distillation2-coordination.md` — the **orphan-backlog** unit, the first shaped by *edge* rather than by producer. The 11 nodes it covers carry **0 `PRODUCED` edges** and are alive only on `MENTIONS`; every earlier unit was organised by producer, so none of them could ever have been reached. `architect` carried one of the 12 edges — `teco`'s U17 tagged it, having promoted the coordinator-facing half into `teco.md`.
- **`b28c5e43-1f76-4d92-a305-7c6e1b9f4a82` (2026-09-02) — promoted, folded into the existing "Compress by pointer" bullet; zero new bullets.** The nearest existing coverage (same bullet's closing sentences) prescribes sweeping *downstream references* after a revision note and enumerates whole sections — delegation table, stage file lists, AC test table, risks. It does **not** say that a single step row has two independently binding columns, which is the entry's point: the SCOPE column is the *build instruction*, so a trim that sweeps only DONE-CONDITIONS leaves the removed contract still commissioned, frequently a few lines above a done-condition that now contradicts it. Added with the method that catches it — **grep every removed or renamed term across the whole plan and rule on each hit**, rather than editing the sites you reasoned about.
- **Verified by re-deriving the shape, and its bound stated.** The entry's own instance (`salesperson-ui` plan v1.10, `messageCount`/`cartTotal`/`orderStatus` left in an S12d SCOPE cell) is not reproducible from a plan revision no longer at `HEAD`, so what was promoted is the **rule**, which follows from the structure of a step table and from the entry's uncontested framing as the same class as the repo-wide-rename rule already in this file. No figure from the entry was restated as fact.
- **Graph:** 0 `PRODUCED` / 1 `MENTIONS` ⇒ `otherRemaining = 0` ⇒ full-node `DETACH DELETE`.
- **Budget:** `architect.md` **1,902 → 1,978 w**. The host line is now **1,617 characters** (was 1,150) — the file's longest, and worth a compaction pass; noted rather than fixed here, since splitting that bullet is an editorial change, not a distillation one.
- **Docs touched:** `claude/architect/{architect.md,kaizen/history.md}`.
## 2026-09-08 — `kaizen_team` retraction pass: false CPython deadlock entry `7f3c1a92` cleared, its correction `2b8d40f1` promoted (1 discarded/retracted, 1 promoted, 0 kept open, 0 `MENTIONS`-tagged)

- **What:** `cobb` processed the two same-session `kaizen_team` entries `architect` wrote on
  2026-09-08 while amending `docs/plans/salesperson-ui.md` S9 (atomic per-participant turn
  reservation) — a false mechanism claim and the correcting entry `architect` wrote after `analyst`
  disproved it, which `architect` could not clear itself (curator-only shape).

  **Retracted and cleared (`7f3c1a92-5b64-4a7e-9d81-2c0a6b4e8f13`).** The entry claimed that
  holding an application lock across `ThreadPoolExecutor.submit()` "can deadlock at interpreter
  exit", because `submit()` takes `_global_shutdown_lock` "which `_python_exit` holds while joining
  worker threads" — a three-party cycle. **The mechanism does not exist.** Re-derived from source
  rather than taken on the correction's word: CPython 3.12.3
  `/usr/lib/python3.12/concurrent/futures/thread.py:23-31` shows `_python_exit` entering
  `with _global_shutdown_lock:` for the single statement `_shutdown = True` (line 26); the
  `q.put(None)` loop (28-29) and the `t.join()` loop (30-31) are both *outside* it. `shutdown()` is
  the same shape — `self._shutdown_lock` closes at line 235, its join loop runs at 236-238.
  Independently staged twice: with the application lock held across `submit()`, a worker blocking on
  that lock, and the main thread falling off the end, `submit()` returned in **0.2 ms** and the
  process exited **rc 0** after exactly the hold time (0.52 s wall for a 0.5 s hold; 3.03 s for a
  3.0 s hold). Held forever it never exits — the same delay at its limit, not a cycle. The entry was
  not merely imprecise: acted on, it would have banned a sound design (`submit()` under a
  reservation lock) on a hazard that is not there. Discarded, not narrowed — the true fact it
  gestured at is the correcting entry's, promoted below.

  **Promoted (`2b8d40f1-6c17-4f52-8a3e-91d7c5b0ae44`), split by audience — both halves verified
  before promotion, neither taken from the entry's prose.**
  - **The corrected CPython fact → `skills/python-web-quirks/SKILL.md`, new section** (+1 clause in
    the frontmatter `description`, which is the routing signal and enumerates every entry). The
    right home: a general Python/threading fact, not a `falkor-chat` one, and the on-demand skill
    already carries this class. Section states the lock scopes with line numbers, the two staged
    measurements above, and the honest residual — `_python_exit` joins workers *unlocked*, so
    interpreter exit is bounded below by the lock's hold time. A latency cost, unbounded only if the
    lock is never released.
  - **The method lesson → `architect.md` Guardrails, folded into the existing "Honesty about
    uncertainty" bullet** rather than filed as a ninth bullet — that bullet already said
    *"distinguish what you verified from what you're inferring"*, and the lesson is the operational
    form of exactly that rule, so a new bullet would have restated a neighbour (§7 cognitive-load
    and prompt-waste both favour the fold). The gap it closes is real: the prompt's existing rules
    cover the detail you *know* you are unsure of (line 35, "verify external specifics"), and this
    failure is the opposite — `submit()` was read from source while `_python_exit`'s lock scope was
    asserted from memory **in the same sentence**, producing prose that reads uniformly verified with
    no seam a reviewer can see. Bullet now carries *"a mechanism claim is only as verified as its
    least-verified clause"* and the operational test: open every function a justification names, not
    just the entry point. Judged "most sessions" — writing multi-mechanism justifications is routine
    for this agent. **+80 words.**

- **Why:** the false entry was flagged for clearing so it could not be promoted in the interim, and
  a deletion is curator-only. The correction was judged on its merits, not accepted on `architect`'s
  or the requester's say-so.
- **Notes:** both entries were current-shape (one `PRODUCED` edge each, no `MENTIONS`); each
  `otherRemaining == 0`, so each was cleared as a full-node `DETACH DELETE` per `agent-maintenance`
  §5 step 4. Neither is about another agent, so no `MENTIONS` tagging applied.

## 2026-09-07 — `kaizen_team` distillation pass 2, chunk B: the remaining 6 current-shape entries (2026-09-03, plus one written 2026-09-07 during chunk A's run) — 6 promoted, 0 discarded, 0 kept open, 0 MENTIONS-tagged

- **What:** `cobb` processed `architect`'s six remaining `kaizen_team` entries
  (agent-maintenance skill §5), closing this agent out. All current-shape; the current-shape read
  returned exactly six and the legacy read returned **zero** entries graph-wide, so the legacy read
  was skipped per the brief. Every fact was **re-derived live** — FastAPI 0.139.0 in
  `falkor-chat/server/.venv`, `grep` over `model-bench` at `5878014`, the plan document itself, the
  LM Studio server, and this session's own scratchpad — not merely checked against its citation.
  Two entries proved **narrower or wider than stated**; both were corrected before promotion.

  **Promoted (6).**
  - **`f1b987fb` + `a4079bf3` (2026-09-03 and 2026-09-07, plan-document verification method) →
    `architect.md` Guardrails, ONE new bullet.** The brief's suspicion was right: they are the same
    underlying rule seen twice, so they were folded rather than filed as two near-duplicate bullets
    beside chunk A's `3c1436d8` promotion. Verified: `docs/plans/small-model-benchmarking.md`
    §3.4.1 now reads *"The forbidden set is a derivation, never a list"* with the v1.5/v1.9
    provenance, confirming both the original defect (a 14-name list beside prose saying "every model
    field", certified by two analyst passes) and its repair; and in `model-bench` at `5878014`,
    `grep -rFn armKind` returns 54 lines (26 `modelbench/`, 24 `tests/`, 4 `docs/`) against
    `grep -rFn arm_kind`'s 18, **all 18 in `tests/`**, with only 3 lines carrying both — the
    disjoint-vocabulary claim reproduces exactly. The plan's v1.11 note confirms rule 5 was
    restated as a *no-forgetting guarantee over token-carrying sites*, not completeness; the bullet
    states that honestly rather than as an absolute. New bullet: *"A completeness claim must be
    derived, not transcribed — and its check must be able to fail."* Judged "most sessions" because
    writing an enumeration beside an invariant, and prescribing a grep-shaped done-condition, are
    both routine for this agent. **+120 words** (chunk A: +88; running total +208 across both
    chunks).
  - **`c4f8a2d1` (2026-09-03, FastAPI's built-in doc routes) →
    `skills/python-web-quirks/SKILL.md`, new section.** The live-verified home for exactly this
    class of fact, and in `cobb`'s remit. **Narrowed before promotion:** the entry's "FastAPI adds
    FOUR documentation routes to **every** app" is false. Enumerated against FastAPI 0.139.0 —
    bare `FastAPI()` → `/docs`, `/docs/oauth2-redirect`, `/openapi.json`, `/redoc`; but
    `openapi_url=None` → **none of the four** (the schema route gates the other three);
    `docs_url=None` → drops `/docs` *and* `/docs/oauth2-redirect` together;
    `swagger_ui_oauth2_redirect_url=None` → drops the redirect alone. They are defaults, not
    unconditional, so the count is a property of the constructor call. The section says so, and
    keeps the entry's real payload (the fourth route is the one hand-written lists miss, and the
    exemption should be derived from a bare app). The `falkor-chat` half was **already documented
    at the site that matters** — `server/tests/test_app.py:756-760` states the four routes verbatim
    with the derive-don't-enumerate reasoning — so nothing was added there; only the general
    framework fact was missing anywhere.
  - **`c1f0a7d2` (2026-09-03, `responses={...}` granularity) →
    `skills/python-web-quirks/SKILL.md`, new section.** Verified against FastAPI 0.139.0: the
    parameter is typed `dict[int | str, dict[str, Any]]`, and a live route declaring
    `{422: ..., "4XX": ...}` keeps both keys — the string form is a wildcard *range*, i.e. coarser
    than a status, never finer. There is no declaration-side key for an error token or a `field`,
    so two 422s differing only by field genuinely collapse. The **plan-specific** conclusion is
    already settled in `docs/plans/salesperson-ui.md` v1.20 (*"compares at **status**
    granularity"*, line 1666) — what was undocumented anywhere is the framework fact and its
    consequence for any declaration-reading contract gate.
  - **`3f1c9a76` (2026-09-03, parallel subagents share one scratchpad) →
    `skills/agent-standards/claude-code.md` § "Bash tool environment".** Per the brief, weighed for
    an agent prompt and **rejected**: it does not change behavior in most sessions, and the
    always-loaded bar is the highest one. It is a verified Claude Code harness fact, which is what
    `claude-code.md` is for, and that file is in `cobb`'s write remit. **Widened before promotion,
    from this session's own evidence:** `$CLAUDE_CODE_SESSION_ID` inside a subagent's Bash env
    resolves to the **parent's** session id, so the scratchpad path is keyed by the parent, not by
    the delegate — and this run found chunk A's working files (`arch_hist.md`, `ds_hist.md`,
    `patch_*.py`, `v0models.json`, 08:10-08:27) still sitting in "its own" scratchpad at 08:33.
    The hazard is therefore **not limited to parallel dispatch**; sequential reuse leaves stale
    files a later delegate can read as its own. The entry's parallel incident is real and cited
    (`docs/reviews/small-model-benchmarking-impl.md` Appendix C.5, lines 964-971). **No `MENTIONS`
    tag:** it is a harness fact binding every agent that writes a scratch file, not a fact about
    `teco` or any one agent's discipline, and a durable greppable reference entry serves better
    than resurfacing a raw node.
  - **`6f2b1d94` (2026-09-03, LM Studio measurement surface) → folded into the **existing** section
    of `claude/data-scientist/lm-studio-model-notes.md` that chunk A created from `443cc4cc`, per
    that pass's hand-off — no second section added.** Re-verified live 2026-09-07 without loading a
    model (chunk A's constraint still holds): `GET /api/v0/models` → 19 models, 10 keys each
    (`id`/`object`/`type`/`publisher`/`arch`/`compatibility_type`/`quantization`/`state`/
    `max_context_length`/`capabilities`); `GET /v1/models` → 19 models, 3 keys
    (`id`/`object`/`owned_by`); `command -v lms` exits 1; `lms.exe ps --json` → `[]` in
    0.30/0.31/0.32 s over three runs. Three clauses added, all genuinely new to the section:
    endpoint timings (1.6-6.5 ms over six calls, `/v1/` **no faster** — so the entry's implied
    "v0 is the fast one" is wrong; the differentiator is content, not cost), the ~0.30 s
    WSL-to-Windows `lms.exe` subprocess price per call, and JIT auto-load. **The entry's 21.068 s
    cold-call figure was deliberately not banked:** `docs/plans/small-model-benchmarking-ml.md`
    §11.4 measures a 3 625.0 ms cold load on the same box, and plan v1.10 records that *"no
    load-cost figure is left anywhere the design is sized against, the two measured cold loads on
    this box differing by ~6x"* — so the KB states the behavior and the stable invariant
    (LM-Studio-side `ttft` **excludes** the JIT load, wall clock includes it) and cites §11.4 for
    the numbers. Logged in `claude/data-scientist/kaizen/history.md` too.

  **Discarded (0). Kept open (0).** No `K-` item was opened: every entry landed in a destination
  inside `cobb`'s write remit, so nothing was left unresolved. K-004 and K-005 (opened by chunk A)
  are untouched and still 🔵.
- **Dedup check.** Every one of the six `entryId`s was grepped against this agent's `plan.md` and
  `history.md`: zero hits. Confirmed by **date and subject**, not by 8-char prefix — `3f1c9a76`
  (2026-09-03, subagent scratchpad) is one character from chunk A's already-cleared `3f1c9a52`
  (2026-09-02, no root pytest config), and they are different entries.
- **Graph ops (in order).** No `MENTIONS` tag was added in this pass, so the ordering invariant had
  nothing to sequence against; the count-and-decide read still ran per entry before deciding.
  All six returned `producedEdges=1, mentionEdges=0` ⇒ `otherRemaining = 0` ⇒ full-node
  `DETACH DELETE`. This `history.md` entry landed and was confirmed **before** any graph mutation.
- **Docs touched:** `claude/architect/{architect.md,kaizen/history.md}`,
  `skills/python-web-quirks/SKILL.md`, `skills/agent-standards/claude-code.md`,
  `skills/README.md` (catalog row re-synced — the two new FastAPI facts, plus two pre-existing
  omissions the row had already drifted on), `claude/data-scientist/lm-studio-model-notes.md`
  + its `kaizen/history.md`, `claude/cobb/kaizen/history.md`. No `claude/README.md` or
  `claude/AGENTS.md` change: nothing here alters what this agent does or when to route to it.
- **Why:** unit U7b of `claude/docs/plans/kaizen-distillation2-coordination.md` (pass 2; pass 1's
  record is `claude/docs/plans/kaizen-distillation-coordination.md`). `architect` is now clear of
  raw capture.
- **Plan items:** none opened; K-004/K-005 unchanged.

## 2026-09-07 — `kaizen_team` distillation pass 2, chunk A: 11 current-shape entries (2026-08-26 → 2026-09-02) — 4 promoted, 5 discarded, 2 kept open, 1 MENTIONS-tagged

- **What:** `cobb` processed `architect`'s eleven `kaizen_team` entries dated 2026-08-26 through
  2026-09-02 (agent-maintenance skill §5). All current-shape
  (`(:Agent {agentId:'architect'})-[:PRODUCED]->(:KaizenEntry)`); zero legacy `author`-property
  entries survive anywhere in the graph, so the legacy read was skipped. `architect`'s five
  2026-09-03 entries are chunk B, a separate unit — **untouched by this pass.** Every fact was
  **re-derived against live source/docs/services**, not merely checked against its own citation;
  two entries changed disposition as a result (`e3f1c0a2` is now false, `9c2a4f81` proved narrower
  than stated).

  **Promoted (4).**
  - **`b3f2a1e4` (2026-08-31, `cypher-mcp` is already multi-instance-ready) → `cypher-mcp/README.md`.**
    Re-derived: `authorize_write(cypher, agent)` (`cypher-mcp/server.py:540`) takes no host, port or
    graph argument and decides purely from the query text plus the declared slug — instance-agnostic
    confirmed, not merely asserted; `FALKORDB_HOST` is a module-level `os.environ.get` at
    `server.py:93` (read once per process); `.mcp.json` holds named entries, each yielding its own
    `mcp__<name>__query` namespace. The README documented neither multi-instance operation nor the
    reason it needs no code change — the fact lived only in `docs/plans/kaizen-team-sandbox.md`
    (§"the access path", still `Status: active`, i.e. unbuilt). Added a closing subsection to
    §"Wiring it elsewhere". *Why the README and not the plan:* the plan may be archived; the server's
    own doc is where a capability of the server belongs.
  - **`443cc4cc` (2026-09-02, LM Studio measurement surface) → `claude/data-scientist/lm-studio-model-notes.md`.**
    The live-verified KB for exactly this domain, per the U2/U6 precedent that `cobb` dispositions a
    single-target KB entry directly rather than `MENTIONS`-tagging it. Re-verified live 2026-09-07,
    every checkable clause: `GET /api/v0/models` → 200 with per-model `id`/`object`/`type`/
    `publisher`/`arch`/`compatibility_type`/`quantization`/`state`/`max_context_length`/
    `capabilities`; `command -v lms` exits 1 while `/mnt/c/Users/<user>/.lmstudio/bin/lms.exe`
    exists and answers; `lms server status --json` → `{"running":true,"port":1234}`; `lms ps --json`
    → `[]`; `lms load --help` documents `--estimate-only` and offers `--context-length` but **no**
    KV-cache flag; `lms version` prints `CLI commit: <sha>` and no app version. The one clause not
    re-derived is the `/api/v0/chat/completions` `stats`/`model_info`/`runtime` response shape —
    confirming it live would require **loading a model on the shared LM Studio server**, which this
    pass must not do; verified instead against LM Studio's own REST docs
    (`lmstudio.ai/docs/developer/rest/endpoints`), which quote all three field groups verbatim. The
    KB section says so. Logged in `claude/data-scientist/kaizen/history.md` too.
    **Note for chunk B:** entry `6f2b1d94` (2026-09-03) covers the same surface with timing data and
    is a refinement, not a contradiction — fold it into this section rather than adding a second one.
  - **`a71d5e6c` (2026-09-01, a doc-kind added mid-document must be swept into every table in the
    family) → `architect.md` Guardrails.** Verified the incident and its repair:
    `docs/plans/doc-reference-convention.md` v1.5 added `manuals/` to the kind list and the
    family-chain rule but missed §9.6's own "who performs the `archived` flip, by kind" table — the
    table that document explicitly calls *"the one place root `AGENTS.md` copies from"* (`:1491`) —
    caught by `analyst` and fixed in v1.5.1 (§9.6's `requirements/*` row now reads
    `requirements/*`, `manuals/*`). The document is repaired; the *generalization* was stated
    nowhere. Promoted as an **extension of the existing sweep sentence's list**, not a new rule:
    "…and every other structurally-identical enumeration, including a routing or authority table
    your revision note never reasoned through." *Not* promoted to root `AGENTS.md`: adding a doc kind
    is a once-a-year event, and that file is the most expensive prose in the repo.
  - **`3c1436d8` (2026-09-02, a plan's own grep-based verification is defeated by prose that quotes
    the wrong value to disown it) → `architect.md` Guardrails, one new bullet.** Verified the
    mechanism from first principles (a grep matches a literal; it cannot see intent) and against the
    cited v1.17 sweep of `docs/plans/salesperson-ui.md`. Promoted because both halves of the trigger
    are routine for this agent, not rare: `architect` writes grep-shaped done-conditions, and root
    `AGENTS.md` collision rule 5 *requires* revising a pre-approval plan in place with a dated
    revision note — which is exactly the prose that re-quotes the old literal. Rule plus one clause
    of why; the incident stays here.

  **Discarded (5).**
  - **`b3f1a9c2` (2026-08-26, message role is a strict user/assistant binary) → already documented
    and already fixed.** Verified `services.py:843` still reads
    `role = "user" if actor_kind == "User" else "assistant"` (the entry's cited `:822` is off by
    one). But the derivation is already in `falkor-chat/docs/DESIGN.md` §1.2/§5.1 (`:53`, `:222-223`)
    and `docs/QUERIES.md:156`; and the entry's forward-looking half — that *any* two consecutive
    same-side turns, not just the CONTEXT tail, violate a strict-alternation template — was closed by
    K-048 the same day the entry was written: `_append_turn` coalesces every same-role run, the
    `_assemble_messages` docstring (`executor.py:1252-1257`) states the invariant, and
    `docs/HISTORY.md` 2026-08-26 records a **sibling-shape test** (two consecutive user turns, no
    assistant between) promoted from recommended to mandatory. Nothing left to add.
  - **`e3f1c0a2` (2026-09-02, `start_workflow_run` SILENTLY IGNORES the caller `run_ctx` on the chat
    path) → discarded, now false.** This is the U4/U6 class the pass exists to catch: promoting it
    verbatim would have shipped a false claim into `falkor-chat`'s docs. Read
    `services.py:2015-2085` today — the chat branch builds
    `initial_ctx = self._dump_ctx(self._chat_start_ctx(caller_ctx, thread_id=thread_id))`, i.e. the
    caller's ctx **merged alongside** the `threadId` anchor, and the reserved-key rejection
    (`_reject_reserved_keys`) plus the size bound are **hoisted ahead of the branch** so both paths
    are screened by one call site. The docstring says so explicitly. salesperson-ui S2 shipped exactly
    the change the entry said would be needed; the entry describes the pre-fix tree.
  - **`9c2a4f81` (2026-09-02, falkor-chat binds ALL background machinery to `config.WS_ID`) →
    discarded: narrower than stated, and its actionable conclusion is already documented more
    strongly.** The mechanism is real — `app._sweep_loop` calls `context_provider()` per tick
    (`app.py:179`) and `_lifespan` calls `services.ensure_actor(provider())` (`:358`) — but
    `provider` is `context_provider or config.get_context` (`:298`), i.e. **injectable**, and
    `create_app` deliberately routes even the storefront's workspace through `provider().ws` rather
    than `config.WS_ID` "so an injected `context_provider` pins the storefront too" (`:321-324`). So
    "not to any per-request workspace" holds for the *default* wiring only. The conclusion the entry
    drew — never add a second workspace env var — is settled in `docs/SERVER.md` §1.3 in stronger
    terms ("There is no `FALKORCHAT_DEMO_WS`, and there never will be", with the reasoning and a
    tripwire in `tests/test_storefront.py`).
  - **`f3c9a2e1` (2026-09-01, a drifted spec catches up via a versioned changelog entry citing
    upstream wording verbatim) → already documented, both halves.** The mechanism half is root
    `AGENTS.md` collision rule 5 (revise in place, bump `Version:`, add a dated revision note — one
    dated line, not a narrative). The verbatim-citation half is stated in the very document the entry
    was written about: `docs/plans/doc-reference-convention.md` §9.6 (`:1491`) — *"root `AGENTS.md`
    copies from here; implementers copy, they do not paraphrase"* — and its v1.5 changelog already
    practises it ("already-shipped wording verbatim"). Nothing generalizable was left unstated.
  - **`3f1c9a52` (2026-09-02, no root pytest config) → promoted, then the entry discarded as
    superseded by its own downstream work.** Re-verified: `git ls-files` shows no root
    `pyproject.toml`/`pytest.ini`/`setup.cfg`/`tox.ini`. It was already published twice by the work
    that produced it — `model-bench/README.md:44-45` and `model-bench/AGENTS.md:130-131` — but only
    inside one component, while the trap is monorepo-wide. Rather than open an item, extended root
    `AGENTS.md`'s **existing** opening sentence ("there is **no root-level build/test script**") with
    the pytest half and the consequence, since that sentence already occupies the exact slot and the
    file must stay small.

  **Kept open (2)** — both name a target outside `cobb`'s write remit; see `plan.md`.
  - **`a3f5c8e2` (2026-09-01, `llm.py` bare-call vs. `tool_calls`-envelope probe collision) →
    `plan.md` K-004.** Re-derived and **still true after K-035**, which is the non-obvious part: the
    K-035 guard suppresses `_normalize_tool_call` when content is bare-call-shaped, but the
    `{"tool_calls": [...]}` branch runs *before* that guard (`llm.py:311-320`), so
    `x({"tool_calls":[...]})` still resolves as a native envelope purely by probe order. Confirmed
    no coverage: the six K-035 pins test `name`/`action`/`tool` shadowing, and both
    `tool_calls`-envelope tests use non-bare-call content (`test_llm.py:578-628`). The docstring
    names a *different* residual. Targets: a `falkor-chat/docs/BACKLOG.md` test-gap item and one
    docstring sentence — both `falkor-chat` source/docs.
  - **`b7d5e214` (2026-09-02, `ws:acme` is not an empty scratch graph) → `plan.md` K-005, and
    `MENTIONS`-tagged to `qa-engineer`.** Re-verified both halves. `config.py:16` still defaults
    `WS_ID` to `"acme"`, and a live label count on `ws:acme` reproduces the entry's figures exactly
    (Entity 544, Chunk 87, Message 52, Document 29, Channel 2, Thread 2, User 1, ReadCursor 1) plus
    labels it did not list (StepRun 78, Step 29, WorkflowRun 21, TraceEvent 13, WorkflowDefSnapshot
    11, Agent 1). `docs/SERVER.md` §1.3 documents the tenancy *decision*, not the "the default target
    is populated" warning; target is `falkor-chat/AGENTS.md`. The entry's **second half is a
    general test-design rule in a different agent's discipline** — a done-condition of the form
    "assert every survivor by label" cannot catch an over-broad delete when the spared rows share
    labels with the targets; the assertion must be positive and name a specific seeded non-target
    row. Tagged `MENTIONS` → `qa-engineer` so it resurfaces in that agent's own pass.
- **Graph ops (in order).** Tagged `MENTIONS` on `b7d5e214` → `qa-engineer` **first**, confirmed
  committed, and only then ran that entry's count-and-decide read (§5's ordering invariant). Counts
  read per entry before deciding: ten entries returned `producedEdges=1, mentionEdges=0`
  ⇒ `otherRemaining = 0` ⇒ full-node `DETACH DELETE`; `b7d5e214` returned
  `producedEdges=1, mentionEdges=1` ⇒ `otherRemaining = 1` ⇒ **`PRODUCED` edge resolved only, node
  and its `MENTIONS` edge kept alive** for `qa-engineer`. Every `history.md`/`plan.md` edit landed
  and was confirmed **before** any graph mutation.
- **Prompt cost:** `architect.md` **+88 net words** for the two promotions (one new Guardrails
  bullet, one extension of an existing sentence's list). No catalog change: neither promotion alters
  what this agent does or when to route to it, and `claude/README.md`/`claude/AGENTS.md` do not
  enumerate guardrails.
- **Why:** unit U7 of `claude/docs/plans/kaizen-distillation2-coordination.md` (pass 2; pass 1's
  record is `claude/docs/plans/kaizen-distillation-coordination.md`).
- **Plan items:** K-004 and K-005 opened (both 🔵). No prior pass had opened an item for any of the
  eleven — checked by grepping every `entryId` against this agent's `plan.md` and `history.md`
  (zero hits; the ids are hand-shaped, so 8-char prefixes were confirmed against date and subject,
  not matched alone).

## 2026-08-25 — `kaizen_team` distillation: 6 entries (3 legacy, 3 current-shape) — 2 promoted to project docs, 1 promoted to Guardrails, 1 MENTIONS-tagged to graph-dba, 2 discarded

- **What:** `cobb` processed all of `architect`'s raw entries in the shared `kaizen_team` graph
  (agent-maintenance skill §5): 3 legacy (`author:'architect'`) and 3 current-shape
  (`(:Agent {agentId:'architect'})-[:PRODUCED]->(:KaizenEntry)`, no `MENTIONS` yet). Each
  re-derived against live docs/source, not just its citation.
  - **`e7c1a9d4` (legacy, 2026-08-21, automated-resume re-park-loop risk class) → promoted to
    project docs.** The fact (distinguish a re-park-loop risk from the CAS-race risk when an
    automated actor can trigger suspend/resume) is still true, but its evidence cited K-028 v2's
    fix (an unconditional-fallback invariant), which `docs/plans/workflow-timers.md`'s v2→v3
    revision note records as **not actually working** — it defeated `evaluate_guard`'s ordinary
    first-arrival case and was replaced by the shipped `ctx.timerFired` marker-guard. Added a
    design-review box to `falkor-chat/docs/DESIGN.md` §6.2 (after "Timer release…") stating the
    risk-class distinction as forward guidance for any future automated resume caller, citing the
    real v3 mechanism, not the superseded v2 one.
  - **`a3f8c2e1-9b7d` (legacy, 2026-08-21, derive-outside-the-lock technique) → promoted to
    project docs.** Verified still true: `_drive_loop` remains SHA-locked
    (`71055f756280`, DESIGN.md §6.2) and the sweep still derives `dueAt` fresh from
    `StepRun.startedAt` rather than writing a `WorkflowRun.wakeAt` inside the lock. The specific
    outcome was already documented; the *generalizable technique* ("before extending the lock,
    check whether the data is already written atomically elsewhere in the run's history") wasn't
    stated anywhere — added one sentence to the SHA-lock box in DESIGN.md §6.2.
  - **`a3f8c2e1-6b4d` (legacy, 2026-08-22, BACKLOG.md milestone rows compiled incrementally by
    teco) → discarded, superseded.** Verified against current `docs/BACKLOG.md`: the M7 row this
    entry cited no longer exists — the 2026-08-25 docs-convention overhaul (root `AGENTS.md`,
    commits `1d8aed7`/`b53dc2e` etc., all dated after this entry) replaced "milestone rows
    compiled incrementally in BACKLOG.md" with "a delivered item leaves BACKLOG.md entirely, not
    even as an index row — its record is HISTORY.md." The entry describes a since-abolished
    practice; root `AGENTS.md`'s current docs-convention section fully supersedes it.
  - **`c4a7d1f0` (current-shape, 2026-08-22, reconciling a plan against a diverging delegated
    -graph note) → promoted to Guardrails.** Verified against `docs/plans/document-ingestion.md`
    (the MatchSuggestion→SAME_AS reconciliation it describes). Extended the existing "compress by
    pointer" Guardrails bullet with one clause: reconciling a plan against a diverging delegated
    `-graph.md`/`-ml.md` note means rewriting the design section in place with a revision note,
    then sweeping every downstream reference for the old premise — a partial find-and-replace
    leaves the plan inconsistent for the analyst gate. Judged "most sessions" because delegating
    `-graph.md`/`-ml.md` notes and later reconciling against them is architect's own documented
    workflow (`architect.md` §4), not a one-off.
  - **`a3f0c1d2` (current-shape, 2026-08-22, FalkorDB/Redis serializes `GRAPH.QUERY` execution,
    enabling atomic check-then-act) → MENTIONS-tagged to `graph-dba`, not promoted here.** Verified
    the underlying mechanism is real (`falkor-chat/AGENTS.md` rule 4 already requires every
    HEAD/TAIL write to be one atomic `GRAPH.QUERY`, and `docs/plans/document-ingestion.md` §3.4
    /`docs/QUERIES.md` §"`create_entity_with_auto_match`" show the same pattern shipped for K-050's
    fusion blocker) — but the *general* FalkorDB-engine fact (single-command serialization is
    *why* folding check-then-act into one query closes the race, generalizing beyond HEAD/TAIL
    writes to any read-then-decide-then-write sequence) is not yet in `claude/graph-dba/falkordb-quirks.md`
    (checked: no `serializ`/`concurren`/`race` hits there). This is graph-dba's domain, not an
    architect-specific practice, so tagged `MENTIONS` rather than promoted from here — see graph
    op below. Left in the graph (PRODUCED edge resolved, node kept) for graph-dba's own
    distillation pass to route into its knowledge base.
  - **`8f3b1e2a` (current-shape, 2026-08-22, no-APOC entity-fusion-as-linking-edge pattern) →
    discarded, already fully documented.** Verified: `falkor-chat/docs/DESIGN.md` §2 rule 4
    ("No APOC, no GDS") plus §5.1/§7.1's `SAME_AS` edge model (`matchId`, `status`, `confidence`
    — the exact "model the merge as a first-class linking edge" pattern this entry describes) are
    both already shipped and documented; nothing left to add.
- **Graph ops:** tagged `MENTIONS` on `a3f0c1d2` → `graph-dba` (curator write, `agent='cobb'`);
  resolved the `PRODUCED` edge on `a3f0c1d2` and `c4a7d1f0` and `8f3b1e2a` individually (each had
  no other edge before this pass, so `c4a7d1f0`/`8f3b1e2a` fully cleared — `a3f0c1d2` kept alive
  by its new `MENTIONS` edge); `DETACH DELETE`d the 3 legacy entries outright.
- **Why:** Unit U3 of `teco`'s team-wide kaizen-distillation pass
  (`claude/docs/plans/kaizen-distillation-coordination.md`).
- **Plan items:** none opened — every entry was either promoted directly or discarded with a
  verified reason; nothing unresolved.

## 2026-08-25 — Output discipline: the once-canonical rule (prompt-waste plan, Stage D — an *addition*)

- **What:** 1,429 → 1,471 w (**+42**). Stage D is the only stage of
  `claude/docs/plans/prompt-waste-reduction.md` that adds rather than cuts; its budget is ≤~120 net
  words across this file and `analyst.md` together, and the unit landed at +92.
  - **The addition** (§ "Handoff to the implementer", immediately after "…the file paths,
    signatures, and findings the implementer needs"): *"**Stand-alone means the implementer never
    re-derives a decision — not that it appears twice.** State each once, in one canonical section;
    cite it elsewhere: a recap table cites, it does not restate; a `-ml.md`/`-graph.md` note's
    conclusion is quoted once, its rationale cited."*
  - **One alignment edit** (§ "How you work" step 4): "fold its **recommendation** into the plan"
    → "fold its **conclusion** into the plan", so step 4 and the new rule use one word for one
    thing. Step 4 previously invited exactly the restatement the new rule forbids.
- **Why:** The file held two rules pointing opposite ways with nothing reconciling them —
  § Handoff's *"it must stand alone"* (which reads as "restate everything the implementer needs")
  and Guardrails' *"Compress by pointer only what nothing else cites literally"* (which reads as
  "don't"). The plan's §2 recorded the cost: one `graph-dba` decision restated **4× (~600 words)**
  inside `falkor-chat/docs/plans/document-ingestion.md`. The resolution is that stand-alone is a
  property of *derivability*, not of repetition.
- **Gate (a) — rule inventory, addition-shaped.** Nothing was removed, so the inventory ran in the
  reverse direction: every existing class-1/2 clause checked for **contradiction** by the addition.
  The one pair at real risk — the new "state each once… cite it elsewhere" against Guardrails'
  "Compress by pointer only what nothing else cites literally" — was put to `cobb` explicitly.
  Verdict: they don't merely coexist, they **interlock in the safe direction** — the new rule
  *manufactures* citations, and being cited is precisely what triggers the guardrail's "must stay"
  protection. The more the new rule is obeyed, the less of the plan is pointer-compressible.
- **Gate (b):** not applicable — nothing removed. **Gates (c)/(d):** `cobb` §7 lint (2 majors, both
  applied below, 0 blockers); `audit-team.sh` PASS.
- **Two corrections from the lint, both free.** (1) I first wrote "a **summary** table cites, it
  does not restate"; the general word reaches this file's *step* table — the one artifact `:41` and
  `:47` twice insist must stay concrete — so an agent skimming could hollow out the table the
  implementer executes from. → "a **recap** table" (0 w, and a recap table is by definition a
  summary of something stated elsewhere, so it cannot be misread). (2) I first wrote `-ml.md` alone,
  specializing the plan's "a sibling note's"; but §2's motivating incident was a **`-graph.md`**
  note, so the rule as written did not bind the case that generated it. → `-ml.md`/`-graph.md` (+1 w).
- **One clause the plan mandated and this unit deliberately did *not* add** — *"revision history is
  one dated line, not a 'Revision note' narrative."* **Relocated to Stage E** (root `AGENTS.md`
  collision rule 5) and struck from the Stage D spec in the same commit, so the decision is on the
  record rather than in session context. Two reasons: this file mentions revision notes **nowhere**,
  so the clause would have introduced the topic solely in order to qualify it; and the rule binds
  **six** agents that revise pre-approval documents (`architect`, `teco`, `tico`, `qa-engineer`,
  `data-scientist`, `graph-dba`), whose single mandated home is rule 5 — not one of the six.
- **Plan items:** none. Feeds the plan's **finding 15** (waste is created at specification time;
  the check is "how many agents does this rule bind?").
- **Watch (observation window open):** the `CPG:` line, the plan-document convention, and
  "don't hand-wave" are unchanged; what to watch is the *new* rule misfiring — a plan that cites a
  canonical section the implementer then cannot find, or a step table thinned to citations. Neither
  appeared in review; both are the failure modes this addition could plausibly cause.

## 2026-08-23 — Freshness-clause grammar fix (Stage B wave 2 micro-shape)
- **What:** "a `teco`-issued brief that states the graph's freshness, take it as given" → "when a `teco`-issued brief states the graph's freshness, take it as given" — closing the hanging-topic construction cobb's wave-1 lint flagged as minor; applied uniformly across all files carrying the clause. No rule change; both branches intact.

## 2026-08-23 — Prompt compression: narratives → history pointers (waste-reduction pilot, Stage A)

- **What:** Compressed `architect.md` 1,738 → 1,547 words per `claude/docs/plans/prompt-waste-reduction.md` §3 (Stage A pilot). Every behavioral rule and mechanism preserved; only class-5/6/7 material (narratives, provenance retellings, duplicate restatements) moved or tightened. Frontmatter, hooks, and all externally-cited anchors untouched: the three verbatim `CPG:` forms, `git add`/`git commit` + "delegated subagent" (audit-team.sh check 8), "never edit your own agent definition" (cited by `docs/plans/generic-cypher-mcp2-coordination.md` P2-B2), and the step name "Investigate the codebase first" (cited by `docs/test-reports/cpg-agent-adoption-report.md:81`).
- **Moved/removed clauses (gate b — each verified already recorded before removal):**
  1. Learning-capture inbox-replacement history ("This replaces the earlier `kaizen/inbox.md`-append convention… fully distilled and removed 2026-08-21") → this file, 2026-08-21 inbox-deletion entry.
  2. Commit-grant provenance retelling ("same as before. Stakeholder decision, 2026-08-21 — see `kaizen/history.md`") → compressed to the dated pointer; full story in this file's 2026-08-21 commit-grant entry.
  3. CPG-freshness decision retelling ("that responsibility is centralized… when a `teco`-issued brief states…") → rule + `(2026-08-19)` pointer kept; full story in this file's 2026-08-19 freshness-centralization entry.
  4. Class-7 dedups: isolated-context restatement in "Handoff" (canonical statement stays in the intro); `PRODUCED`/`:Agent`-edge description in the learning-capture intro (the Cypher template itself shows it); "(use `Write` to create it, `Edit` to amend it in place)"; "not paraphrased, not dropped" (covered by "written verbatim"); minor phrasing tightenings throughout.
- **Rule inventory (gate a):** 32 class-1/2 clauses inventoried pre-edit; all 32 mapped to surviving locations post-edit — zero rule loss. Notables checked one-by-one: the six-section plan skeleton, the CPG-line three-form contract + `not applicable` disambiguation, ambiguity→open-questions-as-deliverable, tico-requirements-first, Explore delegation, data-scientist ML delegation, plan-doc default + path convention + header block, hook-gate-by-pattern guardrail, compress-by-pointer guardrail (both halves), interactive-commit grant + never-list + delegated-subagent carve-out, learning-capture template + skip-rules.
- **Why:** Stakeholder-directed waste reduction — prompts carry rules and one-clause whys; stories live here. Gates: audit-team.sh PASS; cobb §7 lint (gate c) returned **pass with notes, zero rule loss** — its one minor (a telegraphically-clipped CPG-freshness clause that also widened "teco-issued brief" to "a brief") was applied same-session via cobb's suggested rewrite, plus its pre-existing-nit fix (the bare `(2026-08-19)` pointer gained the `kaizen/history.md` anchor). **Same-day calibration ruling (stakeholder, reviewing this pilot):** provenance never earns prompt space at all — no dates, no "stakeholder decision" markers, no `kaizen/history.md` pointers; a rule's non-negotiability is expressed by stating it absolutely, and this file is the standing greppable home for where each rule came from. Both provenance parentheticals (commit-grant, CPG-freshness) accordingly deleted outright; doctrine updated (`prompt-waste-reduction.md` v3, §3 classes 5/6). Normative citations that a rule *uses* (the `CPG:` forms' spec path, the root-`AGENTS.md` header block) stay. Final: 1,545 words. Precedent: this file's own 2026-08-19 de-dup entries — same method, now doctrine.

## 2026-08-21 — Interactive-mode commit grant added (team-wide stakeholder decision)
- **What:** The Bash guardrail's "investigation only" bullet now also grants: when running
  interactively (`claude --agent architect`, a human present turn-by-turn — not a delegated
  subagent), may `git add`/`git commit` its own plan/design document from the session, by
  explicit path, never bulk-staged/pushed/reset/rebased/amended; the grant does not apply when
  spawned as a delegated subagent.
- **Why:** Direct stakeholder ruling, 2026-08-21, after `tico` hit exactly this gap closing out a
  Mode-3 verification pass (its own commissioned artifacts left uncommitted, since only
  `tico`/`teco` had any commit authority). Rather than pin the fix to those two, the stakeholder
  ruled the exception should reach every agent, gated by invocation mode, not identity — full
  rationale, the `claude/AGENTS.md` rewrite, and the `audit-team.sh` check-8 redesign in
  `claude/cobb/kaizen/history.md`, 2026-08-21 entry.
- **Verified:** `bash claude/scripts/audit-team.sh` — clean, all 13 agents pass check 8.
- **Plan items:** none opened — direct implementation of an explicit stakeholder decision.

## 2026-08-21 — `CPG:` line gained a `not applicable` vs. `considered, not relevant` disambiguation (C-408)

- **What:** `cobb` added one clause to this agent's `CPG:` evidence-trail sentence (§ "Context & findings"): `not applicable` is now explicitly scoped to a task with no code-level component at all, distinct from `considered, not relevant` (a code-level task in a component that simply has no loaded CPG). See `claude/cobb/kaizen/history.md`'s matching 2026-08-21 entry for the full reasoning and the defect this closes (`docs/BACKLOG.md` C-408, DEF-4).
- **Why / Verified / Plan items:** see the master entry above.

## 2026-08-21 — `kaizen/inbox.md` deleted (content already fully captured elsewhere)

- **What:** `cobb` deleted this agent's frozen `kaizen/inbox.md` (git history retains it in full, unaltered). It had been frozen — never written to — since the 2026-08-20 graph migration (see that date's entry below, which already confirms this file's own pre-migration content was imported into the graph verbatim at the time).
- **Why:** user-directed team-wide cleanup, "no point keeping a file already in git history." Before deleting any of the 12 agents' frozen inboxes, `cobb` live-confirmed `kaizen_team` — the single shared graph every agent's raw capture has routed through since the 2026-08-20 consolidation — holds **zero** entries for any agent: every raw capture any agent ever wrote there (including this agent's own distillation, immediately above) has since been fully distilled and cleared. Combined with the migration-time import guarantee above, nothing in this file was ever a live, undistilled input to anything — it was a pure redundant backup copy. Same session also completed `G1`'s last 2 of 12 `kaizen_<agent>` graph-key retirements (`kaizen_analyst`/`kaizen_teco`, executed by `graph-dba`), closing `docs/plans/generic-cypher-mcp2-coordination.md`'s one remaining open item.
- **Verified:** live `mcp__cypher__query` count against `kaizen_team` (0 entries) before any deletion.
- **Plan items:** none opened — pure cleanup, no behavior change.

## 2026-08-21 — `kaizen_team` distillation: 4 entries — 1 promoted to Guardrails, 3 discarded (2 already-documented elsewhere, 1 already-actioned)

- **What:** `cobb` processed all 4 `author:'architect'` entries in the shared `kaizen_team` graph
  (agent-maintenance skill §5), all from the same 2026-08-20 episode (authoring/revising
  `docs/plans/generic-cypher-mcp2.md` through several plan-gate rounds).
  - **Promoted (1) → new Guardrails bullet:** `c4a8d2f1` — two plan-revision-rigor lessons from
    that episode's Pass-3 plan-gate majors: compressing a plan section to "see prior version,
    unchanged" breaks any other section that cites the compressed content literally (an
    isolated-context implementer needs the actual Cypher block/command, not a git-history
    pointer); and a blanket find-and-replace across N near-identical files/sections needs the N
    verified textually identical first (a tense mismatch among them means one substitution is
    wrong for some).
  - **Discarded (3):**
    - `a1e3c9d4` — already fully documented in `cypher-mcp/README.md` (the "Read-only" and "Graph
      discovery" sections: a graph materializes only on write, and querying an unknown graph name
      returns the full list of loaded graphs in the error text).
    - `b7f2e1a8` — same episode, same underlying fact (a self-edit "precedent" the plan claimed
      for `architect`/`graph-dba` didn't actually exist, verified against `git show --stat`) as
      `analyst`'s own kaizen capture of this same 2026-08-20 review round — already promoted, with
      the identical origin note, into `claude/analyst/review-techniques.md` ("Ground truth for
      'may an agent edit its own definition?'"). Promoting it again here would duplicate, not add.
    - `f3a7c2e1` — the action it describes (add a dated Version-5 revision note to
      `docs/plans/generic-cypher-mcp2.md` recording the header-retarget instruction's real-world
      enforcement override) is **already done**: the document's header already reads `Version: 5`
      and carries a "Revision note — 2026-08-20 (Version 5)" section. Nothing left to act on.
  - **Docs touched:** `claude/architect/{architect.md,kaizen/history.md}`.
- **Why:** User-requested distillation pass, continuing the oldest-first queue (data-scientist,
  then architect).
- **Plan items:** none opened — the one promotable entry landed directly; the other three needed
  no forward-looking action.

## 2026-08-20 — Learnings capture migrated to a working-memory graph (`kaizen_architect`), mirroring `graph-dba`
- **What:** The "Learning capture" closing-protocol section now writes a `:KaizenEntry` node
  directly into `kaizen_architect` (FalkorDB, via `mcp__cypher__query`) instead of appending to
  `kaizen/inbox.md`. `kaizen/inbox.md` is now a frozen historical snapshot — it had no
  pre-existing entries to migrate; its own header explains the freeze and gives the live-read
  query. The trailing "Your write guard allows exactly this inbox path" clause was dropped — the
  write guard gates `Write`/`Edit`, not the `mcp__cypher__query` MCP tool, so it no longer applies
  to this capture path.
- **Why:** User-directed team-wide redesign ("I will migrate all agents to write their learnings
  to the graph like graph-dba"), reversing yesterday's file-based Learning-capture dedup (entry
  below) — the user determined the whole team should follow `graph-dba`'s existing graph-based
  capture pattern instead of the file-based inbox convention.
- **Plan items:** —

## 2026-08-19 — Learning-capture paragraph de-duplicated against the inbox's own header
- **What:** Trimmed the "Learning capture" paragraph: dropped "(fact, evidence, suggested home; format in the file header)" and "The inbox is raw capture — the team maintainer verifies and promotes entries into prompts, knowledge bases, or project docs" — both already stated verbatim in `kaizen/inbox.md`'s own header template (agent-maintenance skill §5), which the agent necessarily opens to append. Kept: the discipline-specific fact-kind clause, the inbox path, "skip task-specific details," "never edit your own agent definition," and the write-guard clause. Behavior unchanged.
- **Why:** User-directed prompt-verbosity reduction, item 1 of the parked diagnosis (`cobb/kaizen/plan.md`) — the mechanics were literally duplicated (prompt + inbox header say the same thing), not just similar boilerplate; pointing at the file's own header removes the duplication without losing information, since the agent reads that file to act anyway.
- **Plan items:** —

## 2026-08-19 — Freshness-check clause removed (centralized on teco)
- **What:** Dropped the CPG freshness-check paragraph from the CPG-orientation step — still checks whether a relevant CPG exists and uses it via `cpg-analysis`, but no longer queries the `:CpgBuildInfo` freshness marker itself. That responsibility is now `teco`'s alone (`docs/plans/cpg-agent-adoption2.md`, extending the archived `cpg-agent-adoption.md`); running standalone (no `teco`-issued brief), staleness is simply not checked.
- **Why:** User-directed prompt-verbosity reduction: the freshness paragraph was ~130 words, byte-identical across six agent files. Stakeholder chose full centralization over a per-agent dedup, accepting the standalone-run capability loss.
- **Plan items:** —

## 2026-08-16 — Inbox distillation (on-demand, cobb): 2 of 3 entries promoted to `cpg-analysis` skill, 1 already fixed
- **What:** `cobb` processed the three same-day inbox entries at the stakeholder's explicit
  request (not a periodic sweep). **Verified each against live state first:**
  1. `mcp__cypher__query(graph="cpg_falkorchat", "MATCH (n) RETURN count(n)")` → 166,789 nodes (graph
     exists); `graph="cpg_falkor-chat"` → not found, confirming the entry's evidence. **But**
     `skills/cpg-analysis/SKILL.md` §1 already reads "the component-directory name with hyphens
     stripped (`falkor-chat` → `cpg_falkorchat`, …)" — that exact fix landed *earlier the same day*
     in commit `50f9aaa` (U4b-1..5, a different, already-merged unit of the `cpg-agent-adoption`
     milestone), before this distillation pass ran. **Disposition: discard, no skill edit** — the
     architect's dispatch hit the pre-fix skill text mid-flight; the fact it flagged is now fully
     covered by wording that shipped independently the same day, verified by re-reading the file.
  2. Live-reran the batch-`IN` query shape (`MATCH (m:METHOD) WHERE m.NAME IN [...]`) against
     `cpg_falkorchat` — reproduces as described. Not previously documented as a general technique
     (`IN [...]` appeared only inside `code-review.md`'s fixed sink list, never as a stated
     lookup-efficiency habit). **Promoted** to `skills/cpg-analysis/SKILL.md` §3, as a new idiom
     bullet after "Anchor a target method."
  3. Live-reran both the `FILENAME`-scoped and unscoped variants of the caller query against
     `cpg_falkorchat` — both execute as described (shapes match; exact row counts not
     re-verified against the moving `executor.py`/`test_executor_agent.py` source, consistent
     with this skill's own "Verified figures are dated evidence, not targets" policy). Not
     previously named as a technique (`impact-analysis.md` Q1 only documented filtering tests
     *out* via `STARTS WITH 'tests/'`, not the two-pass scoped-then-unscoped pattern for telling
     the design answer apart from the test-surface answer). **Promoted** to
     `skills/cpg-analysis/references/impact-analysis.md`, as a new callout + verified example
     after Q1's existing paragraph.
- **Why:** Stakeholder-requested, scoped to exactly these three entries (not a full inbox sweep).
  Standing distillation duty (agent-maintenance skill §5).
- **Docs touched:** `skills/cpg-analysis/SKILL.md`, `skills/cpg-analysis/references/impact-analysis.md`,
  `claude/architect/kaizen/inbox.md` (cleared to the standard empty placeholder — no other entries
  were pending).
- **Plan items:** —

## 2026-08-16 — U7 fix round: freshness-check sequencing hardened, `CPG:` line anchored (DEF-1/DEF-2/DEF-3)
- **What:** Two wording tightenings per `docs/plans/cpg-agent-adoption-coordination.md` unit U7,
  following U6's `qa-engineer` live-dispatch acceptance pass
  (`docs/test-reports/cpg-agent-adoption-report.md`). (1) The freshness-check sentence now reads
  "query the freshness check … in that same tool call/step, before deciding whether the result
  needs further cross-verification — this is not a separate, optional judgment call" (previously
  "also run the freshness check … as part of that same step") — closes DEF-2 (`architect`, this
  agent, used the CPG and correctly emitted the `CPG:` line, but explicitly declined to run the
  freshness check, reasoning that a grep/CPG agreement substitute was sufficient — it isn't,
  since a stale-but-coincidentally-still-correct CPG wouldn't produce a mismatch either). (2) The
  `CPG:` line instruction now reads "written verbatim and required in all three cases including
  when the CPG isn't relevant — not paraphrased, not dropped" — closes DEF-1 (`coder`) and DEF-3
  (`tdd-engineer`), both format/omission failures on this same wiring pattern. Applied identically
  (phrasing pattern, not restructuring) across all six wired agents; only
  `coder`/`architect`/`tdd-engineer` were live-tested, but the near-verbatim wiring pattern means
  the same gap plausibly existed in `analyst`/`qa-engineer`/`frontend-engineer` too.
- **Why:** U6's acceptance pass found the M4 wiring (U4b/U4b-2) was correctly worded but didn't
  survive contact with a real dispatched agent's own judgment calls — all three live-tested
  dispatches failed a different way (format, skip, silence). Design intent
  (`docs/plans/cpg-agent-adoption.md` §2.3, §3) unchanged: still agent judgment on staleness
  threshold, still no self-triggered rebuild, still a suggestion not a hard rule about *when*
  something counts as stale — only the sequencing and the anchoring got tightened.
- **Plan items:** none new; closes U7.
- **Same-day addendum (U8 diff-gate follow-up):** `analyst`'s U8 diff gate
  (`docs/reviews/cpg-agent-adoption.md`, Pass 3 — approve with suggestions, zero blockers)
  flagged two minors and a nit against this same freshness sentence: (a) `frontend-engineer.md`
  was missing the "tool call/" qualifier the other five files carried, undercutting the U7
  ledger row's and commit message's "identically" claim; (b) the trailing "this is not a
  separate, optional judgment call" had an ambiguous pronoun referent — a literal reading could
  bind "this" to the cross-verification *decision* rather than the freshness *query itself*,
  exactly the room this agent's own DEF-2 dispatch used to reason past a softer version of this
  sentence; (c) nit — "query the freshness check" mismatched a reference-doc noun with a query
  verb, when the actual queried object is the `:CpgBuildInfo` marker (the report's own
  recommendation said "marker"). Fixed all three: the sentence now reads "…query the freshness
  marker (per `skills/cpg-analysis/references/freshness.md`) in that same tool call/step, before
  you decide whether the CPG's answer needs further cross-verification — running the freshness
  check itself is not optional, and skipping it in favor of a substitute check (e.g. grep
  agreement) doesn't satisfy this." — byte-identical across all six files now. The `CPG:`-line
  wording from the original U7 pass was untouched (U8 raised no finding against it).

## 2026-08-16 — M4 cpg-agent-adoption: discovery wording defaulted, freshness-check bundled, evidence-trail line added
- **What:** Three edits per `docs/plans/cpg-agent-adoption.md` §2.4/§3 (U4b). (1) Frontmatter
  `description` reworded from "With a loaded Joern CPG, uses the `cpg-analysis` skill for
  call-graph impact analysis" to "Checks whether a relevant CPG exists as part of its normal
  orientation and, when one does, uses the `cpg-analysis` skill for call-graph impact analysis" —
  conditional → default-orientation framing. (2) "How you work" step 2 ("Investigate the codebase
  first") gained a sentence: check whether a relevant CPG exists (first guess `cpg_<component>`,
  per `skills/cpg-analysis/SKILL.md` §1), and when one is found and used, also run the freshness
  check (`skills/cpg-analysis/references/freshness.md`) as part of the same step, noting what it
  says in Context & findings and surfacing a refresh suggestion — not a silent rebuild — if it
  looks stale. (3) The plan skeleton's item 2 ("Context & findings") gained the one-line `CPG:`
  evidence-trail convention (`CPG: used <graph> — <clause>` / `CPG: considered, not relevant —
  <clause>` / `CPG: not applicable — <clause>`).
- **Why:** M4 (`cpg-agent-adoption`) widens CPG discovery from a conditional check to a default
  orientation step across the three already-wired consumers (`analyst`, `architect`,
  `qa-engineer`), bundles the freshness recipe into that same step (FR-6's surfacing half), and
  adds a spot-checkable `CPG:` evidence trail (AC-2). Per `docs/plans/cpg-agent-adoption.md`
  §2.1-2.3, §3, §6 step 2.
- **Plan items:** none.

## 2026-08-11 — Inbox distillation: 1 entry, merged with analyst's duplicate finding — no prompt change

- **What:** `cobb` processed the single entry in `architect/kaizen/inbox.md` (§5) — the LM Studio
  `/v1` missing-prefix / 200-envelope quirk, independently captured by both `architect` and
  `analyst` during the same K-042 work.
- **Disposition:** the falkor-chat-specific half was already fully documented in
  `falkor-chat/docs/DESIGN.md` §14.8 before this pass; the general, reusable half (the 200+envelope
  shape, and the `urlparse` schemeless-base-URL trap) was promoted once, from `analyst`'s copy of
  the same finding, into `skills/python-web-quirks/SKILL.md` — no separate action needed here.
- **Verified:** `bash claude/scripts/audit-team.sh` clean.
- **Docs touched:** `claude/architect/kaizen/{history,inbox}.md`.

## 2026-08-09 — Held entries 15/16 promoted: consolidated Kiro-facts edit landed
`cobb` closed out the two held-for-consolidated-follow-up entries (exact-CWD-only local-agent
discovery, no upward walk; and the `mcpServers` remote-entry schema using `url` with no `type`
field): the race window against `analyst`'s held entry 28 (same target file) is over, so both
facts were re-verified (now against `kiro-cli 2.16.2`, up from `2.14.1` at original
live-verification — both held) and written into `skills/agent-standards/kiro.md` — discovery
fact into the CLI custom-agents `Location` bullet, `mcpServers` schema fact into the
`mcpServers` config-key bullet (which also corrected the doc's prior "each needs `command`"
phrasing to cover the `url`-keyed remote case). `inbox.md` entries 15 and 16 cleared; `inbox.md`
is back to the standard empty placeholder.

## 2026-08-09 — Description gained a `python-web-quirks` skill routing clause
- **What:** Frontmatter `description` gained one clause, appended to the existing `cpg-analysis`
  routing sentence: in a Python web/async codebase, the agent also uses the new
  `skills/python-web-quirks/SKILL.md` for asyncio/FastAPI/Starlette/pydantic gotchas. No body
  change.
- **Why:** `python-web-quirks` was created distilling three general Python/web-framework facts
  from `analyst`'s learnings inbox. Stakeholder wired it to `coder`/`tdd-engineer`/`architect`/
  `analyst` at minimum, mirroring the existing `cpg-analysis` wiring pattern. See
  `claude/analyst/kaizen/history.md` (2026-08-09) for the full distillation record and
  `skills/README.md` for the catalog entry.
- **Plan items:** none.

## 2026-08-09 — Inbox entry #1 routed to a separate architect-persona task, not edited here
- **What:** Entry #1 (2026-07-19, `_drive_loop` byte-identity-lock SHA-reproduction command) is
  disposed as: **promoted to `falkor-chat/docs/DESIGN.md`** (the doc-drift note near §6.2, where
  the lock's SHA is already quoted with corrected byte-count guidance but no robust reproduction
  command) — via a dedicated architect-persona task, teco-coordinated, running in parallel, **not
  edited by cobb**. Content and destination were fully decided in the 2026-08-09 read-only
  proposal pass; only execution was delegated. Cleared from `kaizen/inbox.md`.
- **Why:** Role-responsibility principle — this is `falkor-chat/` project-doc content within a
  component `architect` owns the plans/design for; a team-maintainer session (cobb) editing it
  directly would cross a boundary that belongs to the producing discipline, not the team
  maintainer. Distillation duty (agent-maintenance skill §5), teco-coordinated.
- **Plan items:** —

## 2026-08-09 — Guardrails: verify what a hook pattern-matches before treating its prompt as a gate
- **What:** Added one Guardrails bullet: *"Verify hook gates by pattern, not intent. When a plan
  step's verification depends on a `PreToolUse` hook firing, check what the hook actually
  pattern-matches (the command text, not the intent) before treating the prompt as a gate — a
  destructive operation wrapped inside a script can bypass a hook that only greps literal command
  strings."* No other prompt/frontmatter/catalog change.
- **Why:** Distilled from inbox entry #9 (2026-07-25, `guard-destructive-ops.sh` matches the Bash
  *command string*, so `skills/joern-cpg/scripts/pipeline.sh --reset` bypassed the destructive-ops
  approval prompt entirely — the token the guard greps for never appears in the command text the
  hook sees). Stakeholder judged this **high-recurrence** (hook-gated approval prompts are this
  team's central safety mechanism — five doc-scoped write guards plus the destructive-ops guards
  across `devops`/`graph-dba`/`qa-engineer`) and **already-proven-costly** (the bypass this entry
  describes actually happened and shipped; the fix landed as C-311 on 2026-08-08, roughly two
  weeks after the gap was knowable). Cleared from inbox alongside the rest of the distillation pass
  (see the batched entry below) — its *specific* finding (the `pipeline.sh --reset` bypass) is
  independently discarded there as already fixed; this entry is the promotion of its *general*
  habit.
- **Plan items:** —

## 2026-08-09 — Inbox distillation: 13 entries discarded (already fully covered elsewhere)
- **What:** Processed 13 inbox entries against current repo state (agent-maintenance skill §5):
  2026-07-24 `MERGE … ON CREATE SET` create-only-for-properties/additive-for-structure;
  2026-07-24 `OPTIONAL MATCH` + `collect(DISTINCT …)` non-aggregated-field-is-a-grouping-key;
  2026-07-24 falkor-chat def-publish-has-no-graph-seam; 2026-07-24 FalkorDB silently ignores
  `EXPLAIN`/`PROFILE` inside `GRAPH.QUERY`; 2026-07-24 `tools:` allowlist makes new MCP tools
  invisible to `analyst`/`architect`; 2026-07-25 check-7-forbids-absolute-home-paths +
  `.mcp.json`'s portable `$CLAUDE_PROJECT_DIR` form; 2026-07-25 `teco`'s write guard can't own
  edits outside `docs/plans/`; 2026-07-25 `guard-destructive-ops.sh` command-string bypass
  (specific finding only — general habit promoted separately above); 2026-07-25 MCP output over
  threshold persists to disk, truncation notices belong at the head; 2026-07-25 an MCP server's
  `instructions=` is injected every session and is probe-verifiable; 2026-07-26 leading-slash
  markdown links aren't agent-followable; 2026-07-26 this repo cites paths as backticked strings,
  not markdown links, so a link-checker is nearly blind here; 2026-07-27 root `AGENTS.md` reaches
  subagents too via `CLAUDE.md`'s `@AGENTS.md` import.
  **Verified each still-true-but-superseded or already-promoted:** the MERGE-additive defect was
  closed outright by **K-034** (`falkor-chat/docs/HISTORY.md:60`, 2026-08-01) — re-publish now
  fails loudly (409) instead of silently minting parallel structure, so the trap the entry warned
  about no longer exists. The grouping-key hazard and the all-rows-read decision it drove are
  documented verbatim in `falkor-chat/docs/QUERIES.md:925-941` (the K-031 §11.2 callout). The
  def-publish-no-seam finding and its throwaway-`ws:`-probe workaround are recorded as executed in
  `falkor-chat/docs/HISTORY.md:903-919` (K-031 V-1). The `EXPLAIN`/`PROFILE` behavior is documented
  verbatim in `skills/cpg-analysis/SKILL.md:65-75`. The `tools:` allowlist gotcha is documented in
  `skills/agent-standards/claude-code.md:373-375`, and the concrete instance was closed the day
  after capture (this file, 2026-07-25 entry: `mcp__cypher__query` added to `architect`'s and
  `analyst`'s `tools:`). The check-7/`.mcp.json` portable-form fact is documented in
  `skills/agent-standards/claude-code.md:259-276` and is what the repo's actual `.mcp.json` uses
  today. The `pipeline.sh --reset` bypass is fixed (C-311, `claude/scripts/guard-destructive-ops.sh`,
  2026-08-08). The MCP output-limit/disk-persistence and server-`instructions=`-injection facts are
  both documented near-verbatim in `skills/agent-standards/claude-code.md:307-347`. The
  `teco`-write-guard-scope fact is stated directly in `claude/teco/teco.md:67`. The leading-slash
  and backticked-path-citation facts are exactly what root `AGENTS.md`'s citation convention now
  states — those two entries **became** the shipped convention. The subagent-reachability fact is
  documented in `skills/agent-standards/claude-code.md:141-142,175` and is what motivated this
  agent's own 2026-07-27 "header block from root `AGENTS.md`" prompt line (this file). In every
  case the promotion had already happened, dated the same day or within days of capture, as a
  byproduct of shipping the plan that surfaced the fact — only the inbox-clear step was
  outstanding. Entry #4's secondary "verify a live probe has a graph/tenancy seam before
  scheduling it" habit is **parked, not promoted** — recorded in `kaizen/plan.md`'s parking lot
  (judged narrow/single-occurrence; revisit on recurrence). All 13 cleared from
  `kaizen/inbox.md`; entries #1, #15, #16 handled separately (see the adjacent 2026-08-09 entries
  and `kaizen/inbox.md`, which still carries #15/#16 pending a consolidated `skills/agent-standards/kiro.md`
  follow-up).
- **Why:** Standing distillation duty (agent-maintenance skill §5), teco-coordinated pass over the
  full inbox (267 lines, 16 entries — the largest in the team).
- **Plan items:** —

## 2026-07-28 — Inbox entry distilled: `audit-team.sh` is not a usable bare pass/fail done-condition
- **What:** Processed the 2026-07-25 inbox entry *"`audit-team.sh` passes is an unusable plan
  done-condition"* (agent-maintenance skill §5). **Verified still true:** re-ran
  `claude/scripts/audit-team.sh` — check 7 still greps every tracked file in the repo for
  personal identifiers, so a plan step worded as a bare "assert it passes" is unsatisfiable
  the moment any unrelated leak exists anywhere in the repo. **Routed to project docs**, not
  this agent's always-loaded prompt (the fact only bites when a plan step references this one
  script — too narrow to pay for on every session): added a callout to
  `skills/agent-maintenance/SKILL.md` §4, right after the deterministic-half paragraph the
  entry was actually about, stating the fix (assert "no new FAIL line" against a captured
  before-state, not a bare pass). The entry's own personal-path fragment (`claude/joern/…`,
  `docs/plans/m2-cpg-analysis-skill.md:327`) was stale bookkeeping only — the `joern` agent it
  cited was retired 2026-07-28 (see `claude/graph-dba/kaizen/history.md`) and the four leak
  sources it listed were fixed the same day (this task) — so nothing else needed carrying
  forward. Entry removed from `claude/architect/kaizen/inbox.md`.
- **Why:** Standing distillation duty (agent-maintenance skill §5) — cobb processed this entry
  on the coordinator's explicit request, tied to the same-day joern-agent cleanup that had
  left `claude/joern/kaizen/inbox.md:19` (cited by this entry) a dangling path.
- **Plan items:** —

## 2026-07-27 — Unpinned from `model: opus` (team-wide)
- **What:** Removed the `model: opus` frontmatter line. The field is now absent, so the agent runs on Claude Code's default — `model` **defaults to `inherit`** (re-verified 2026-07-27 against `code.claude.com/docs/en/sub-agents`), i.e. the model the session/system default selects. No other frontmatter or body change.
- **Why:** User no longer wants the team locked to Opus. Model choice belongs at the session level (one decision, changeable with `/model`), not duplicated across 13 frontmatter files where it silently overrides whatever the user picked.
- **Plan items:** —

## 2026-07-27 — Plans open with the canonical header block (step 2 of `docs/plans/doc-reference-convention.md`)
- **What:** One line added to *How you work* step 5, immediately after the plan-document convention: *"Open the document with the header block from root `AGENTS.md`."* No frontmatter, hook, `description` or catalog change.
- **Why:** `docs/plans/doc-reference-convention.md` v1.4 §9.6 makes a three-field header (`Status:` · `Owner:` · `Tracks:`) the lifecycle signal that replaces the milestone filename prefix and the move-to-`archive/` rule; a plan is the most-cited document kind in the repo, and `architect` is the agent that creates it, so the field is only ever present if this prompt asks for it. The line is a **pointer, not an inlined template** (v1.4 M20): root `AGENTS.md` reaches every agent through the root `CLAUDE.md` `@AGENTS.md` import, so the second hop costs nothing, while eight copies of a still-settling block would drift — §9.6 stays the one place the block is stated. The sentence is byte-identical across all six producing prompts on purpose; the convention's coverage check greps for it literally. `claude/README.md` row 9 re-checked — it cites the plan write path and the hook, not the document's internal structure; no edit needed.
- **Plan items:** none. Note for whoever picks up the parking-lot "self-review checklist before delivering a plan" idea: the header block is now the first thing that checklist should assert.

## 2026-07-25 — `tools:` allowlist gains `mcp__cypher__query` (M3 / C-304)
- **What:** Frontmatter `tools:` now ends `…, Agent, mcp__cypher__query`. `claude/README.md` row 9 updated to say the `cpg-analysis` skill reaches the graph through that MCP tool and why the allowlist entry is required. No body or `description` change — the impact-analysis CPG clause added on 2026-07-19 stays accurate.
- **Why:** M3 replaces the CPG read path with a single MCP tool, `mcp__cypher__query(graph, cypher)` (`docs/plans/cpg-query-access.md` S5). **`tools:` is an allowlist, not a hint** — an agent that declares one does not see MCP tools absent from it, so without this line the tool would have been invisible to `architect` (and `analyst`) while `qa-engineer`, which declares no allowlist, inherited it. `redis-cli GRAPH.QUERY` remains the documented fallback and is the only path under OpenCode/Kiro.
- **Verification note:** this is the *edit*; the live proof (a cold `architect` actually calling the tool) needs the server wired in S3 and is verified in S9, per the plan's m-4 split.
- **Plan items:** none.

## 2026-07-24 — Description slimmed further (second team-wide token-cost pass)
- **What:** Frontmatter `description` compressed 627 → 453 chars (-27%): tightened phrasing, dropped restated detail, kept every routing/boundary clause. `claude/scripts/audit-team.sh` boundary-pair symmetry (architect↔data-scientist) re-verified green. No body/catalog change.
- **Why:** All 13 agents' descriptions are auto-injected into every session and subagent spawn; the roster grew to 13 (graph-dba, joern added) since the first pass on 2026-07-11, and per-agent `/context` output showed room to cut further. User-requested via a `/context` token audit.
- **Plan items:** none.

## 2026-07-24 — Frontmatter: `permissionMode: acceptEdits`
- **What:** Added `permissionMode: acceptEdits` to the frontmatter, matching the same-day change to `coder`/`tdd-engineer`/`frontend-engineer`. File-edit/write approvals are session-scoped in Claude Code (unlike Bash approvals, which persist permanently per repo+command), so users otherwise have to re-grant write permission every session even with a global `Edit`/`Write` allow rule in `~/.claude/settings.json`.
- **Why:** Verified against current Claude Code docs (`hooks-guide.md` "Hooks and permission modes") that this is safe here specifically: `PreToolUse` hooks fire *before* any permission-mode check, and a hook's `"ask"` decision still forces the prompt even under `acceptEdits`/`bypassPermissions` — modes can't loosen what a hook tightens. `architect`'s `guard-plan-doc-writes.sh` hook (escalates to ask on any Write/Edit outside the allowed plan-doc paths) keeps working exactly as before; only writes it would already let through silently — writes inside the allowed doc paths — stop re-prompting every session.
- **Plan items:** none.

## 2026-07-19 — CPG capability wired into the routing description (M2 / C-207)
- **What:** Frontmatter `description` gained one clause: for call-graph impact analysis over code with a loaded Joern CPG in FalkorDB, the architect uses the `cpg-analysis` skill (graph-dba-owned). `claude/README.md` catalog entry updated to match. No body change (skill is progressively disclosed).
- **Why:** M2 delivered the `cpg-analysis` skill; `architect` is a named consumer of the impact-analysis recipe (FR-10). C-207 makes the routing contract advertise it. Wired by cobb as part of Gate-2b (skill passed the standards vet).
- **Plan items:** none.

## 2026-07-12 — Learning-capture loop: kaizen inbox + closing protocol + guard allowlist
- **What:** Added `kaizen/inbox.md` (append-only learnings inbox, seeded empty) and a "Learning capture" closing-protocol section to the prompt; the doc-scoped write guard's allowlist gained exactly the agent's own inbox path (`<name>/kaizen/inbox.md`), with the escalation message updated to match.
- **Why:** Team-wide self-improvement loop (agent-maintenance skill §5, added the same day): capture is cheap and unreviewed during runs, promotion is curated — cobb periodically verifies each entry and routes it to the prompt, an on-demand knowledge base, or project docs. Requested by the user.
- **Plan items:** none.

## 2026-07-11 — Description slimmed (team-wide token-cost pass)
- **What:** Frontmatter `description` compressed from 761 to 498 chars: capability lists tightened, reciprocal boundary prose reduced to short route-away clauses that still name the counterpart agents (audit check 6 boundary symmetry preserved — full pass green), and "how I work" detail dropped from the description since the prompt body already carries it. Routing semantics unchanged; no body/catalog changes needed.
- **Why:** All 12 agents' descriptions are auto-injected into every session and into every subagent spawn that carries the `Agent` tool; team-wide they cost 12,609 chars (~3.1K tokens) per injection. The pass cut them to 7,036 chars (~44%), saving ≈1,400 tokens per session/spawn with the same routing contract.
- **Plan items:** none.

## 2026-07-11 — Guard hook refactored to a thin wrapper over a shared core
- **What:** `guard-plan-doc-writes.sh` was reduced from a ~60-line standalone script to a thin wrapper that `exec`s the new shared core `claude/scripts/guard-doc-writes.sh` with two parameters — this agent's allowed-path globs (`docs/plans/*|*/docs/plans/*`) and its escalation-message template (`__PATH__` placeholder for the offending path). The core carries the shared machinery unchanged: jq→python3 path extraction, fail-open on unparseable input, `/tmp/*` always allowed, `permissionDecision: "ask"` JSON emit. The wrapper resolves the core via `readlink -f "$0"`, so it works when invoked through the `~/.claude/agents/<name>` deployment symlink; the frontmatter hook command is unchanged. Verified: `bash -n`, allowed/denied/scratchpad/fail-open cases through the symlink path, the no-jq python3 fallback, and `claude/scripts/audit-team.sh` all pass.
- **Why:** a repo redundancy audit (2026-07-11) found the five doc-scoped guards (analyst, architect, data-scientist, teco, tico) byte-identical except one `case` glob and one message string — ~250 duplicated lines that had to be patched five times per fix. One parameterized core removes the drift risk. (`devops/hooks/guard-destructive-ops.sh` stays standalone — it matches Bash command patterns, not write paths.)
- **Plan items:** none.

## 2026-07-10 — Hook command made machine-independent (`$HOME` symlink path)
- **What:** the frontmatter `PreToolUse` hook command was rewired from the absolute repo path (`/home/<user>/prg/graphmind-ai-lab/claude/architect/hooks/guard-plan-doc-writes.sh`) to `$HOME/.claude/agents/architect/hooks/guard-plan-doc-writes.sh`, which resolves through the user-scope deployment symlink (`~/.claude/agents/architect` → the repo folder). Shell-form hook commands (no `args`) run via `sh -c`, so `$HOME` expands — verified 2026-07-10 against `code.claude.com/docs/en/hooks`. Resolution through the symlink confirmed (`test -x` passes).
- **Why:** the committed agent source leaked the user's personal home path into the repo; the symlink path is identical on any machine that follows the deployment convention (`~/.claude/agents/<name>` → `claude/<name>`), keeping the hook enforceable without machine-specific paths. (`${CLAUDE_PROJECT_DIR}` was rejected: the agents are user-scoped and must guard in any project, where the project dir isn't this repo.)
- **Plan items:** none.

## 2026-07-09 — data-scientist boundary clause (description + delegate-the-method step)
- **What:** Frontmatter `description` now names the `data-scientist` as the supplier of a design's AI/ML/DS method (model/embedding selection, retrieval strategy, evaluation methodology, experiment design), and "How you work" step 4 (Decide) instructs delegating such method calls to it via the `Agent` tool — it returns a method note at `<component>/docs/plans/<slug>-ml.md` (or inline) that the plan folds in, rather than the architect guessing the method. Pair `architect:data-scientist` added to `claude/scripts/audit-team.sh` `BOUNDARY_PAIRS` (check 6, description symmetry).
- **Why:** The `data-scientist` agent was created 2026-07-09 explicitly to work alongside the architect; the consumer side must state the convention too (agent-maintenance §4 handoff symmetry).
- **Plan items:** none.

## 2026-07-09 — Consume tico's requirements doc by path (handoff symmetry)
- **What:** "Understand the request" now states that a feature requirements document from `tico` may arrive as a path (`<component>/docs/requirements/<slug>.md`) — read it first as the stakeholder-confirmed WHAT/WHY the plan turns into a HOW; its acceptance criteria feed the test strategy.
- **Why:** `tico` was created 2026-07-09 as the requirements half of a tico→architect handoff; the consumer side must state the convention too (agent-maintenance §4 handoff symmetry).
- **Plan items:** none.

## 2026-07-09 — K-002 ✅: live handoff validation (teco K-001 run, falkor-chat M3 slice 1)
- **What:** The architect ran as the planning half of a real orchestrated delivery — teco
  delegated it the M3 decomposition + slice-1 plan for falkor-chat. It produced
  `falkor-chat/docs/plans/m3-workflow-engine.md` (Part A: six kaizen items K-020…K-025 in the
  component's exact item format; Part B: full slice-1 plan with data model, DDL reconciliation,
  query shapes, service surface, build order, enumerated suite-count expectations) and returned
  the path. Two isolated-context implementers executed it cold: graph-dba (gate) and
  tdd-engineer (impl) — no re-investigation loops, suites landed green (query 193/193,
  pytest 196), structural parity + idempotency proven.
- **Friction observed (the K-002 payload):** one plan gap — `publish_workflow_def` was specced
  without a `start_key` parameter; the implementer resolved it (exactly one step declares
  `start: True`) and it was surfaced as a contract to lock at K-022. One gate-level design
  amendment — the plan's `STARTS WITH stepUid` scoping PROFILEd as a label scan, so graph-dba
  added a `HAS_STEP` containment edge; a reasonable division of labor (live PROFILE data is the
  gate's job), not a plan defect. Verdict: the six-section template held; no template change
  needed from a single datapoint — recheck if the parameter-contract gap recurs.
- **Prompt changes:** none.
- **Plan items:** K-002 ✅ done (moved here — the live validation this item waited on; evidence
  shared with teco K-001, see `claude/teco/docs/HISTORY.md` 2026-07-09). Plan is now empty of
  active items.

## 2026-07-08 — Plan-doc handoff by default, subagent-context awareness, hook-enforced Write/Edit (K-001 ✅, K-003 ✅)
- **What:** Four changes from a team-level design review (architect as a member of teco's roster):
  1. **Plan document is now the default deliverable** (K-001 ✅): step 5 rewritten — write the plan to `<component>/docs/plans/<slug>.md` (kebab-case; repo-root `docs/plans/` for cross-component work) and return the *path* + the "ready to implement" summary; inline delivery only for quick assessments. Handoff section updated to match ("implement the plan at `<path>`"). Convention matches what falkor-chat already used de facto (`falkor-chat/docs/archive/plans/m2-graphrag.md`).
  2. **Subagent-context awareness:** new opener paragraph — the brief is the architect's *entire* context (no user conversation, no other agents' work) and its final message is terminal (`AskUserQuestion` unavailable to subagents). Step 1 reframed: design-changing ambiguity → return findings + the one or two sharp open questions *as the deliverable* and stop, instead of "ask questions" (impossible mid-run).
  3. **TDD-destined plans:** test-strategy section now says to sequence as an ordered list of behaviors/test cases (red→green) when the repo mandates TDD or the implementer is `tdd-engineer` (this user's default preference).
  4. **Harness-enforced read-only-on-code** (K-003 ✅): new subagent-scoped `PreToolUse` hook `architect/hooks/guard-plan-doc-writes.sh`, wired in frontmatter (`matcher: Write|Edit`, absolute path — devops precedent). Escalates any `Write`/`Edit` targeting a path outside `docs/plans/` (or `/tmp` scratchpad) to the human (`permissionDecision: "ask"`); fail-open on unparseable input with the prompt guardrail as backstop. Smoke-tested: code path → ask; absolute + relative `docs/plans/` → pass; `/tmp/` → pass; garbage stdin → pass.
  - Sibling + catalog sync (same change): `teco.md` now hands off the architect's plan **by path** (never paraphrased into the brief) and gained a coordination-doc convention (`docs/plans/<slug>-coordination.md`, teco K-003 ✅); `coder.md` orient step expects the plan-document path and reads the file as source of truth (`claude/AGENTS.md`, `claude/README.md`, root `AGENTS.md` updated).
- **Why:** Review found (a) inline-by-default delivery was optimized for a human caller and pessimal for the orchestrated path — teco copying a plan "verbatim" into a brief is a lossy telephone game, while a file handed off by path is lossless and durable; (b) the prompt assumed an interactive caller it doesn't have as a subagent; (c) the read-only contract was prompt-only while a working hook precedent existed in-repo (`devops/hooks/guard-destructive-ops.sh`).
- **Decision — Bash deliberately NOT hooked:** the hook closes the realistic *accidental* failure mode (the editing tools drifting into source). Mutating the tree via Bash would be a deliberate guardrail violation, which prompt-guarding handles reliably for Opus-class models, and pattern-matching "bash writes" is brittle/noisy. Accepted residual risk; escalation path recorded in the plan's parking lot.
- **Plan items:** K-001 ✅ done, K-003 ✅ done (both moved here); K-002 remains open — the convention is baked but the live architect→coder validation run is still pending.

## 2026-06-21 — Added `Edit`, scoped to plan/design docs
- **What:** Added `Edit` to the frontmatter `tools` list (`Read, Grep, Glob, Bash, Write, Edit, WebFetch, WebSearch, Agent`). Reworded the Guardrails so `Write`/`Edit` are explicitly for **one purpose** — authoring (`Write`) and revising in place (`Edit`) the plan/design document — and never source/tests/config. Updated catalog entries that previously asserted "no `Edit`" (`claude/AGENTS.md`, root `AGENTS.md`). The `description` and `claude/README.md` ("does NOT edit source code" / "without editing code") were left unchanged — still accurate, since the agent edits plan docs, not code.
- **Why:** User asked to enable `Edit` for the architect. Flagged the design tension (its whole contract is read-only-on-code; the description is also the auto-delegation routing signal) and confirmed intent: **plan docs only**, not code. Previously the agent could only `Write` (overwrite) plan files; `Edit` lets it amend a plan in place during an architect→coder iteration without rewriting the whole document. Tool gating still can't enforce "plan docs only" (both Write and Edit can target any path) — the prompt guardrail carries that, same as before.
- **Plan items:** advances the spirit of K-003 (tool gating) but in the *loosening* direction; K-003 (stricter gating) left open — see plan note.

## 2026-06-20 — Dropped "senior" framing
- **What:** Removed "senior" from the `description` ("Senior software architect" → "Software architect") and the body opener ("You are a senior software architect" → "You are a software architect"). Mirrored in the catalog entries (`claude/README.md`, `claude/CLAUDE.md`, root `AGENTS.md`).
- **Why:** User flagged the overconfidence concern with seniority framing. Evidence (persona-prompting studies, e.g. Zheng et al. 2024) shows role labels are weak-to-neutral for correctness and authority framing can dent calibration; behavior is driven by the concrete process + guardrails, not the title. Chose the most conservative option (drop the word entirely) over keeping it. Note: this goes one step *further* than the 2026-06-05 collection precedent, which dropped boasts but **kept** "Senior" as an altitude signal — so architect/coder are now inconsistent with cobb/tdd-engineer/graph-dba/dra-claudia until those are harmonized (flagged to user).
- **Plan items:** —

## 2026-06-20 — Created
- **What:** Created the `architect` subagent (`architect/architect.md`, `model: opus`). Read-only design/planning agent: investigates the codebase, weighs trade-offs, and produces a step-by-step implementation plan/spec (goal & scope, context & findings, design & rationale, ordered steps, test strategy, risks & open questions). Tools restricted to `Read, Grep, Glob, Bash, Write, WebFetch, WebSearch, Agent`; **no `Edit`/`NotebookEdit`** and a hard guardrail that `Write` is for the plan document only — it never edits source/tests/config. Designed as the planning half of an **architect→coder handoff**: the plan stands alone so an isolated-context implementer can execute it.
- **Why:** User asked to create two complementary Claude Code subagents, "the architect" and "the coder," distinct from the existing `tdd-engineer` (strict TDD implementer) and the OpenCode `coding-senior`, with a sequential architect→coder handoff.
- **Plan items:** seeded K-001..K-003.

## Decisions recorded at creation
- **Why `Write` but no `Edit`:** the agent must be able to emit a *durable* plan document for the handoff (an isolated coder context won't see the architect's investigation otherwise), but must not surgically modify existing code. Omitting `Edit`/`NotebookEdit` + a prompt guardrail signals "planning only" while still allowing the deliverable. `Bash` is investigation-only by guardrail. Tool gating can't fully enforce "plan docs only" (Write can overwrite any path) — the guardrail carries that.
