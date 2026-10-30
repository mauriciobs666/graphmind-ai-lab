# Review — connected code graph eval protocol, key-normalizer amendment (step 8a)

> **Status:** active · **Owner:** `analyst` · **Tracks:** C-1001 (M10) · **Reviews:** `docs/plans/connected-code-graph-eval-protocol2-ml.md` v1.0

## Pass 1

### Scope & verdict

**Reviewed:** `docs/plans/connected-code-graph-eval-protocol2-ml.md` v1.0 ("the note"). I read it
against:
- `docs/plans/connected-code-graph-eval-protocol-ml.md` v1.2 ("EP"): §2.2, §3.2–§3.5, §4.1–§4.4,
  §5.4, §6.2–§6.3, §8, §9.1–§9.4, §10.2–§10.4, §13.1, §13.4, §13.6, §14 and Appendix B.0/B.2;
- `docs/plans/connected-code-graph-bundle-spec2.md` v1.3 ("BS2"): §1.7.1, §1.8, §1.10, §1.12, the
  §2 SA-9(a) row, the §3 F-8 row and §4.1–§4.2;
- spec §6 (`docs/plans/connected-code-graph-bundle-spec.md` v1.3);
- the p0-2 rows for S06, S12, S14, S30, S34, S47 and S49, plus its §15/§16.1 F-8 lines
  (`docs/plans/connected-code-graph-p0-2.md` v1.1).

**Method:** static reading plus scratch execution in the session scratchpad. I emulated BS2
v1.3's token screen from the spec text alone, and separately emulated v1.2's `normalize` steps
1/4/5/6/7 and step 8a. I ran the oracle under both my screen and the working tree's
`cpg/ccg/bundle.py` `screen_fails`, and also ran adversarial key sets, a 300,000-string
differential fuzz, a segment recount under four extraction definitions, and a leak-check
simulation using EP §9.4's exact v1.2 pattern. I edited no file other than this review, ran no
tree-mutating git command, and touched no container.

**Verdict: needs changes.** There are no blockers. There are two majors:
- **M1:** §1.4's writer-form probe makes the leak check fail legitimate reports.
- **M2:** gold is normalized at two different times, and no guard catches drift between them.

The core claim holds under execution: 8a only coarsens v1.2, and the function is idempotent.

**Findings:** 0 blocker · 2 major · 5 minor · 3 nit.

CPG: considered, not relevant — the artifact is an eval-method document. The code questions (the
`screen_fails` body, route literals) were answered by reading and executing the working-tree
files directly, and a graph of the in-flux `bundle.py` would be no fresher than the file itself.

**Brief item 1, the equality claim: verified** (Appendix A.1–A.3).
- **K1–K23:** every `http` row reproduces under both screens, and every output is a fixed point.
  K23 reproduces.
- **Coarsening:** over 1,682 adversarial keys (38 segment shapes, at one and two segments, plus
  lowercase-verb variants), 0 v1.2 equivalence classes split under v1.3.
- **Idempotence:** 0 keys are non-idempotent under either screen. That includes all four
  placeholders, escaped look-alikes (`%7B%23%7D`, `{%23}`), `*`, `..` and mixed segments.
- **The one inaccuracy:** §1.3's description of *which* pairs merge is incomplete (m2).

### Findings

#### M1 (major): §1.4's writer-form probe makes the leak check fail legitimate reports, and changes EP §9.4 beyond what 8a needs

**Evidence.** In a scratch simulation using EP §9.4's exact v1.2 pattern and its "exactly equal"
removal rule (Appendix A.5), the held-out writer form `get /shop/api/products` is not exactly
equal to dev's `GET /shop/api/products`, so it survives removal. The report is then casefolded,
so a report that legitimately names the dev route **fails**.
- A dev route can share a held-out anchor's `keyNorm` without contact. Two ways: a dev AC-24 gold
  anchor (§10.2, "contact does not apply"), or a dev unit's `serves[]`, since a held-out contract
  anchor under a dev unit is allowed by §3.5's one-way rule.
- The raw probe also catches content that v1.2 never probed for, such as a held-out writer's
  parameter name `{item_id}` (A.5, row 6). That is a semantic change to §9.4, not a clarification.

**Why it matters.** `--check-report` is part of S47's done-condition (p0-2 §3 S47 row; §6.4's
"Leak check" step) and of every later report check. A false positive fails a clean report, and the
fix would then need a protocol revision.

**Fix.** Make the second probe the **pre-8a form**, meaning v1.2's `normalize` output, and remove
it by equality against the pre-8a forms of dev values and C. It still carries the literal, so
both halves of K29 come out as written (A.5, rows 1–2). It also has no writer-spelling variance
(rows 3 and 6 pass).

Also restate K29's dev item as a dev AC-24 gold anchor or a dev unit's `serves[]` entry. A dev
*key item anchored* at `POST /hooks/{#}` has the held-out item's own `anchorId`, so it sits in the
same cluster, and the §9.1 split can never produce that state.

#### M2 (major): gold is normalized at two times, with no guard against screen drift between them

**Evidence.**
- AC-5 anchor ids are computed by `normalize` at S45 and frozen at S47 (EP §9.2).
- AC-24 gold goes through `normalize_contract` when the P1 harness scores it at a gate (§1.2,
  "before EP §10.4 compares id sets"). That run uses whatever `bundle.py` exists at that gate.
- EP has no contract analogue of §5.4. §5.1 asks "from its contract point". What happens when that
  `cpId` is absent from the gate graph is unspecified, and the default reading is that the
  anchor's items are not hits: a **miss with no stated signal**.
- The screen has already moved twice in BS2's own life: T6 was added in v1.1 and widened in v1.2.
  BS2 §1.12 makes a spec/`bundle.py` disagreement a `bundle.py` defect, and a defect fix can only
  land by changing that file.

**Why it matters.** Suppose a screen fix lands after S47. Every masked held-out item then misses
without any signal, and gold and AC-5 are normalized under different screens. Exposure is 0 at
today's tree (see the recount), so this is latent, not live.

**Fix.**
1. S47 stores AC-24 gold **already normalized** (logged as V3 is), so the harness never
   re-normalizes it.
2. `eval-manifest.yaml` records a screen fingerprint: BS2 version plus the SHA-256 of
   `cpg/ccg/bundle.py` at `EVAL_COMMIT`.
3. At each gate the harness re-runs `normalize_contract` over the stored writer forms and **stops
   loudly** if any `anchorId` or gold id differs.

This is the weighing the brief asked for (item 4): **importing the screen is right**. The eval has
to equal the graph, and a reimplementation would grade spec-correct anchors as misses wherever
`bundle.py` deviates. This is the same precedent as `ids.py` (EP §2.2). The real risk is drift
across epochs, and an assertion closes it. It belongs in S49's verbs and the P1 harness, not in
S10.

#### m1 (minor): §3's obligation table misses steps the note itself binds

**Evidence.** The note's own text binds four steps its §3 table does not account for:
- §1.5 changes `eval-manifest.yaml`'s version field. S40 writes that file (EP §13.1 step 8), but
  §3 lists S40 as "**none**".
- §1.2's C conversion and §1.5's manifest string run at **S47** (EP §14: S47 takes §8–§9), but the
  S47 row carries only §1.5.
- §1.1's V3 log is reviewed at **S45**, which is not listed.
- S49's p0-2 dependencies (S48 only) do not include S12, whose function K1–K22 must also run
  against (§1.1, R1), or S10's final screen.

**Fix.** Rewrite §3 from the list in the "Obligations" section below.

#### m2 (minor): §1.3's "exactly one class" of new merges is incomplete

**Evidence.** Every merge outside the stated class (6,300 pairs over the adversarial set) pairs a
failing literal with a writer-written `{#}`: `POST /hooks/x7k2q9w4m1p8z3v6` and `POST /hooks/{#}`
(K2 and K20) share a `cpId`. With `{#}` admitted as a member, 0 pairs remain outside (A.3).

**Fix.** State the class as follows: same verb, same segment count, and every differing position
is either a screen-failing literal or `{#}`. Add the K2 ≡ K20 identity as an oracle case.
Mechanically this is harmless, because the graph makes the same merge.

#### m3 (minor): §1.2 normalizes AC-24 gold and C with all of `normalize`, not only 8a, and does not say so

**Evidence.**
- Under v1.2, AC-24 gold and C entries were converted by §2.2 **without** `normalize`.
  `normalize` ran only on key items (EP §13.4, §8.2 "converted to `anchorId` by §2.2").
- §1.2 now also applies steps 1/4–7 and the `mcp__<server>__` strip to both.
- That makes AC-24 scoring more lenient for gold-writer slips, and can add contacts.

This is **monotone**: by idempotence (A.2), any pair equal under v1.2 stays equal. So contacts and
understood items can only be added. That is conservative for the split. §0's "Does it move AC-5?
No" is still true, but the AC-24 change is undeclared.

**Fix.** Add one sentence to §0 declaring the AC-24/C leniency and why it is monotone, and log
gold rewrites (see M2). Each of the three additions is needed for F-8's purpose; Appendix A.6
gives the reasons.

#### m4 (minor): the conditional import of S12's 8a function from a spike

**Evidence.** §1.1 says "If S12's `assemble.py` already exports its 8a step … `match_keys.py`
imports that function too." That file is `cpg/spikes/vertical-slice/assemble.py` (p0-2 S12 row),
which does not exist yet. It is spike code, and P1 replaces it. Whether the gold depends on that
spike would then turn on build order.

**Fix.** Choose one design now. Either the 8a segment loop lives in `cpg/ccg/` beside
`screen_fails`, and both `assemble.py` and `match_keys.py` import it, or `match_keys.py` owns its
loop and S12's tests run K1–K22 against S12's own function. The `architect` or `coder` owns that
placement. Either choice makes the dependency fixed rather than dependent on timing.

#### m5 (minor): the "43 literal route segments" figure does not reproduce

**Evidence.** Under four extraction definitions (see the recount section) I count 39, 41, 50 and
563 distinct literal segments. None is 43. The conclusion holds under all of them, with **0
failing** in census scope under both my spec-derived v1.3 screen and the working tree's. The note
gives no extraction definition, so its figure cannot be checked.

**Fix.** Replace "43" with the definition and its count, or with "0 fail under any census-scope
extraction (39–50 segments)".

#### Nits

- **n1, mixed segments.** `GET /f/{x7k2q9w4m1p8z3v6}.json` normalizes to `GET /f/{#}`, so a
  parameter *name* is screened when the parameter shares a segment with a literal. BS2 §1.8 does
  not say how step 7 and 8a treat a partially parameterized segment. Exposure is 0, since no
  census-scope route has one (recount, "mixed: []"). Pin it with S12 as one oracle row.
- **n2, dot segments.** EP §13.4 glosses step 5 as "runs of `/` collapsed" and drops spec step 5's
  dot-segment resolution. That gap predates this note, but it now shows up as a different 8a
  output: `GET /a/x7k2q9w4m1p8z3v6/../b` becomes `GET /a/{#}/../b` instead of `GET /a/b`. Restate
  §1.1's list as "step 5 (collapse, and resolve `.`/`..`)".
- **n3, consumer-only gold.** §1.1 forces provider form. An AC-24 gold anchor on a route that
  nobody in the repo serves (an external API) would convert to `{#}` where the graph holds `{?}`.
  Appendix B.2 does not restrict gold to served routes. Exposure is 0 today. Either restrict gold
  `http` anchors to served routes in a method-owner check at S47, or note the limit.

### Segment recount (brief item 3)

The screen is my emulation of BS2 v1.3 §1.7.1, written from the spec text. It includes T6's
widened `._:+,-` class and is scoped per segment. I also cross-checked every count against the
working tree's `screen_fails`. Both screens gave identical results, and the fuzz found 0
disagreements in 300,000 random strings.

| Def. | What is extracted (census scope, test globs excluded) | Distinct literal segments | Fail the screen |
|---|---|---|---|
| D1 | Python route-decorator paths (AST, decorators only), `include_router(prefix=)`, `app.mount` paths, plus `salesperson/src/api/*` path literals | **39** | **0** |
| D2 | D1, plus every `salesperson/src` path literal and `falkor-chat/web/app.js`'s `api(...)` paths | **41** | **0** |
| D3 | every `/`-leading, whitespace-free string or template literal in census-scope `.py/.ts/.tsx/.js/.jsx` | **50** | **0** |
| D4 | every `http(s)://…/path` in **every** tracked non-binary file outside the exclusions, docs and lockfile included (a wider net than census scope) | 563 | 13, all outside census scope: 11 `package-lock.json` tarball names, 2 model filenames in a `docs/plans` doc, all T6 |

**Spread: 39–50 in census scope (D1–D3), 0 failing in every definition.** The note's 43 is inside
the range but not reproduced. D4 matters only as a reminder that T6 masks dotted version names
such as `binding-win32-x64-msvc-1.2.7.tgz` and `Mistral-7B-Instruct-v0.3.Q4_K_M.gguf`. That
over-redaction is BS2's accepted cost, it is on consumer-only external URLs, and it is not
AC-5-relevant. Appendix A.4 lists D1 and the scripts.

### Obligations on `docs/plans/connected-code-graph-p0-2.md` (brief item 5)

These are the obligations the note creates as written. Items marked **(if M1/M2/m4 adopted)** come
from this review's fixes.

| p0-2 location | Obligation |
|---|---|
| **§3 S06 row** and **§14 row (line 869)** | The accepted contract is EP v1.2 **plus** `-eval-protocol2-ml` v1.0, once gated. The consumers of §14 gain this note. |
| **§3 S10 row** (in flight) | None new. S10's final `screen_fails` must keep BS2 v1.3 outcomes; K1–K22's screen outcomes are spec-derived (reproduced independently here). The figures in the note's §0 and §2 were measured before S10 finished, so S49 re-runs them on the final file. |
| **§3 S12 row** | Its 8a must pass K1–K22 in provider form (note §3, R1), in addition to "SL-1/SL-2 unchanged". **(if m4 adopted)** The placement of the 8a segment function is decided: in `cpg/ccg/` and imported, or S12-owned and conformance-tested. **(n1)** One mixed-segment row is agreed with S49. |
| **§3 S14 row** | **None.** The lists stay raw §3.5-form locators. Writers never screen, and S47/S49 normalize them. |
| **§3 S30 row** | **None**, for the same reason as S14. |
| **§3 S34 row** | **None.** `cpg/tests/golden/answer-ac1.json`'s contract locators enter C through `normalize_contract` at S47; that is S49/S47 work. |
| **§3 S40 row** | `eval-manifest.yaml`'s "this note's path and version" field records **both** documents (note §1.5). **(if M2 adopted)** It also records the screen fingerprint: the BS2 version plus the SHA-256 of `cpg/ccg/bundle.py` at `EVAL_COMMIT`. |
| **§3 S41–S44, S52 rows** | **None.** Appendix B is unchanged. |
| **§3 S45 row** | `normalize` applies 8a after step 7. 8a rewrites are V3 replace-and-log entries (`seq`, rule `8a`, masked count, no content on stdout), and the method owner reviews the log. |
| **§3 S47 row** | Every C source, including S14, S30, S56, `cpg/tests/golden/*` and S34's fixture, is converted through `normalize_contract` before `anchorId` (note §1.2, K27). The held-out manifest's `protocol` is `…eval-protocol-ml.md@1.2+…eval-protocol2-ml.md@1.0` (§1.5). The leak check probes both `keyNorm` forms (§1.4); **(if M1 adopted)** the second form is the pre-8a one. **(if M2 adopted)** AC-24 gold is stored normalized, with each rewrite logged. |
| **§3 S49 row** | `normalize` gains 8a, and `normalize_contract` is exported as a pure function (§1.2). It is applied to every C locator and fixture. `--check-report` probes the second `keyNorm` form. **Oracle K1–K29** joins the "Oracle cases pass" list, plus K2 ≡ K20 (m2) and, **(if M1 adopted)**, the writer-spelling case from Appendix A.5 row 3. Add **S12 and S10 to its dependencies** (m1). **(if M2 adopted)** Add a re-normalize-and-compare verb that the P1 harness can call. |
| **P1 gate harness** (EP §14 row) | It imports `normalize_contract` for AC-24 gold contract anchors and never reimplements it (§1.2). **(if M2 adopted)** Instead, it asserts the stored gold and anchor ids against a gate-time re-normalization, and stops on any difference. |
| **§15 follow-ups line** ("`data-scientist` — `-spec2` F-8 …") | Mark F-8 delivered by `-eval-protocol2-ml` v1.0, encoded at S49. |
| **§16.1 row BS2-21** | Change "Declined from a P0 step" to placed: S49, with S40, S45 and S47 as above. |

### What's solid

- **The core identity reasoning is right and executable.** v1.3 is v1.2 followed by one
  segment-wise map. Placeholders are fixed points, the provider-form choice is argued from B.0,
  §4.1 and `serves[]`, and `{#}` ≠ `{}` keeps EP §6.3's N2 sound. All of it held under adversarial
  execution.
- **The oracle is well built.** It covers T1–T6 one by one, ordering (step 6 before the screen in
  K13, step 7 before it in K15), the accepted residuals (K17, K18), idempotence (K20, K21) and the
  first segment (K22). Its "decided by" column is accurate against my rule-by-rule emulation
  (A.1).
- **The successor-versus-errata argument is correct.** The "steps 1, 4, 5, 6 and 7" set is
  closed, so 8a is a normative identity change. The `Extended by:` pointer is in EP v1.2's header.
- **Importing the screen is the right call** (M2 only adds a drift guard). R2's reasoning, that
  the raw literal stays in eval artifacts with the repo's own exposure, is sound.

### Open questions

1. **For `data-scientist`.** Should the AC-24 gold leniency of m3, where every gold-writer slip is
   now silently corrected, be logged and counted in S47's report as AC-5's V3 rewrites are? I lean
   yes, but it is a method call.
2. **For `architect` or `teco`.** Where should 8a's segment function live (m4)? `cpg/ccg/` is my
   suggestion, but it widens S10's or S12's file set. That choice belongs to whoever owns that
   sequencing.

## Pass 2

### Scope & verdict

**Reviewed:** `docs/plans/connected-code-graph-eval-protocol2-ml.md` v1.1, revised in place: §5
holds the dispositions and §3 the obligations.

**Method:** my own scratch scripts. `norm.py` is the v1.2 steps with `dots=True` for the restated
step 5. On top of it, `pass2.py` adds the v1.1 oracle and leak rule, and `interact.py` probes how
the fixes combine. I re-ran the 300,000-string fuzz of `screen_fails` against my spec emulation on
the current (still-modified) `bundle.py`: 0 disagreements. I edited only this file.

**Verdict: approve with suggestions.** There are 0 blockers and 0 majors, plus 3 new minors and
1 new nit. All of them come from the fixes interacting. The stopping condition at the end of this
pass closes them without a Pass 3.

CPG: considered, not relevant — a method document; every question was answered by executing
scratch emulations against the working-tree `cpg/ccg/{bundle,ids}.py`.

### Pass 1 dispositions

- **M1: fixed.** I ran the pre-8a probe under EP §9.4's v1.2 pattern with removal by matching
  form. K29a fails, K29b passes, K36 passes, K37 passes, K38 fails, all as stated (`pass2.py`).
  The false positive from A.5 row 3 is gone.
- **M2: fixed**, with the departure judged below. K35: 0 differences under the same screen; under
  a T1-at-12 double, exactly K33's id changes. Gold is stored normalized (§1.2), and
  `anchor-not-in-graph` gives the missing signal (§1.5). New finding P2-m1 is about this guard.
- **m1: fixed.** §3 matches my Pass 1 item 5 row for row. Two rows are missing; see P2-m4.
- **m2: fixed.** §1.3's class admits `{#}`. K34 holds.
- **m3: fixed.** Declared in §0, and gold rewrites are logged and counted.
- **m4: fixed.** One shared function in `cpg/ccg/`, with a fallback.
- **m5: fixed.** 39 segments with 0 failing reproduces my D1 exactly. I did not check the "51
  paths" figure; my extractor counts paths under a different definition.
- **n1, n2, n3: fixed.** K30–K32 and K39 hold. n2's fix introduces P2-m3.

**Oracle (brief item 4).** I reproduced 39 checks and all hold. K26 is not executable before S49.
- 26 `http` unit rows under both screens, each output a fixed point: 26/26 and 26/26;
- K23;
- K24, K25, K27, K28, K34, K35 and K39;
- K29a, K29b, K36, K37 and K38.

§2's own list (27 + 3 + 3 + 1 + 5) also sums to 39. So "37 of 38 executable checks" is a counting
slip (P2-n1).

### The departure (brief item 2): fingerprint for diagnosis, id comparison as the stop

**Sound, given P2-m1 and P2-m4.** I see no rule a screen change could bypass by hiding inside
`bundle.py` while leaving the ids intact. And a hash-based stop would also fire on edits that
change no outcome. Every scoring input that depends on the screen is a contract id:
- key-item `anchor` and `serves[]`;
- S(u), which is built from U (EP §3.5);
- the probe, chosen from those ids;
- AC-24 gold;
- C.

`renormalize-check` covers all of these. The leak probes are recomputed at check time on both
sides, so they cannot drift against each other. There are two holes:
- the n3 consumer-form gold (P2-m1);
- the **graph side**, which no eval-side check sees (P2-m4).

The `units` served points in `eval-manifest.yaml` are not re-checked either. They feed only n3's
"is it served?" test, so a missed change there can only re-trigger the method owner's
source check.

### New findings

#### P2-m1 (minor): `renormalize-check` cannot reproduce an n3 consumer-form gold id

**Evidence.**
- §1.5 says the check "re-runs `normalize_contract` … over every stored writer-form contract
  locator".
- n3's `{?}` (§1.2) is a method-owner decision, not a function of the writer form.
- In `interact.py`, re-running the provider form on K10's input does not equal the stored
  `{?}` id. Re-running with the stored form does.
- **Two readings follow.** Taken literally, any unserved masked gold **stops every gate**. If an
  implementer instead skips the rewritten locators, a screen change that unmasks the segment is
  never seen, and AC-24 scores it as not understood with no flag.

**Fix.** Store the form (`provider` | `consumer`) per gold locator in S47's rewrite log, and make
it an argument of the shared function and of `renormalize-check`. Add **K40**: an n3 gold re-run
under the same screen gives 0 differences.

#### P2-m2 (minor): §1.4 does not say which object the pre-8a form is computed from

**Evidence.** §1.2 stores gold already normalized and keeps the writer's original beside it. If
the pre-8a form is computed from the **stored** locator, it gives `POST /hooks/{#}`, and the
literal disappears. In `interact.py`:
- a report leaking a **held-out gold** literal `post /hooks/x7k2q9w4m1p8z3v6` **passes**. That is
  a silent leak;
- a report naming a **dev gold** literal, which is public by the removal rule, **fails**.

Computed from the writer original, both come out right. K29a catches this for AC-5 records, but
no oracle case covers AC-24 gold. M1's fix and M2's fix meet at exactly this point.

**Fix.** In §1.4, say that both probes, and the dev/C values they are removed against, are
computed from the writer's original object, for AC-5 records and AC-24 gold alike. Add **K41**:
the held-out gold literal fails, and the same literal as dev gold passes.

#### P2-m3 (minor): n2's dot resolution breaks idempotence for percent-encoded dots

**Evidence.** Step 5, including the new dot half, runs before step 6 decodes `%2E`, which is an
unreserved `.`. In `interact.py`, each of these inputs changes again when normalized a second
time, so idempotence fails:

| Input | First pass | Second pass |
|---|---|---|
| `GET /a/x7k2q9w4m1p8z3v6/%2E%2E/b` | `GET /a/{#}/../b` | `GET /a/b` |
| `GET /a/%2e/x7k2q9w4m1p8z3v6` | `GET /a/./{#}` | `GET /a/{#}` |
| `GET /a/%2E%2E` | `GET /a/..` | `GET /` |

This falsifies §1.1's idempotence clause, which §0/m3's monotonicity argument rests on. Exposure
is nil, since no census route is encoded that way.

**Fix.** The function is shared with the system, so this is a spec choice for `architect` (BS2).
Either resolve dot segments after step 6, as RFC 3986 §6.2.2 orders them, with **K42**
`GET /a/%2E%2E/b` giving `GET /b` as a fixed point; or narrow the idempotence claim to exclude
encoded dot segments.

#### P2-m4 (minor): §3 lacks the graph-side and n3-form obligations

**Evidence.** `anchor-not-in-graph` signals divergence on the system side, but it does not stop
the gate. The guard only holds if P1's assembler, which replaces `assemble.py`, imports the shared
8a function. No §3 row says so. The S47 and S49 rows also do not carry P2-m1's stored form.

**Fix.** Add a row: "P1 assembler: imports the shared 8a function (§1.1), never reimplements it".
Add the form field to the S47 and S49 rows.

#### P2-n1 (nit): oracle tally

The tally reads "37 of 38 executable checks", but §2's own observed list sums to 39 checks run,
with K26 not run. Restate it as 39 run, 39 hold, plus K26 pending.

### Stopping condition (no Pass 3 needed)

v1.2 is acceptable, and `teco` may close this without another analyst pass, iff all of these hold:
1. §1.5 and §1.2 make the gold form an input to `renormalize-check`, and K40 is in §2;
2. §1.4 names the writer original as the source of both probes and of the removal values, and K41
   is in §2;
3. either K42 is in §2, with BS2's owner agreeing to the step order, or §1.1's idempotence claim
   excludes encoded dot segments;
4. §3 carries the P1-assembler row and the form field;
5. the tally says 39.

**Falsifier:** run the scratchpad's `interact.py` (I1–I3) against the stated K40–K42 outputs. Any
mismatch reopens the finding concerned.

## Appendix (Pass 1)

### A.1 Oracle reproduction (scratch `oracle.py`)

- 22 `http` rows × 2 screens (my spec emulation, and the working tree's
  `cpg/ccg/bundle.py:282` `screen_fails`): **44 checks, 0 mismatches**. Each check asserted
  `out == expected` and `normalize(out) == out`.
- K23: `mcp__falkor__x7k2q9w4m1p8z3v6` gives `x7k2q9w4m1p8z3v6`.
- Rules each failing segment trips, from my emulation:

| Row | Rules tripped |
|---|---|
| K2, K3, K12, K22 | T1, T6 |
| K5, K6 | T6 |
| K7 | T3 |
| K8 | T3, T4, T6 |
| K9 | T5 only |
| K10 | T2 (last segment only) |
| K15's raw `{…}` | T1, T6, but it is never screened because step 7 runs first |
| K13 raw | none, because `%` splits the runs. It fails only after decoding. |

### A.2 Coarsening and idempotence (scratch `adv.py`)

The key set is 38 segment shapes at one and two segments, 1,682 keys, including lowercase-verb
variants. The shapes include the four placeholders, `%7B%23%7D`, `{%23}`, `*`, `{p:path}`,
`<int:id>`, `[id]`, `:id`, `..`, `.`, `%6d`/`%2f` escapes, the `:`/`,`/`.`-chunked T6 forms,
mixed `{x…}.json`, `AKIA…`, `%41KIA…` and `sk-…`.
- Non-idempotent under v1.3: **0** (my screen), **0** (`bundle.py`).
- Non-idempotent under v1.2: **0**.
- v1.2 classes split by v1.3: **0**.

### A.3 Merge class

- Merged pairs outside §1.3's stated class: 6,300.
- After admitting `{#}` as a member: **0**. Example: (`POST /3f2a9c1e5b7d`, `POST /{#}`) gives
  `POST /{#}`.

### A.4 Recount detail (scratch `segcount.py`, `urls.py`)

**Scope.** `git ls-files -co --exclude-standard`, minus `cpg/`, `deprecated/`,
`*connected-code-graph*`, `docs/BACKLOG.md`, EP §3.3's test globs and `node_modules`. That left
134 code files. The content-check exclusions (EP §13.1 step 3) could not be emulated without
S48's kit.

**D1's 39 segments:**

> `advance api batch catalog channels confirm deletion diff document-updates documents due health
> history input matches materialize mcp messages order participants pending presenter readiness
> recheck reject reset reset-all search session shop snapshots state step-runs threads trace
> versions workflow-defs workflow-runs workspaces`

**Increments:**
- D2 adds `...` and `products`.
- D3 adds `0 chat cmdline completions embeddings models proc v0 v1`.

**Sources.** 59 Python route/mount paths were found: `falkor-chat/server/falkorchat/{api,
storefront_api,app}.py`.

**A defect I hit and fixed.** A first D1 pass counted 338 segments because it matched every
`.get(` call, not just decorators. It was fixed by restricting to decorator nodes.

### A.5 Leak-check simulation (scratch `leak.py`, EP §9.4's v1.2 pattern verbatim)

"Note" is §1.4's raw writer-form probe. "Alt" is the pre-8a v1.2 form proposed in M1.

| # | Report / held-out writer form / dev value | Note | Alt |
|---|---|---|---|
| 1 | `missed post /hooks/x7k2q9w4m1p8z3v6 here` / K2 / `POST /hooks/a1b2…` | FAIL | FAIL |
| 2 | `missed post /hooks/{#} here` / K2 / `POST /hooks/a1b2…` | pass | pass |
| 3 | `dev route GET /shop/api/products ok` / `get /shop/api/products` / `GET /shop/api/products` | **FAIL** (false positive) | pass |
| 4 | same report / `GET shop/api/products` / same dev value | pass | pass |
| 5 | `dev route GET /api/items/{} ok` / `GET /api/items/{item_id}` / `GET /api/items/{}` | pass | pass |
| 6 | `held-out GET /api/items/{item_id} missed` / same held-out form / — | FAIL (new probe class) | pass |

### A.6 Brief item 2, the three additions

| Addition | Needed for F-8? | Moves AC-5 thresholds/scope? | Moves EP §9 split/leak semantics? |
|---|---|---|---|
| `normalize_contract` on AC-24 gold | **Yes.** Without it a masked contract-point question is unscorable, and gold writers cannot screen (F-8). | No. It is AC-24, and it adds leniency (m3). | No. AC-24's split is by `qid`. Timing issue: M2. |
| C via `normalize_contract` | **Yes.** K27's leak path is real. | No. | Contact can only grow (monotone by idempotence, m3). That is conservative, and exposure is 0 today. |
| Leak probe in both forms | **Yes**, a narrow need: the normalized probe can be removed by a dev `{#}` twin. | No. | **Yes, beyond the need:** a false positive and a new probe class (M1). The pre-8a form meets the need without either. |
