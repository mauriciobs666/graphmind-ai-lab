# Methodology review — connected code graph rule-pack spec, §8 infrastructure signatures and §9 severity (S30)

> **Status:** active · **Owner:** `data-scientist` · **Tracks:** C-1001 (M10) · **Reviews:** `docs/plans/connected-code-graph-rule-pack-spec.md` (uncommitted, S30), §8, §9, with §10 (N-18, N-22) and §11 read against the eval protocol

**Verdict: needs changes** (light: edits to §8.2, §9 and §10 only, no redesign). The format itself is fine, and §8 and §9 claim no accuracy figure. The five majors are about what the tests can prove: the §9.6 vectors and the §8 samples and negatives cannot fail where it matters (REVIEW.md: a check that cannot fail is a finding), and §9's rows disagree with its formula. Fix these before S32 encodes them.

Evidence bar: every finding below is **verified** (run) unless marked inferred. I extracted the 45 `signatures` from §8.2 and ran them in Python, and I implemented §9's formulas and mutated them. Scratch scripts are in the session scratchpad and were not committed.

## Findings

### Major

**M1. §8's samples are circular, and at P5 the pack will be tuned on its own evaluation set.**
- Each `sample` is written by the author of its `regex` (§8.1), and S32 asserts only that the regex matches it. Verified: all 45 samples match their regexes, and none of the 7 shared negatives matches any entry. That is a typo guard, not evidence that the pattern matches what a runtime prints. §8.2 itself says the patterns were checked against samples and negatives "not against recorded failures".
- §8.2 then lets P5's ≥ 5 recorded failures "add or drop entries". In this lab's own protocol (`docs/plans/connected-code-graph-ml.md`, held-out discipline) an item that is debugged becomes dev. A pack tuned on the same ≥ 5 failures then scored on them has no held-out evidence.
- The recorded failures cannot cover most of the pack. By runtime the pack has python 13, jvm 9, node 6, dotnet 6, go 6, any 5. The proving repo is Python plus TypeScript, so jvm, dotnet and go (21 entries) can never meet a recorded failure there. Their validity is the author's recall of exception names.
- Fix (do any two):
  - Source samples from artifacts not written by the regex author. Use verbatim text from the runtime's own source or docs, or from real captures. Run the actual failure for each of the five classes in the lab's two runtimes and store the raw lines as fixtures.
  - Add a `sampleSource` field with values `captured`, `runtime-source` or `authored`. Report coverage by source, and state in §8 that `authored`-only entries are unvalidated.
  - In P5, split the ≥ 5 infrastructure failures into dev and held-out, or record ≥ 10, so editing the pack does not burn the only evidence.
  - Mark jvm, dotnet and go entries `unvalidated` in the pack, and do not let any report count them as supported.

**M2. The negative controls cannot catch a false positive, and the pack has several (verified).**
- The 7 shared negatives are all clean application-exception lines. None contains a near-miss token such as `timed out`, `connection`, `memory`, `137`, `timeout` or `refused`, which is where these regexes can fail. The negatives test "does not match unrelated words", which the patterns already satisfy by construction.
- False positives I ran:
  - `ValueError: Test timed out waiting for user input` and `RuntimeError: request timed out after retries (app-level)` match `python-timeout`. §8.2 admits `timed out` is loose.
  - `ERROR: container exit code 1370` matches `any-oom-kill`. The `137` has no boundary. §8.2 notes that a non-OOM `SIGKILL` also matches, but not this one.
  - `redis.exceptions.ResponseError: oom-kill` matches `any-oom-kill` through the bare `oom-kill` alternative.
  - `INFO exit code 137 handled` matches `any-oom-kill`.
  - Unnarrowed, a Python httpx DNS line matches `dotnet-dns-failure` (`Name or service not known`). `redis.exceptions.ConnectionError: Error -2 connecting ... Name or service not known` also matches the dotnet entry. Cross-runtime leakage happens whenever no runtime marker is in the evidence, which §8.1 treats as normal.
- Fix: add a near-miss negative set, kept per class with at least 3 each.
  - Application text containing the trigger word: `timed out waiting for user input`, `exit code 1370`, a test named `test_oom_kill_handler`, a log line `retrying after timeout (attempt 2)`.
  - Lines from the AC-12 "no match" controls (≥ 20 negative inputs in ML §8.3) should be run through this pack too, since a signature false positive creates a spurious "outside the mapped code" hypothesis there.
  - Add a `\b` after `137` or require `exited with code 137`, drop bare `oom-kill`, and add a test that a regex edit which broadens an entry fails S32.
- The spec's mitigation is that a false match never suppresses a mapped-code hypothesis. That is true of correctness but not of ranking. The outside-branch score is exactly what ML §8.3 evaluates ("should rank first; report its rank"). A false infra signature on a mapped-code bug lowers that bug's top-5 rate in a way the pack's own tests cannot see.

**M3. Recall gaps on the very failures P5 will record (verified).**
- ML §8.3 lists "block a network route" as a planted failure. Realistic outputs miss every entry:
  - `OSError: [Errno 113] No route to host`
  - `ConnectionResetError: [Errno 104] Connection reset by peer`
  - `requests`' `ConnectionError: HTTPConnectionPool(...) Max retries exceeded` (matches only inside the `requests.exceptions.` form)
  - Network unreachable (`ENETUNREACH`, `EHOSTUNREACH`)
  - A bare `Killed` or a container exit with no text
- For a stopped database, the typical Python message `psycopg2.OperationalError: connection to server at "db" ... failed: Connection refused` and redis's `Error 111 connecting ... Connection refused.` match **only** `third-party-exception`. They never match class `connection-refused`, because `python-connection-refused` needs `ConnectionRefusedError` or `[Errno 111]`.
- This matters because §8.1 emits `{class, runtime, id}` and the ranker (ML §8.3) may weight classes differently from the `third-party-exception` bucket. S30 should state which class the ranker is allowed to score.
- Fix: broaden `connection-refused` to also match `Connection refused` and `ECONNREFUSED` as a plain-text token in the any/python entries. Decide whether `No route to host` and `Connection reset` belong in an existing class (probably `connection-refused` or `timeout`) or whether the closed class set from ML §8.3 must change. Either way, put the gap in §8 so P5 can read it as a known hole and not as a surprise.

### Minor

**m1. Detection narrowing is a gate for multi-service evidence.**
- §8.1 says "narrows, never gates", but when any `runtimes` regex matches, entries of other runtimes stop applying. Infra failures are recorded from `docker compose logs`, which interleave services (a Python traceback plus a Node `ECONNREFUSED` line in one evidence blob). The Python marker narrows the pack and the Node line is lost.
- Compose prefixes (`svc-1  | `) defeat the `^\s+at` and `^goroutine` detectors but not the unanchored `Traceback` one, so detection behaves differently per runtime.
- Fix: apply narrowing per evidence block, not across the whole evidence, or drop narrowing and rely on the distinctive names. Add a vector with a mixed Python-and-Node blob.

**m2. Multi-emission is not deduplicated.** §8.1 says one line may emit several classes and "all are emitted". Observed: `httpx.ConnectError: [Errno -2] Name or service not known` yields `python-dns-failure`, `python-third-party-httpx` and (unnarrowed) `dotnet-dns-failure`. Say who collapses these to one cause. If the ranker counts signatures, one failure scores three times. Specify `(class, line)` dedupe, or state that the ranker scores per class.

**m3. No metric is defined for the pack.** §8 gives no statement of how P5 turns recorded failures into a number. The only quantity in the protocol is the outside-branch rank in the ≥ 5 failures, to be "reported". With 5 failures, 5 of 5 correct has a Wilson 95% lower bound of about 0.57, so no recall claim can be made from it. Say in §8.2 that P5 reports a rank and counts, not a rate, and that precision is measured on the negative set (M2), not on the failures.

### Severity (§9)

**M4. The test vectors are not discriminating (verified by mutation).** I coded §9.2 to §9.5 and ran the ten §9.6 vectors against it, then against variants. All of these pass all 10 vectors:
- Injection penalty weight 3 instead of 1 (any weight ≥ 1 passes: both penalised injection vectors land on the floor).
- Authorization penalty weight 2 or 3.
- Injection reduced to two levels (critical iff network × exec and clean, else low). Authorization likewise.
- A flow adjustment of 0 or 2 instead of +1 for secrets.
- `at head, G` base of 0 or 2 instead of 1.

Causes: the two penalised injection and authz vectors both end exactly at 0, so single-penalty behaviour is never tested. There is no vector with `File.role = example` (cap 0), none for the flow +1 or the cap at 3, none with a negative pre-floor value, and no vector for the three middle rows. Rows never exercised: INJ-2, INJ-3, AUTHZ-2, AUTHZ-3, RCFG-2, RCFG-4 (6 of 16). Only SEC-3 touches a test-directory case.

Fix: add at least these vectors (one per mutant killed):
- network × data, clean → INJ-2; local × data, clean → INJ-3; network × exec with one penalty → INJ-2 (kills penalty weight 2 and 3).
- network × read of `sensitive`, clean → AUTHZ-2; internal × write, one penalty → AUTHZ-3.
- exposed-service in a production file → RCFG-2; host-compromise in `docker-compose.dev.yml` → RCFG-2.
- Any finding in an example file → rank 0 (one per kind, kills a missing cap).
- Secret at head, P, confirmed flow → SEC-1; at head, G, confirmed flow → SEC-2; at head, G → SEC-3; removed, P → SEC-3.
- local × data with two penalties (pre-floor −1) → INJ-4 (floor).
Per root `REVIEW.md`, state in the test plan the mutant each new vector kills.

**M5. Rows disagree with the formula where the formula is the authority.**
- §9.5 secrets: a K secret at head under a test directory is 3 − 1 = 2, which is `high`. But SEC-1 says "at head, class K" with no test-directory exception and SEC-2 does not list it. SEC-4 lists only combinations that reach the floor. A reader following the rows gives `critical`; one following the formula gives `high`. The same gap exists for P with a flow in a test directory (2 + 1 − 1 = 2, SEC-2, not listed).
- §9.3 authorization: the base table scores "network × write/delete/privilege change = 3" without requiring `sensitive`, while AUTHZ-1 requires a write on a `sensitive` entity. A network write to a non-sensitive entity is 3 by the table and "every other combination" (AUTHZ-4) by the rows.
- §9.4 configuration: an *exposed-app* entry in a dev variant is 1 − 1 = 0 (RCFG-4 through "any other combination"), while an *exposed-service* in a dev variant is RCFG-3. That is consistent but implicit.
- Fix: make the formula the single source and say rows are derived. Add the rank-to-row mapping once and delete the per-row prose, or state the order of operations (base, flow adjustment, test penalty, cap, floor) and make every row read from it. Resolve the `sensitive` condition in the table.

**m4. The dev-variant glob over-matches.** `**/*{dev,local,test,override}*` matches `docker-compose.latest.yml`, `contest.yml`, `device.yml` and `attestation.yml`, because `test` and `dev` are substrings. A production file named for `latest` or `attestation` loses a rank. Use token boundaries (`(^|[._-])(dev|local|test|override)([._-]|$)`) or `File.role`.

**m5. Severity is policy, not a measured quantity, and the spec should say so.** There is no labelled severity data, and AC-15 requires only that a severity with a reason is shown. So the table is checkable for internal consistency (M4, M5) but not for accuracy. The text should state this to stop a later "severity precision" claim, and note that ranks are not comparable across kinds (an INJ-1 `critical` and an RCFG-1 `critical` come from different bases).

### Relation to §10, §11 and the protocol

- **N-22** (a sample that its regex does not match, or a shared-negative hit) is a valid failing check: a regex typo turns it red. It is the typo guard of M1, not a quality gate. Add near-miss negatives (M2) to its input. **N-18** rejects only `level: severe` and `rank: 4`. Add an N-n for a row whose stated rank contradicts the formula (an M5 row), or the validator will accept an inconsistent table.
- **§11** does not touch §8 or §9. Its contact list is about walkthrough anchors and carries no infra or severity claim. I found no precision or recall claim in §8, §9, §10 or §11 that the eval protocol cannot back. The claim to guard is the future one: §8's P5 validation and O-1 ("§8 only, at P5") should read as dev/held-out as in M1, and the `data-scientist` consult at P5 should be on the dev/held-out split before the pack is edited.
- **O-1** is right that §9 has no method call, but §9's test vectors (M4) are an evaluation-design question and are worth the same consult.

## Disposition summary

| ID | Severity | Fix lives in |
|---|---|---|
| M1 | major | §8.1 (`sampleSource`), §8.2 (P5 split), §12 |
| M2 | major | §8.2 negatives, §10 N-22, `any-oom-kill` regex |
| M3 | major | §8.2 python/any entries, class decision |
| M4 | major | §9.6 vectors, S32 test plan |
| M5 | major | §9.3 to §9.5 rows |
| m1 to m5 | minor | §8.1, §9.4, §9.5 |

## Pass 2 (spec v1.1)

Rechecked by running: `cpg/tests/test_rule_pack_spec.py` (52 passed, via `cpg/.venv`); my own extraction of the 45 signatures from §8.2 (samples, 27 negatives, the capture file); my four false-positive lines; and an independent evaluator of §9.1's YAML run on the 44 vectors against 23 mutants of my own.

### Dispositions

- **M1 (circular samples, tuning on the evaluation set): fixed in part.** `sampleSource` exists and the P5 dev/held-out rule is written (alternating, dev first, held-out scored once, ranks and counts only, consult before edits). Disclosure adequacy is judged below (N1).
- **M2 (negatives): fixed.** 27 negatives (7 shared, 20 near-miss, at least 3 per class). All five lines I ran no longer match: `Test timed out waiting for user input`, `request timed out after retries`, `exit code 1370`, `redis ... ResponseError: oom-kill`, and the Python DNS line no longer reaches `dotnet-dns-failure`. `INFO exit code 137 handled` still matches `any-oom-kill`; that is the disclosed SIGKILL imprecision, not a defect. See N2 for the remaining weakness.
- **M3 (recall gaps): fixed, with a disclosed hole.** `No route to host`, `Connection reset by peer`, Errno 104/113 and the psycopg and redis `Connection refused` lines now match class `connection-refused` (verified). Bare `Killed` and requests' `Max retries exceeded` still match nothing; both are listed under known holes (O-15 covers the class question).
- **M4 (vectors): fixed.** 44 vectors, all 16 rows used. Of 23 mutants I wrote, 20 die (the earlier 8 survivors all die: injection penalty 3, authorization penalty 2 and 3, both two-level collapses, flow adjustment 0 and 2, secrets-G-at-head 0 and 2); 3 survive (N3).
- **M5 (rows against formula): fixed.** Formula in §9.1 is the single source, order of operations stated, rows derived and validated by N-24, `operation: write` defined as always sensitive. Verified: K at head under `tests/` is rank 2 (V37), K+flow under tests is 2 (V41).
- **m1 (narrowing): fixed.** Per-block narrowing; the mixed-blob case yields the Python and the Node signatures. The test confirms that without per-block narrowing the Node signature is lost.
- **m2 (dedupe): specified, not tested** (N4).
- **m3 (no P5 metric): fixed.** Ranks and counts, with the Wilson bound stated.
- **m4 (glob over-match): fixed.** Basename token regex; V25 (`latest`), V44 (`contest/`) kill the substring mutants.
- **m5 (severity is policy): fixed.** Stated at the top of §9.
- **N-18 / N-22 relation: fixed.** N-24 rejects a row whose level contradicts its rank; N-22 now includes near-miss negatives and the mutation list.

### Is the circularity disclosure adequate?

Largely yes, and honest: the 11/34 split is stated in §8.2, the default for an omitted `sampleSource` is `authored`, a pack may not call `authored` entries supported, jvm/dotnet/go are marked as carrying no claim, and the capture file matches the 11 captured samples byte for byte (checked). Three gaps remain, all at the level of what "supported" is allowed to mean:

- **N1 (major-minor, wording/granularity).** `captured` is per entry, but most regexes are multi-alternative and one sample exercises one alternative. `python-connection-refused` has 7 alternatives (ConnectionRefusedError, ConnectionResetError, five Errno codes) and `jvm-connection-refused` has three; the sample covers the first only, and the added alternatives (reset, no route, 113, 101, 104, NoRouteToHost) are exactly the ones the revision introduced after M3 and are authored. "Supported" therefore overstates the entry. Fix: define supported as "every alternative has a captured or runtime-source sample" or split multi-alternative entries; at minimum say in §8.1 that a captured sample supports the matched alternative only.
- **N5 (minor).** Two captured lines are not the failure P5 will see. `python-oom-kill` is a `MemoryError` provoked in-process; a container OOM kill gives the Python process a SIGKILL and no such line (the real signal is the `any-oom-kill` text, which is authored). The capture came from a single machine, Python 3.12.3 and OpenJDK 21, by the regex author, who then fitted the regexes knowing them. So captures are dev data for these 11 entries, not independent validation; say so beside the count.
- The disclosure does not say that the regexes were written or revised after seeing the captures. State it.

### New findings

- **N2 (minor).** The near-miss set is thin where the new broad alternatives are. `any-connection-refused` now matches the bare phrases `Connection refused` and `Connection reset by peer` (capitalised, anywhere in a line). The connection-refused near-misses are underscore or lower-case ones and contain none of those, so app text such as `WARNING: Connection refused handled gracefully by retry wrapper` or `Network is unreachable (cached)` matches (verified) and no negative notices. Likewise `curl: (28)` text. Add capitalised app-text near-misses for these. The mutation list in the test also covers only 6 of 45 entries and one of them is `.*`, which any shared negative kills; add one realistic broadening per class (my trigger-word mutants per class were all caught, so the negative set can already do this; the test just does not ask).
- **N3 (minor).** Three mutants survive the 44 vectors: narrowing the secrets test-path regex to top-level `tests/` only, dropping `e2e` and `__tests__`, and removing the `local` token from the dev-variant regex. Every vector uses `tests/test_x.py` and `dev`/`override` names. Add `src/tests/unit/test_x.py`, `web/__tests__/a.test.ts` (or `e2e/`) and `.env.local`/`docker-compose.local.yml` vectors.
- **N4 (minor).** The §8.1 dedupe rule (`(block, line, class)` and the third-party collapse that keeps `library`) is normative but no test or vector exercises it. The cross-runtime cases assert id sets, not the deduplicated pairs. Add a case: the psycopg line must yield one `connection-refused` pair carrying `library: psycopg`, not two pairs.
- **N6 (nit).** `cpg/tests/test_rule_pack_spec.py` line 747 asserts `... or True`, which cannot fail. Remove it or make it a real assertion.

## Verdict (Pass 2): approve with suggestions

M1 to M5 and the minors are resolved or honestly disclosed; the formula vectors now kill every mutant I had previously shown surviving. What remains (N1 to N5) tightens what "supported" and the negatives can claim and adds a few vectors; none blocks S32 encoding the pack, provided N1's wording lands before a report counts "supported" entries.

## Pass 3 (spec v1.2, hardened tests)

Rechecked by running: `cd cpg && .venv/bin/python -m pytest -q tests/test_rule_pack_spec.py` (101 passed); my §9.1 evaluator on scratchpad copies, 23 mutants against the 44 table vectors plus the test's 11 extra vectors X1 to X11; my per-class trigger-word broadening mutants against the 28 negatives; the test's `pairs_of` on a two-library line; and a top-level versus nested alternative count of the 11 captured entries.

### Dispositions

- **N1 (supported per alternative): fixed in wording; one inconsistency (ruling (a)).** §8.1 now says a sample supports only the alternative it matches, defines *supported* as every alternative sampled, and bars calling partly supported entries supported.
- **N2 (near-miss negatives): fixed.** 28 negatives, at least 3 per class. The capitalised-app-text lines moved to a `known-false-matches` block that S32 asserts *do* match. That is an honest disclosure but means the bare-phrase alternative is deliberately unguarded. See ruling (c) for the mutation breadth.
- **N3 (surviving mutants): fixed in the test, not in the spec (N7).** With the 44 table vectors alone my 3 survivors persist (test-path restricted to top-level `tests/`, `e2e`/`__tests__` dropped, `local` dropped). With X1 to X11 added, all 23 of my mutants die.
- **N4 (dedupe untested): fixed.** Tests cover the psycopg one-pair case, the pymongo own-pair case, the library folded into every other class pair, the Go two-pair case and the block as part of the key. All pass.
- **N5 (captures are dev data): fixed.** §8.2 states the single machine, the regex author, revision after seeing the captures, and that `MemoryError` is not a container OOM line.
- **N6 (`or True`): fixed.** `grep 'or True'` finds nothing.

### Open items

- **(a) Definition of "supported": rule for nested alternatives; fix §8.1's wording.** I counted the 11 captured entries. By top-level `|` only, six are single-alternative (`python-oom-kill`, `python-third-party-redis`, `python-third-party-httpx`, `jvm-connection-refused`, `jvm-dns-failure`, `jvm-oom-kill`); at any group depth only three are. The sample for `python-third-party-redis` covers `ConnectionError` of `(?:ConnectionError|TimeoutError|BusyLoadingError)`, and the `httpx` and `jvm-connection-refused` samples cover one of four and one of three respectively. A sample is a concrete string, so it proves one alternative wherever that alternative sits; the nested count (3) is the right one, §8.2 and the test already use it, and §8.1's "top-level `|` alternative" would call three entries supported on one sample out of three or four. Edit §8.1 to say "every alternative at any group depth" and have the test assert that wording. Not a blocker: no report may use the §8.1 text to over-claim, because §8.2's count and the test are already the stricter one.
- **(b) Third-party dedupe: the rule is class grouping, and one phrase contradicts it.** The "Dedupe, exactly" text says "otherwise each `third-party-exception` is its own pair" and, in the same paragraph, "two entries of one class on one line are one pair". On `sqlalchemy.exc.OperationalError: (psycopg2.OperationalError) ...` the test's reference gives one `third-party-exception` pair with ids `[psycopg, sqlalchemy]` and libraries `[psycopg, sqlalchemy]` (verified). That is the right outcome for scoring (one failure, one pair per class). Replace "each `third-party-exception` is its own pair" with "all `third-party-exception` matches on the line form one pair", and add that two-library line as a dedupe case so S32 cannot read it per-library.
- **(c) My N2 "one broadening mutant per class": does not block.** Run against the 28 negatives, a trigger-word broadening of every one of the 45 entries (class trigger regex such as `[Tt]imeout`, `oom|137|[Mm]emory`, `[Cc]onnection|refused|reset`, `resolve|lookup|name`, `ssl|cert|CERT`, `psycopg|redis|requests`) is caught by a negative of its own class, so the negative set already has the capacity; the test just asserts it for 6 entries (`python-timeout`, `any-oom-kill` twice, `python-oom-kill`, `any-connection-refused`, `go-timeout: .*`). Residual risk: a future edit to a dns, tls or third-party entry is not mutation-checked. Add a parametrised per-class broadening in the test at the next touch.

### New finding

- **N7 (minor).** X1 to X11 live only in `cpg/tests/test_rule_pack_spec.py`. §9.3 says S32 "repeats both checks" on the spec's 44 vectors, and on those 44 alone three of my mutants survive. Either add the eleven rows to §9.3 or say S32 must import the test's vector list.

## Verdict (Pass 3): approve

Every Pass 2 finding is fixed or honestly disclosed, 101 tests pass, and all 23 of my formula mutants die with the vector set the tests use. The remaining items (the §8.1 "top-level" wording, the one contradicting dedupe phrase, N7, a per-class mutation sweep) are wording and test-breadth edits that do not change what any report may claim; they can land with the S31/S32 hand-off.
