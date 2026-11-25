# Close-out CO-3 implementation gate (curl/wget pack, Markdown inline-span carrier)

> **Status:** active · **Owner:** `analyst` · **Tracks:** C-1001 (P1 close-out CO-3)

**Scope.** Commit `2b61c3c0` (executed on a `git archive` of that sha, committed into a scratch repository so the tests that call `git` can run; the user's tree untouched). Baseline: `docs/plans/connected-code-graph-p1-closeout.md` §3.3, §3.6, §4 CO-3 row. Evidence `cpg/eval/results/co3/`. Root `REVIEW.md` applied (no component `REVIEW.md`). No held-out or custodian data read; the dev gold's four DEF-1 paths are the ones the implementer's own script names.

**Verdict: needs changes.** One blocker: the commit turns an existing offline test red and the report does not say so (B1). The pack itself is sound on every decoy axis I planted and its mutation coverage is good; the open decision is the shared `urlLike` (M1), where I recommend a per-lifting option before the freeze commit F.

CPG: considered, not relevant — a pack YAML, a Markdown extractor change and a registration line, read and executed directly; no loaded `cpg_<component>` models this code.

## Checks run

| Check | Result |
|---|---|
| `test_pack_shell_http_client.py`, `test_co3_curl_end_to_end.py`, `test_markdown_extractor.py`, `test_refresh.py`, `test_packrt_generic.py`, `test_rule_pack_schema.py`, `test_rule_pack_leaf_mutation.py` | 1089 passed, 1 skipped (the 5.5-minute opt-in leaf mutation run, **not** run); `test_markdown_extractor.py` first failed only for lack of a git repository in the archive, then 194 passed |
| fixture gate suites `test_fixture_gate_p1.py`, `_ac.py`, `test_fixtures.py` (`-m "not live"`, exit status read) | exit 0 (3 / 8 / 80); live tiers not run: the `DEFAULT_PACKS` change is therefore **unverified against the live fixture build** |
| `test_shell_extractor.py`, `test_assemble_rows.py`, `test_script_resolver.py`, `test_ac24_smoke.py`, `test_gate.py` | exit 0 |
| `tests/test_superset_dev_tool.py` at `2b61c3c0` vs `2b61c3c0^` (both in a committed scratch repository) | **1 failed vs 155 passed** (B1) |
| the implementer's `co3_dev.py` re-run | reproduces `summary.json` exactly (6 items, 4 dev targets hit, 7 confirmed / 7 partial-key / 1 runtime-built / 5 prompt-mention consumers) |
| own probes (`an/c3b.py`, `c3f.py`): 47 decoy/twin pairs through the real shell and Markdown extractors, pack runtime, linker and answer builder with a hand-made provider fact | results below |
| own mutants: 24 on the pack YAML and the carrier | 21 killed, 3 survived (below) |

## Findings

### B1 (blocker, verified): `test_superset_dev_tool.py::test_every_cpg_file_the_build_path_imports_or_reads_is_pinned_by_the_record_or_the_tool_line` is red

At `2b61c3c0^` the file passes (155); at `2b61c3c0` it fails: `packs <= set(sd.EXTRA_PIN_FILES) and len(packs) == 13` sees `cpg/packs/shell-http-client/pack.yaml` and `negatives.yaml` as unpinned, because the test reads the **live** `refresh.DEFAULT_PACKS`. The commit message and the ledger list the green suites and omit this file. Root cause: the test pins "the packs `DEFAULT_PACKS` read" for a record frozen at an earlier pack set, so any added default pack reddens it.

Fix route: do not edit `EXTRA_PIN_FILES` or the `-tool` line (the docstring says a post-pin fix needs a new tool id; the committed line holds exactly the 25 keys). Change the test's expectation to the pack set the record was built with (an explicit list of the 13 files) and assert that the packs added since are named, or compare `DEFAULT_PACKS` against that list minus a stated set of later packs. Planted check for the fix: the test goes red when a pinned pack file is removed from the list and stays green when a later pack is added.

### M1 (major, verified): `urlLike` makes every `/`-leading flag value a confirmed consumer; 4 of the 7 confirmed shell consumers in this repository are such artifacts

Corpus measurement (every tracked `*.sh`/`*.md` through the pack, `an/c3c.py`): 20 consumers, of which `confirmed/lifted-role` are `GET /api/v0/models`, `GET /dev/null` x3, `GET /health`, a `/maven2/...jar` and a `/work/plugins/...jar`; the `/dev/null` x3 and the `/work/plugins` key come from `-o`/`-O` output paths, i.e. 4 of 7. Planted (`an/c3b.py`, provider fact of the same key):
- `curl -o /tmp/health http://h/x` with provider `GET /tmp/health` gives a **confirmed `key-match` link** to the provider;
- `curl -d /health http://h/other`, `curl -H /health ...`, `curl --data-raw /health ...` link `GET /health` **confirmed** to a provider whose route is `/health` (URL was `/other`);
- `curl -K /etc/curlrc`, `-w '/health\n'` give consumers `GET /etc/curlrc`, `GET /health\n`;
- constant `ftp://h/pub/file` and `file:///etc/passwd` give consumers `GET /pub/file`, `GET /etc/passwd`, linked confirmed to providers of those keys.

In this repository no provider route matches (all route paths in `falkor-chat/server/falkorchat/api.py` start with a literal segment such as `/channels/`, `/threads/`, `/workflow-defs/`; `app.mount("/", StaticFiles)` is not a decorator route). So no wrong edge exists *today*, verified only against the decorator routes (a catch-all or `/{a}/{b}` provider would still need to be a recognised provider fact: with hand-made providers `GET /{a}/{b}` and `GET /dev/{x}` `/dev/null` did **not** link). But the requirement is "at least 95% of confirmed items real" (`docs/requirements/connected-code-graph.md` ~l.349), the label is `confirmed`, and a shell operand starting with `/` is **never** a valid curl URL (curl needs a host), so the leading-`/` acceptance has no true positive for this pack.

Recommendation: land before F a per-lifting option (for example `urlLike: scheme`, default unchanged) in `ccg/packrt/lift.py`, `ccg/schemas/rule-pack-v1.schema.json` and the rule-pack spec, accepting `http(s)://` and a computed-origin run then `/` only, and use it in `shell-http-client`. It must be per lifting, not a change of `is_url_like`: `python-http-client` relies on leading-`/` paths (its own header names "a relative URL without a leading `/`" as the uncovered case). Run: with `ccg.packrt.lift.is_url_like` replaced by the strict predicate, the same probes lose `/dev/null`, `-o /tmp/health`, `-d /health`, `ftp://`, `file://` and keep every twin; `co3_dev.py` gives the same 6 items and the same 4 dev hits, with confirmed consumers 7 -> 3. **Not run:** the effect on `python-http-client` (not touched by a per-lifting option by construction), and a live fixture build. If the owner defers it, the known limits must be recorded as such in the requirements' bound language and the CO-9 rehearsal must report wrong-`confirmed` for http separately (see Answers). The change moves core files before F, as does anything else in CO-3.

### m1 (minor, verified): shell argument positions and wrappers

Not found: a URL after the tenth argument (`a10`), a wrapped `curl` (`sudo`, `time`, `nohup`, `command`), `--url`-less forms are fine (`--url URL` works). These are false negatives only; they should be named in the pack header with the other limits (currently `-X`, `-o`, `ftp://`).

### m2 (minor, verified): the Markdown carrier is stricter than the shell side on credentials

The carrier refuses userinfo (`user:pw@`) and token-shaped text; the shell pack records `curl http://admin:hunter2@h:8200/health` as a confirmed consumer (path key kept, the literal in the bundle's `a0`, as for any shell argument). The key carries no userinfo and the host is dropped, so nothing secret reaches a key or an answer; the inconsistency is only worth a sentence in the pack header. A sentence-final period inside a span (`http://h/health.`) keys `GET /health.` (no link; a false negative).

### nit: surviving mutants

`_URL_CARRIERS` with `wget` added (equivalent: `wget` is not in `COMMANDS`, no mention exists); the `/?#` stop characters of the authority cut (an `@` after `?` in a query would be read as userinfo and the carrier dropped: a false negative, untested); `return None` -> `continue` after a userinfo URL (the next `http(s)` token would be carried; untested), and `found.setdefault` -> assignment (equivalent for one span).

## Answers to the coordinator's questions

1. **Are the negatives real?** Yes. One decoy per axis, each with a live twin, all through the real extractors: comment (decoy `# curl ... URL`, twin the same line with a trailing comment: consumer); single-quoted and `echo` operand (decoys no consumer, twin plain call: consumer); heredoc (quoted: none; unquoted with `$(curl ...)`: consumer, correct); `curl` as an operand of `command -v`, `xargs`, `ssh`, `echo`, and `bash -c '...'`: none; `curls`/`xcurl`/`./curl`/`wget2`/`curl2`: none; scheme-less host: none; Markdown fence, `ftp://`/`file://`/`redis://`, `user:pw@`, token-shaped userinfo/path/query, URL outside the span, uppercase `HTTP://`, empty host: no carrier; twin inline span: `possible/prompt-mention` consumer and link. The eval-wrapped `curl` is a live consumer. Mutants on the pack: all seed positions, callee regex (tight and loose), `callKind`, `ownSite`, `urlLike` on each lifting, the verb, the mention `status`/`reason`/`arg`/`callKind` are killed.
2. **Mutants:** 24 run, 21 killed, 3 survived (nit above).
3. **Known limits (`-o`, non-http schemes, userinfo): acceptable for P1?** Pinned correctly (the tests assert the consumers exist). Not acceptable without M1's bound being met: they are the `confirmed` label on a non-URL. Preferred: the per-lifting `urlLike: scheme` before F. `-X POST` keyed `GET` is a smaller issue: a consumer `GET /health` can match a `POST /health` provider? In the probe `curl -X POST ... /health` with provider key `POST /health` gave **no** link, so the verb gap fails closed here.
4. **Can the `-o` path consumer link to a real provider?** Yes in principle (plant above: confirmed `key-match` when a provider route equals the output path or a data operand), not in this repository (verified against the route decorators; the `/dev/null` and `GET /work/plugins/...` keys match no route). The `/dev/null` x3 are confirmed consumers with no provider; they create contract points that no answer item reaches unless an anchor names them.
5. **Host dropped from every http key: new?** No. Rule 1 lives in `ccg/packrt/httpkey.py:split_url` (shared by `normalize_key_parts` for every consumer URL): `http://other-host.example:9999/health`, `ftp://h/pub/file`, `http://u:p@h/health`, `file:///etc/passwd`, `HTTP://H/x` all lose the authority (run). `python-http-client`'s header says the same ("the runtime's rule 1"). A wrong-host URL therefore links to a provider of the same path, confirmed; it is a documented runtime decision (requirement-level), not an effect of this pack. I did not run the Python pack (needs Joern); the claim is from the shared function.
6. **Pack-set registration:** `packSetHash` changes by construction (it hashes pack id, version and content; `test_assemble_rows.py::test_pack_set_hash_*` pass), which moves every snapshot `manifestHash` and any `ccg-prereg` record taken after this commit; none exists yet for AC-5 (`ccg-prereg-v1`/`ac5-gate-1` is not written). The **frozen `superset_dev.py` record**: the tool's pins already mismatch the working tree for 35 files (including `ccg/packrt/lift.py`, `httpkey.py`, `refresh.py`, the rule-pack schema, `assemble/rows.py`; run `an` script over `sd.pinned_files`), so the tool could not be re-run at HEAD before CO-3 either; the measured result file is untouched. What CO-3 adds is the red offline test of B1, and nothing else. The live fixture build (`test_fixture_binding_live.py` reads `DEFAULT_PACKS`) was not run; the fixture scripts hold no `curl`/`wget` site (probe), so no consumer can appear there, but the pack hash and any pack-count golden are unchecked live.

## Dev evidence

`summary.json` reproduces. Its four DEF-1 hits are measured with a **hand-made provider fact** (stated in `co3_dev.py`'s docstring; the real Python provider needs Joern), so "dev http 18 of 18" and "family L 3 of 3" are an offline proxy until the live ask runs; the 3 shell items hit as `possible` (partial-key), the Markdown span as `possible`/`prompt-mention`, consistent with the plan. Noise (wrong-`possible`/`confirmed` bound) is not computed (no labels); M1 is the known contribution of this unit.

## What's solid

- The Markdown carrier: inline span only (a fence has no call site), first `http(s)` token, quote-pair stripped, refused on userinfo, empty host, token screen and length cap (those mutants killed); the mention rule can only produce `possible/prompt-mention`.
- The shell pack finds the URL after any number of flags (a0..a9), through variables and base+path templates (`"$B/health"`), inside functions, pipes and `$(...)`, and ignores comments, quoted strings, heredocs and operand positions via the parser, not a regex.
- Registration touches only the tuple; the engine, linker and assembler are unchanged.

## Open questions

1. Does the stakeholder accept a per-lifting `urlLike: scheme` option (core change before F) or defer M1 with a recorded bound?
2. Who owns the repaired `test_superset_dev_tool.py` expectation (B1): the engineer of CO-3, with the frozen tool line left untouched?

### Proposed REVIEW.md lines

Target: root `REVIEW.md`, *Always check*. Source: B1 (a registration change reddened a frozen tool's test the report did not list) and M1.
- A change to a registry that frozen records or tools read (`DEFAULT_PACKS`, a pin list) is checked by running every test file that imports the registry's consumers, named by a grep for the registry, at the commit and at its parent; the report lists the files.
- A pack whose lifting trusts a shared predicate (`urlLike`) over a call whose operands include non-URL paths is probed with an operand of each non-URL kind (output path, data file, header value) against a provider of the same key.

## Stopping condition (falsifiable)

This gate closes when: (a) `tests/test_superset_dev_tool.py` is green at the fix commit with the `-tool` line and `EXTRA_PIN_FILES` byte-identical to `2b61c3c0`, and goes red when its listed pack set is edited; (b) either the per-lifting option lands with `shell-http-client` using it, and a rerun of `an/c3b.py`-style plants (output path, data operand, `ftp://`, `file://`, each with a same-key provider) yields no consumer and no link while the twins stay, `co3_dev.py` still hits the 4 dev targets, and `python-http-client`'s tests are unchanged; or the owner records the known limits as a decision with the CO-9 wrong-`confirmed` bound for http reported separately. If any such plant still links `confirmed`, M1 reopens.

Growth-stops check: not applicable (a new review).

## Pass 2 (fix commit `62e9bcb4`)

Executed on a `git archive` of `62e9bcb4` (scratch repository for the `git`-calling tests), baseline `2b61c3c0^` for byte comparisons; user tree untouched. Plants: `an/p2.py` (48 shell lines through the real extractor and pack), `an/p3.py` (9 through the real assembler, linker with a provider `GET /health`), `an/p4.py` (corpus), `an/mut.py` (19 mutants). Live fixture build not run.

**Verdict: approve with suggestions.** B1 and M1 are closed by execution; no blocker. Two minors and nits below.

CPG: considered, not relevant — a pack option, a predicate and a test list read and executed directly.

**Dispositions**
- **B1 fixed.** `test_superset_dev_tool.py`: 155 passed. `cpg/eval/superset_dev.py` and `cpg/eval/preregistration.jsonl`: `git diff 2b61c3c0^ 62e9bcb4` empty. `RECORD_PACKS` = `DEFAULT_PACKS` at `2b61c3c0^` (`git show`, 9 names), and `EXTRA_PIN_FILES` holds 13 pack files, none of `shell-http-client`. Plants: an unpinned imported module (red), an extra yaml in a record pack (red), a name removed from `RECORD_PACKS` (red), a later pack added to `DEFAULT_PACKS` (green). The test therefore does not guard a later pack's files at all, by design (the record is frozen).
- **M1 fixed.** `-o /tmp/health`, `-H`, `-d`, `--data-raw`, `--data-binary @`, `-T`, `-F f=@`, `--output=`, `-X /health`, `-w`, `wget -O`, `ftp://`, `file://`, `-K`, bare `-o /dev/null`: no `/health`-keyed consumer, no link (only the real URL's own key, e.g. `GET /x`). Twins stay: `http`/`https`/`HTTP`, port, userinfo, `$B/health`, `"$B"/health`, `"http://$H/health"`, `--url`, `wget -qO-`. Corpus (`an/p4.py`, shell): confirmed 3 (all real URLs), `possible/partial-key` 7, `runtime-built` 1. `co3_dev.py` rerun reproduces the committed `summary.json` (4 of 4 dev hits, confirmed 3). Default mode unchanged: `diff -rq` of all packs against `2b61c3c0^` shows only the new directory; `urlLike: true` stays in `python-http-client` (6 liftings) and the Markdown `mention-url`; `lift.py` takes the strict branch only for the string `"scheme"`; its test files and `test_p1_49a_*`, `test_packrt_generic.py` are green. Userinfo as a live twin: agree. `http://user:pw@h/health` is a valid curl URL, so a real consumer; the key keeps no authority, and the literal sits in `a0` like any shell argument (m2, pack header now says so).
- **m1 fixed in prose.** The header names the tenth-argument, wrapper and sentence-period limits as unpinned false negatives.
- **m2 fixed in prose** (pack header).
- **Surviving nits:** the `wget` mutant is still equivalent; the other two Markdown-carrier mutants were not re-run.
- **Same-revision fixes.** The schema enum `[true,false,"scheme"]` rejects `1`, `0`, `None`, `"strict"`, `"Scheme"`, `""`, `["scheme"]` (`pack_errors`), so the `lift.py` truthiness (`lifting.get("urlLike")`, then `mode == "scheme"`) cannot meet an unknown string; schema tests green. 1484 passed, 1 skipped (the opt-in leaf-mutation run, not run) over `test_rule_pack_schema`, `_packrt_lift_options`, `_pack_shell_http_client`, `_script_resolver`, `_shell_extractor`, `_markdown_extractor`, `_packrt_generic`, `_refresh`, `_superset_dev_tool`, `_rule_pack_leaf_mutation`.
- **Question 5 (the Markdown absolute-path guard).** The claim holds. `extractors/markdown/extract.py:67` `_SCRIPT` has no leading `/` and the token is `fullmatch`ed (l.148); 13 forms (`/opt/u.sh`, `//opt/u.sh`, `bash /opt/a/u.sh`, `FOO=1 /opt/a/u.sh`, fenced, ...) yield no callee or argument starting with `/`. Removing `text.startswith("/") and not dynamic` from `ccg/assemble/rows.py:655` leaves 632 tests green over the resolver, assembler, shell, Markdown and end-to-end files. It is dead for shell as well: a non-dynamic shell miss returns at `rows.py:613`. It can matter only for a mention record with a leading `/` from a source other than this extractor, where `_segments` would strip the `/` and suffix-match (l.615). **Recommendation: keep it** (one line, defends the tail match against a relaxed `_SCRIPT` or a second carrier), but say in its comment that it is a second line, and pin it with an assembler-level hand-made mention record whose callee is `/opt/app/scripts/u.sh`; that is the only input that can turn it red, and I did not run that test. Delete it only if the owner prefers no unpinned guards; the cost of deleting is the silent suffix match for an absolute path.

**New findings**

**N1 (minor, verified): the P2-1 fix trades recall for a safe parse on a constructed class.** `_hide_expansion_semicolons` now returns the original bytes when a `${` stays open (`extract.py:135`), so the retry fails. Files with an unclosed `${` in a single quote or comment *and* a real `${v:+$v; }` lose every call and get a `parse-error` warning; at `2b61c3c0^` the same files gave `echo`, `curl` with no warning (plants `echo '${'`, `# use ${x` before and after the real expansion, base vs fix). Corpus: 173 shell files, 14308 calls, 7 warned files at both revisions, so no loss here. False negatives only, and the docstring states the behaviour. Suggested: leave it (fail-closed is the better side for a call name with `;`), or hide only up to the last balanced close; either way list it in the shell README limits.

**N2 (minor, verified): a dynamic-prefix output or data operand is still a `possible/partial-key` consumer and links.** `curl -o "$D/health" http://h/o` and `curl -d "$X/health" http://h/o` give a second consumer `GET /health` (`possible`, `partial-key`), linked to a `GET /health` provider (`an/p3.py`, d3/d4). It is bounded to `possible` and has no flag grammar to avoid it, but the pack header's "a path operand (`-o /tmp/out`, `-d /health`) names no URL" reads as covering it. Suggested: one header sentence naming the computed-prefix operand as a remaining `possible` false positive; the CO-9 wrong-`possible` bound is where it is counted.

**Nits.** Mutant `^https?://` -> `^https?:` survives (`is_url_like_scheme("http:x")` is False either way, so the mutant is only visible on `http:` followed by a non-slash that the test list lacks; add `("http:/x", False)`). `mode=True` default of `_url_like_alternatives` is unreachable (its only caller passes `mode`; mutant to `"scheme"` survives): drop the default. Mutants on `is_url_like_scheme` and `_url_like_alternatives`: 19 run, 17 killed, 2 survived (those two); all 8 `lift.py` branch mutants (mode ignored in `any`, in the filter, at the call, truthy mode, inverted, `all`, `[]` for `None`, unresolved dropped) killed.

**Stopping condition (falsifiable).** Closed when (a) `test_superset_dev_tool.py` is green with `superset_dev.py` and `preregistration.jsonl` empty-diff against `2b61c3c0^` (true now); (b) with `lift.py` as at `62e9bcb4`, `an/p2.py`'s decoy list yields no consumer keyed by an operand that is not the URL, except the computed-prefix `possible` case in N2, and its twin list yields the same keys; and (c) `co3_dev.py` reproduces `summary.json` with 4 of 4 hits. Reopens if any literal-path operand links `confirmed`, if a pack other than `shell-http-client` differs from `2b61c3c0^`, or if the corpus count of `confirmed` rises above 3 without a real URL behind it.

Growth-stops check: this pass adds findings and dispositions only; no prior text restated.
