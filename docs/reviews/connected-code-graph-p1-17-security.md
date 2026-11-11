# Security review — P1-17 dispatch 2 (binder) and the regex guard

> **Status:** active · **Owner:** `security-expert` · **Tracks:** C-1001 (P1-17)

## Scope and verdict

Lens 1 (code/app), advisory. Reviewed the uncommitted `cpg/ccg/linker/bind.py`, its use in `link.py`/`matchers.py`, and the tests `cpg/tests/test_linker_bind.py`, `test_linker_bind_launch.py`, `slice/test_linker_conformance.py`, against `docs/plans/connected-code-graph-p1-linking.md` P1-17 (trust rules (a)-(e), behaviours (4)(5)(6)(13)(14)). Threat model: launch/binding/mount/port/command facts, URLs and bundles come from untrusted repo files. All probes were run with `cpg/.venv/bin/python`, cwd `cpg/`, `ulimit -v 1500000`, killed on timeout, inputs in the scratchpad `sec-p1-17b/`. Pure in-process probing of the code under review; no service touched, no FR-10 exploitation attempted.

**Verdict: needs changes** (one blocker: ReDoS guard bypass).

CPG: not applicable — the code under review is uncommitted and not in any loaded `cpg_*` graph; paths were traced by reading plus live probes.

## Findings

### B1 (blocker, confirmed) — the regex guard misses `{,N}` / `{,}` quantifiers, admitting catastrophic patterns
`matchers.py:_scan` treats `{` as a quantifier only when `re.fullmatch(r"(\d+)(,(\d*))?", ...)` matches, so `{,5}`, `{,}` read as literal braces. Python's `re` treats them as `{0,5}` / `*`. The ambiguity check (quantified group holding a quantifier/alternation) and both quantifier caps are therefore skipped.
Evidence (`_regex_hit`, 30-char text, 5 s alarm): `(a+){,}b`, `(a+){,99}b`, `(a|aa){,99}b`, `(a*){,}b` all admitted by `regex_pattern_refusal` (returns `None`) and all exceeded 5 s (exponential). Control: `(a+)+b` and `(a|a)*b` are correctly refused. The cap-4 bypass: `\w{,255}\w{,255}\w{,255}\w{,255}\w{,255}\w{,255}x` on 255 chars did not finish in 8 s; so did eight `\w{,}`.
Impact: one hostile regex-form key in a repo file hangs the linker with no timeout (Python `re` is uninterruptible in-process); the 256-char text cap does not help. DoS of the whole link run.
Fix: in `_scan`, accept `{,N}`, `{,}` and `{N,}` per Python's grammar (`\{(\d*)(,(\d*))?\}` with at least a comma or digit; `{,}` and `{,N}` have lower bound 0, `{,}`/`{N,}` are unbounded), count them toward both caps and the ambiguity flag. Better, remove reliance on a token scan: reject any pattern containing `{` that is not exactly `{n}`/`{n,m}` with a digit start, or evaluate via an RE2 binding / a worker with a hard time budget. Add planted tests: `(a+){,}b`, `(a|aa){,99}b`, `\w{,255}`x6, each asserted refused AND, as a control, `(a+)+b` refused today.
Related: `rulepack.regex_problem` (the P1-16 pack-regex gate) shares no quantifier logic, so check other consumers of untrusted regexes (`ccg/packrt/match.py`) for the same `{,N}` blind spot — not reviewed here.

### M1 (major, confirmed) — guard admits ~0.45–0.5 s polynomial patterns, with no global budget
`\w*\w*\w*\w{0,300}x`, `\w*\w*\w*\w{,300}x` and `[\s\S]*[\s\S]*[\s\S]*[\s\S]{0,300}x` are admitted and take 0.44–0.49 s per match on a 255-char non-matching literal (O(n^3)). The guard is per-match; the linker evaluates patterns × literals, so N pattern keys against M literals costs up to N·M·0.45 s. Fix: lower `MAX_UNBOUNDED` to 2 (n^2: ~2 ms) or add a per-link match budget (count/elapsed) that stops evaluating and reports `refused`; time the new bound.

### M2 (major, confirmed) — binder is O(declared-services × launch-facts) per site; a hostile Compose file stalls linking
`_port_match` calls `_service_launch(name, "port")` for every declared service, and each call scans all launch records; `_idents`/`_service_chains` do the same. Probe (one compose file, one port-fact + mount + command per service, 20 loopback sites): 200 services 0.27 s; 1000 services 6.3 s; 3000 services 58 s (≈2.9 s per site, quadratic). Cycles/deep chains: none — `_launch_chains` is a fixed-length walk and terminates; no recursion exists.
Fix: index launch facts once in `__init__` by `(service, kind)` (and by `(repo, path)` for `_launch_chains`), then `_port_match` iterates a port->services index. Add a test with 2,000 services asserting bounded time/call count.

### m1 (minor, confirmed) — absolute compose-file path breaks trust rule (b) when a repo sits at the system root
`_system_dir` joins `base` and `rec.path` with `posixpath.join`; an absolute `rec.path` discards `base`. `_locate` with a `"."` repo accepts any target that does not start with `..`, so `/etc/x` counts as "inside". Probe: repos `{"fx-svc": "."}`, compose file path `/etc/compose.yml`, mount `x`, a unit with `rootPath` `/etc/x` -> binds (`('job',)`). Nothing is opened (lexical only), and it can only reach a unit whose own forged `rootPath` is that absolute string, so the exposure is mis-binding, not file access. Fix: in `_system_dir`/`_locate` refuse a `rec.path` starting with `/` and a `target` starting with `/` (`return None`), and test `/etc/compose.yml` with a root repo.

### m2 (minor, confirmed) — records are keyed by global `id`; a later bundle silently shadows an earlier one
`self._entries[r["id"]]`/`self._launch[r["id"]]` are plain dict stores across all bundles. A forged bundle reusing a legitimate `configEntry` id (ids are derivable) replaces it: with `C1` re-declared under `not.a.server`, the legitimate mention binding went from `('svc',)` to `None` (`p8.py`). It can also attribute evidence to the legitimate entry's id with attacker content. Fix: detect a duplicate id across bundles in `Binder.__init__` (and `Corpus`) and raise/skip both, or key by `(unitId, id)`.

### m3 (minor, confirmed) — reader-side metadata set is narrower than the writer's (trust rule (d) parity)
`bind.forbidden_host` only handles dotted-quad/IPv6 literals; the writer's `_forbidden_target` also covers `inet_aton` forms and `::/96`, NAT64. The hole is closed upstream in practice: `parse_bind_url` returned `None` for decimal, octal, `169.254.43518`, mapped and NAT64 forms, and classes `169.254.169.254`, `fe80::`, `fd00:ec2::254`, `2002::` as `ip` (never bound), and `localhost.`/trailing-dot is refused. The residual: `http://0xa9fea9fe/` parses as `external` and, if a Compose service of exactly that name is declared, is classed `service` (reader does not re-run the writer's rule). Since nothing dials, the effect is only binding to a service the attacker named. Not exploitable as a metadata bind; fix is hygiene: call the writer's `_forbidden_target` (or move it into `ccg.valuehash`/`bundle` for one definition) instead of a second set. RFC1918, `0.0.0.0` and `host.docker.internal` stay `ip`/loopback as designed.

### m4 (minor, theoretical-inferred) — Binder does not re-validate `bindParts`; it relies on `bundle.validate` upstream
With a copied `bindKeyCheck` (the check is a public header value, not a secret), `bindParts` of `{}`, a missing `hostHmac`, a string `pathHmacs`, a list `host` or a string `parts` raised `KeyError`/`AttributeError` (probe `p4.py`) — one hostile bundle crashes the whole link. The schema (`bundle.py` V4a–V4i) rejects these shapes, so this holds only if every caller validates before `Binder(...)`; not yet enforced by the constructor. Fix: document the precondition in the class docstring and have the constructor defensively skip an entry whose parts raise (or run `bundle.validate`).

### m5 (minor, by-design, open for the maintainer) — config `mcpServers.<server>.url` entries bind system-wide, unscoped by repo, file or format
`_config_urls` matches on `keyPath` only. A `.mcp.json` in repo `fx-job`, a compose-format file, or any path (`node_modules/x/README.json`) naming `mcpServers.<server>.url` binds mentions of `<server>` in every other repo (`p4.py`: all three bound `svc`). Same for `_servers` (launch chain). Any repo in the manifest can redirect another repo's `mcp__<server>__*` mentions to its own unit; the result stays `possible/prompt-mention` so the damage is attribution, not confirmation. Also port squatting: a unit declaring a port fact for 8200 turned the legitimate link into `port-shared` and gained its own link (`p9.py`) — documented behaviour, noted for completeness. Suggested: scope config entries to the mention's repo (plus the manifest root) or require `format == "mcp-json"` and `path` basename `.mcp.json`; the plan is silent, so this is a design decision for the architect.

## Probed and held (all confirmed)
- Lexical mounts: `../../../../etc`, `/var/run`, `..` (resolves to the file's own repo, correct), `../../fx-job/../../..`, sibling prefix `../../fx-job-evil`, `a\x00b`, 5,000-deep `../` and `a/`, backslash separators, `%2e%2e`, unicode — none bound an outside path; outside targets yield `Unmapped` with one stable ref; each ran in 7–15 ms. Sibling-prefix is exact (`base + "/"`).
- Service identity (c): `.mcp.json` `explicit` and compose `origin` stay apart (test and code; `_servers` vs `_service_launch`).
- Key epoch: unusable bundle -> parts re-derived from `bindValue`, parts-only entries unbound, one value-free `bind-key-mismatch <unit>` line; metadata host forged into usable `bindParts` (`service`/`loopback` kind with `169.254.169.254` or `[::ffff:a9fe:a9fe]`) bound nothing.
- Resource/termination: launch chain has no loop; huge `bindValue` (100k segments) returned in bounded time via the parser.
- Purity: `bind.py` imports no `os`/`pathlib`/`socket`; its token paths only hit set lookups of known files.

## What's solid
Lexical path handling, `Unmapped` ref canonicalisation, origin separation, forbidden-host drop at the `_from_parts` choke point, deterministic ordering, the value-free mismatch line, and `_config_urls` refusing a clause other than `keyPath`.

## Open questions
1. Architect: should `mcpServers` config URLs be repo-scoped (m5)? 2. Who guarantees `bundle.validate` runs before `Binder` (m4)? 3. For B1, is an RE2 engine (`google-re2`) acceptable instead of a hand-written scanner?

### Proposed REVIEW.md lines
Root `REVIEW.md`, "Always check": *A regex/pattern complexity guard is checked with planted catastrophic patterns for each quantifier spelling the engine accepts (`*`, `+`, `?`, `{n}`, `{n,}`, `{,n}`, `{,}`, lazy forms) and timed, not only by the spellings the scanner recognises* — source: B1.

## Pass 2 (re-review of the fix batch; compact)

**Verdict: needs changes** — B1 (the `{,N}` spelling) is closed, but the regex guard is still not time-bounded, and a measured admitted pair exceeds the 0.3 s bar.

### Stopping condition (stated before the broad probe)
A structural guard over `re._parser.parse` with refuse-by-default is **sound by construction against super-polynomial (exponential) blow-up only**: it forbids a repeat whose body holds a repeat or branch, refuses backrefs, lookarounds, atomic and possessive forms, and bounds branch counts. It is **not sound by construction for time**: polynomial cost is `n^wide × ∏(narrow spans+1) × ∏branch-multiplicities × per-step constant`, and caps on each factor do not bound the product unless the product itself is capped. So the guard counts as closed only if **both** hold: (1) the cap set is replaced by an explicit cost model (product above) with a refusal threshold, **or** evaluation runs under a hard time limit (killable worker/RE2); and (2) corner-biased search (cap-corner constructions plus a hill-climb over admitted patterns, `(?i)`/`(?s)` variants, 256-char texts of the overlapping alphabet) finds **no admitted pair above 50 ms**, with 0.3 s as the hard fail line. A pair budget alone cannot close it: its worst-case stall is `MAX_REGEX_PAIRS × worst admitted pair`, acceptable only if that is ≤ ~60 s, i.e. a worst pair ≤ ~12 ms (typical 13 ms is not the worst case).

### R1 (major, confirmed) — admitted patterns reach 0.4 s; the pair budget turns that into ~37 minutes
Corner constructions, `"a"*255+"!"`: `(?:a.|.a)`×4 `\w*\w*\w{0,8}\w{0,8}x` 147 ms; with `(?i)` 406 ms; `(?is)\S{0,8}[a-z]{2,9}\S{3,}a{,}(?:a|.)(?:.|a)(?:a|.)(?:.|a)x` 440 ms (n=64: 20 ms, n=128: 100 ms, n=255: 440 ms, so n^3-ish). All pass `regex_pattern_refusal` (returns `None`). The cause: unquantified alternation groups multiply a degree-2 polynomial, and `(?i)`/`(?s)` add a constant 3×. At 0.44 s × 5000 pairs the worst stall is ~37 min, not the ~1 min estimated. My own grammar fuzz (`fz.py`, nested groups, overlapping branches, backrefs, lookarounds, conditionals, inline flags, `{,n}`, atomic/possessive, unicode, 63,000 generated, ~19,000 admitted, 6 adversarial texts each, 0.3 s alarm) found **zero** admitted pair over 50 ms and **zero** guard exceptions — a random grammar mostly generates trivial admitted patterns, so it is weak evidence; the corner-biased generator (`cz.py`, 16,000 generated, ~4,000 admitted) found 15 admitted pairs over 0.3 s. Backrefs, lookarounds, conditionals, atomic and possessive were all refused; `{,n}`/`{,}` are now counted; zero parser exceptions.
Fix: gate on a cost estimate, e.g. refuse when `256^wide × ∏(span+1 of narrow) × ∏(branches) > ~2·10^6` (the cost of `\w*\w*\w{0,8}\w{0,8}x` is 9 ms), then re-run the corner search against it; or run `re` in a worker with a hard deadline, or RE2. Pair budget stays as a second layer, sized from the measured worst pair, not the typical one.

### Dispositions
- **B1** closed for the `{,N}`, `{,}`, `{N,}` spellings (parsed by the engine's own parser); the guard class stays open under R1.
- **M1** not closed: `\w*\w*\w*\w{0,300}x` is now refused, but see R1.
- **M2** fixed: 3000 services × 20 loopback sites 58 s → 0.01 s; 6000 × 40 sites 0.03 s (same workload as `p5.py`; the probe's mounts reach no built unit, so it measures lookup cost only).
- **m1** fixed: all five absolute/outside compose-path probes now bind nothing (`p2.py`).
- **m2** fixed as stated: a later bundle reusing id `C1` makes the legitimate binding `None` and `counts()["binding_duplicate_ids"] == 1`. Residual by design: the forged duplicate still suppresses the legitimate entry; acceptable as fail-closed provided the count is surfaced in the run output.
- **m3** fixed: `forbidden_host("0xa9fea9fe")` and `[::ffff:a9fe:a9fe]` are now True; ordinary and RFC1918 hosts stay False.
- **m4** fixed: all malformed `bindParts` shapes (`{}`, missing `hostHmac`, string parts, list host, int) are skipped without raising.
- **m5** (judged): the `mcp-json` format gate works (a compose-format `mcpServers` key no longer binds). Still open: a `.mcp.json` from another repo, or any path with `mcp-json` format (`node_modules/x/README.json`), still binds mentions system-wide. Recommended rule for the architect: bind only when the entry's file basename is `.mcp.json` and its repo is the mention's own repo or the manifest root; surface cross-repo matches as `possible` at most. Port squatting (`port-shared`) is documented behaviour, unchanged.

### Proposed REVIEW.md lines
Root `REVIEW.md`, "Always check": *A complexity guard over untrusted patterns is justified with a cost model (product of the bounded factors), and probed with a search biased to the cap corner (every cap at its limit, flag variants, overlapping alphabet), not only a random grammar; "no cap exceeded" is not "bounded time".* — source: R1.

## Pass 3 (final re-check; compact)

**Verdict: approve with suggestions.** R1 is closed against the Pass 2 stopping condition; no blocker or major remains.

### R1 — closed (measured bound, as the implementer states it)
Own corner-biased search (`hc.py`, not the implementer's generator): atoms with overlapping alphabets and non-foldable alternations, all quantifier spellings (`{,}`, `{,9}`, `{2,9}`, `{0,255}`, `{3,}`, lazy), 2–9 tokens, flags `""`, `(?i)`, `(?s)`, `(?is)`, `(?a)`, hill-climbing on a 40-member population mutated toward the slowest, 8 adversarial 256-char texts per pair (all-`a`, near-miss `a…ab!`, `aA`, mid-mismatch, `é`, trailing newline), 0.3 s alarm per pair, `ulimit -v 1500000`. **100,000 candidates, ~49,000 admitted, worst admitted pair 11.4 ms** (seed 2; typical worst per seed 1–4 ms), zero pairs over 50 ms, zero guard exceptions. Re-ran the Pass 2 killers: the 440 ms `(?is)\S{0,8}…` pattern and the 147/406 ms alternation patterns are now refused ("cost model over 100000"); `(a+){,}b` still refused. The bar (≤ 50 ms) holds with ~4× margin, and 800 pairs × 12 ms ≈ 10 s is acceptable. Caveat unchanged: a measured bound over a search, not a proof; hill-climbing in a ~10-token grammar can miss a shape, so keep the P2 item (killable worker or linear-time engine) on the backlog. The guard also refuses the 9 ms `\w*\w*\w{0,8}\w{0,8}x`: stricter than needed, harmless.

### m5 (architect ruling) — judged sound
Probe (`q5.py`, readable ids with `verify_ids` off): `.mcp.json` in the mention's own repo, nested `a/b/.mcp.json` and `.mcp.json.example` bind; the same entry from another repo, `README.json` and `.mcp.json.bak` bind nothing. Accepting the `.example` basename is acceptable: the extractor gives that file role `example`, which already weakens the result to `possible/example-config` (existing test `test_a_chain_through_an_example_file_is_possible_example_config`), and a primary definition wins over it. Nit: the basename match accepts a vendored `node_modules/evil/.mcp.json` in the mention's own repo (it bound `svc`); same trust domain as the repo, so acceptable, but excluding vendored directories where the extractor already does would be tidier.

### D1 (`verify_ids`) — severity minor
`_id_ok` requires the record's `repoId` to equal the header's and its id to equal `ids.config_entry_id(repo, path, keyPath)` / `launch_fact_id(repo, path, kind, ordinal)`. This proves the record claims its own location, not that the repo exists as claimed: the id has no content in it, so a forger who knows `(repo, path, keyPath)` (public) can make a record that passes and clashes, and both are dropped (`binding_duplicate_ids`) — suppression, not redirection. A forged bundle that declares another repo's `repoId` can also bind mentions in that repo through its own `.mcp.json` (m5 scoping trusts the header's `repoId`). Both need a repo-ownership trust anchor (the manifest signing which bundle may speak for which repo). Severity: minor while bundles come only from the trusted extractor pipeline; it becomes major the day a third-party or unsigned bundle is accepted. Keep the count visible in the run output.

### L1 — verified
`forbidden_host("0xa9fea9fe")`, `169.254.169.254` and `[::ffff:a9fe:a9fe]` are True, `metadata.google.internal` and `10.0.0.1` False; unchanged from Pass 2. The earlier `p2`/`p8`/`p9` probes (absolute paths, shadowing, port squatting) behave as in Pass 2.

### Proposed REVIEW.md lines
None beyond the Pass 2 proposal.
