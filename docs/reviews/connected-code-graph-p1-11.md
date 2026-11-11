# C-1001 P1: P1-11 config, Compose, `.mcp.json`, Kiro and Dockerfile extractors — gate review

> **Status:** active · **Owner:** `analyst` · **Tracks:** C-1001 (P1)

## Scope & verdict

Reviewed (uncommitted, all new): `cpg/extractors/config/{__init__,formats,launch,extract}.py`, `cpg/tests/test_config_extractor.py`,
`cpg/tests/data/config/`, the delegate's `cpg/.ccg-work/p1-11/REPORT.md`, against `docs/plans/connected-code-graph-p1-extraction.md`
§ P1-11, `-bundle-spec3.md` §1.6/§1.7/§5.17, `-bundle-spec5.md`, `-p1-linking.md` P1-17, the fixtures' AC6-EX rows, and the uncommitted
`writer_rules.value_hash_fields`. The security advisory (`docs/reviews/connected-code-graph-p1-11-security.md`) was not read or edited.

**The plan's P1-11 section was amended at 00:01 on 2026-11-11, after the code (23:48) and REPORT (23:55) were written** (a "Hostile-input
rules" block: args-only argv, primary-role declared services, absolute `mount`, link-local hosts, bounded parsing B2). I judged the code
against the amended text, since that is what the gate's "Done" line now requires; findings B1, B2, M2 and M3 are the amendment's rules, each
**verified to be unmet** below. The delegate built to the earlier text and is not at fault for them.

**Verdict: needs changes** (two blockers). Of the nine conformance and decision items you listed, the extractor is sound; the failures are
on hostile input.

CPG: considered, not relevant — new, uncommitted component code read directly; no graph covers it.

**Executed:** `tests/test_config_extractor.py` (185 passed), with `test_writer_rules.py` and `test_valuehash.py` (512 passed, including the
repo AST tripwire applied to `extractors/config/*.py`); 11 mutants (below); about 30 hostile and planted inputs through `extract_config`
(`ulimit -v 1000000`, `timeout -s KILL 60`, `python -I`); the fixtures' config files through the extractor. **Read only:** the Dockerfile
`ONBUILD`/heredoc/`SHELL` behaviour, TOML line scan, `.env` multi-line. No Docker, network, store or key outside test keys.

## Findings

### B1 — blocker — a flag-first argv leaks its secret value in clear (verified)
`launch.py:73-76` / `extract.py:186` pass the raw list to `launch_argv_fields`, so `gate_argv` treats element 0 as the program and gives
`--password` no successor context. Executed, emitted records:
`compose command: ['--password','hunter2']` gives `command:"--password", args:["hunter2"]`; likewise `['-p','hunter2lower']`,
`['--api-key','ABCDEF1234']`, a `.mcp.json` `command: "--password"`, a Dockerfile `CMD ["--password","hunter2lower"]` beside an `ENTRYPOINT`,
and shell-form `CMD --password hunter2lower`. The value reaches the bundle unhashed, against "secrets never stored". Every existing
planted-secret test (`test_a_secret_flag_value_in_an_argv_is_redacted_by_the_gate`) puts a program first, so none can fail on this.
**Fix (checked at the gate):** `bundle.gate_argv([bundle.REDACTED, *elements])` returns `['<redacted>','--password','<redacted>']` for all
three shapes above and leaves `['--port','8000','--host','x']` intact. Apply it where the plan says: first element starting with `-`, or
a `CMD` beside an `ENTRYPOINT`; add the plan's six planted rows plus the assertion "no emitted `command` starts with `-`". Whether the
wrapper belongs in the extractor or in `writer_rules.launch_argv_fields` is the tripwire-constrained design call: put it in
`writer_rules` (the one owner of `command`/`args`).

### B2 — blocker — a 2 KB merge-key bomb exhausts memory; `MemoryError` escapes the unit (verified)
`formats.py:316-323` builds `merged_items += [... for tgt.items]`, an eager per-merger copy, before `Walk` and before any node budget. Executed:
a 419-byte file (`a_i: {<<: [*a_{i-1} x9]}`, six levels) peaks at 399 MB and 2.2 s; the 2,007-byte 29-level file raises `MemoryError` out of
`extract_config` at `ulimit -v 1000000` after 4.8 s (unlimited, it takes the host). The 512 KiB cap is bypassed (the file is tiny), `MAX_NODES`
counts only in `Walk`, and nothing catches `MemoryError` (`grep MemoryError cpg/extractors/config/` is empty). The plan's B2 requires the node
account in `conv`, lazy merge expansion, and a per-file `MemoryError` catch. The delegate's alias-bomb tests cover aliases only, not merges.
**Fix (checked):** after the `merged_items +=` line, `if len(merged_items) > MAX_NODES: raise FormatError("nodes")` makes the same file yield
`skipped: nodes` in 0.02 s, 31 MB, and keeps the 185 tests green; that is the minimum. Do the plan's full version (shared counter, lazy
expansion, `MemoryError` catch as a `skipped` reason) and add the plan's 22-level test, asserting the reason code, not wall time.

### M1 — major — a non-NFC or untrimmed key invalidates the whole unit's bundle (verified)
`render_key_path` (`formats.py:121`) emits segments as written; spec3 §8.1 requires `keyPath` NFC and stripped. Executed through
`bundle.validate`: JSON keys `"café"`, `" lead"`, `"trail "`, `" nb"` (and the YAML/Compose forms `' a': 1`, `'B ': 1`) each fail
with `keyPath: string is not NFC` / `identity field must be stripped`. One odd key in a hostile or merely sloppy file fails every record of
the unit at validation. **Fix (checked):** NFC-normalise and strip each named segment before quoting; with that, all four validate
(`' '` becomes the quoted `[""]`). Test per axis: NFD key, leading, trailing, NBSP, whitespace-only, in a JSON key, a YAML key and an INI key.

### M2 — major — an example Compose file still declares services for W2 (verified)
`extract.py:99-100` unions `compose_declared_services` over every Compose file. Executed: `docker-compose.example.yml` with
`container_name: metadata`, plus `api_host: metadata` in another file, binds `{'api_host': 'metadata'}`. The plan now limits W2 to
`primary`-role files (and the plan's test is this input). The same set feeds `default_fields`. `extra_declared_services` carries the same
rule: the caller must pass primary-only names (say so in the docstring and the P1-44 text, see m3).

### M3 — major — absolute `mountPath` and link-local hosts are not withheld (verified)
Executed: `build: /var/run` and `context: /etc` emit `mount` facts (`_SAFE_CONTEXT` admits a leading `/`); `bindValue` is emitted for
`http://169.254.169.254/latest`, `http://[fe80::1]/x`, `fd00:ec2::254` and `169.254.0.5`. The plan requires both withheld and P1-17's trust
rule (d) says "P1-11 already withholds it", which is false today. The link-local rule must live in `writer_rules.bind_fields`
(`bindValue` is writer-only; the tripwire forbids the extractor writing it) and needs `test_writer_rules.py` rows, so it touches the
committed P1-07 file: `teco` to route (coder or tdd-engineer with that file in scope).

### m1 — minor — the alias-bomb test fails by hanging, not by assertion (verified)
Mutant s30 (`Walk` node budget off) ran to the 30 s KILL (rc 137) with no assertion: the `< 20` s check never runs when the loop never returns,
and the suite has no `pytest-timeout`. A real defect hangs CI. Make the test assert a structural bound: run `Walk` with a small `max_nodes`
and assert `truncated` plus the leaf count, or run the bomb in a subprocess with a timeout.

### m2 — minor — the context-segment screen is untested (verified)
Mutant o3 (drop `not any(bundle.screen_fails(seg) ...)` in `_safe_context`) survives all 185 tests. The code is right
(`_safe_context('x7k2-q9w4-m1p8-z3v6/app')` is `False`); the bad-context table has no token-shaped segment. Add one row.

### m3 — minor — the call convention is not in P1-44 (verified by grep)
`grep -n 'extract_config\|extra_declared_services' docs/plans/connected-code-graph-p1*.md` finds only the P1-11 text. W2's set is
per-repository: P1-44 must call `extract_config` once per repo (all its Compose files in one call) or pass `extra_declared_services`,
primary-role only. Add that to P1-44's text. `ConfigResult.roles` is empty for oversize/unparseable files, so P1-44 must use `file_role`
for their `File` records.

### m4 — minor — published and container ports are indistinguishable (design call, inferred)
`8200:8000` emits two `port` facts of one service; no field says which is host-side. A consumer `localhost:8000` can then match a service
that only exposes 8000 inside the network. The fixtures do not exercise it (`8000:8000`, `8300:8300`), and spec3 §1.6 has no field. Pin the
reading in P1-17 or the spec, or accept the ambiguity as `possible`: `architect` to rule.

### Nits
Oversize files are UTF-8-decoded before the size check (`extract.py:86` vs `:90`); `truncatedFiles` counts one file twice when two launch kinds hit
`MAX_FACTS` (a 100,000-`EXPOSE` Dockerfile reported 2). `render_key_path` also quotes empty and control-character segments, which spec3 §5.17's
sentence ("quotes only `.` or `[`") does not say; the schema pattern forces it, so name it at the feature's spec update.

## The delegate's decisions

All five are **in the amended plan or its P1-17 text, or reversible without a spec change**; none is invented against the spec.
- Compose `build.context`/`build: ./x` as a `mount` fact with `service` and `mountPath` as written: ratified by P1-17's trust rule (b)
  ("a `mountPath` is joined to its compose file's directory") and (c) (identity `(origin, path, service)`, so a `.mcp.json` server and a
  Compose service of one name stay apart; the extractor's `origin` is `explicit` vs `compose`). No dedicated field is needed.
- `.mcp.json` command + args as a `command` fact with `service` = server name: same (c). Spec3's "the Compose service" wording stretches; say so at the spec update.
- Dockerfile final stage, ARGs before the first FROM, last CMD/ENTRYPOINT, lower-case `entrypoint[i]`/`command[i]`: matches the plan's derivation row; `ONBUILD` is correctly not read as `CMD`. A Compose `build.target` selecting a non-final stage is ignored (accepted, document it).
- `${VAR:-x}` in any non-alias scalar: plan names env defaults; the default is kept only for a role-named variable (tested, incl. `DB_PASSWORD_URL:-hunter2` and `S_HOST:-ghp_…` withheld).
- keyPath quoting: see Nits.

**P1-17 derivability (executed over `cpg/fixtures/system1`):** AC6-EX-1 (`.mcp.example.json`: role `example`, `command` fact `service:"fx-svc"`
plus `mcpServers.fx-svc.command` entries), AC6-EX-2 (`services.web.environment.VITE_REPORTS_BASE`, `example:true`), AC6-EX-4
(`services.svc.environment.FX_PEER_BASE` and a `mount` fact `service:"svc", mountPath:".."` from `compose.example.yaml`), AC6-EX-3
(`.env.example` `FX_DB_URL`, `example:true`) are all derivable; `../../fx-job` and `..` come out as written.

## Hostile-input probes (item 3)

Each cap was driven with a file at or past it; all failed closed with a reason code and no text, except B2. **512 KiB**: oversize `skipped`.
**Depth 32**: deep block/flow YAML, 260,000-deep JSON, 10,000-deep TOML, a recursive anchor `a: &a [*a]` each `skipped: depth|yaml|toml` in under
0.1 s. **50k nodes**: a 1.5 KB 9-ary alias bomb returns `truncated` in 0.17 s, 31 MB. **10k entries**: truncated for JSON/YAML/TOML/INI/.env/Dockerfile.
`yaml.SafeLoader` subclass composes, never constructs (`!!python/object` marker file absent, tested). JSON is strict (NaN, trailing comma,
control characters refused). Secrets: errors and the report carry paths and codes only (non-UTF-8 message names the file; planted-secret
blob test passes); a `password:`/token value is hashed only; a token-shaped key stays in `keyPath` by spec design (rejected-option note, spec3 §6).
**Worst case at the cap (residual risk benchmarked):** 512 KiB flow-map YAML 3.6 s / 135 MB, block sequence 3.0 s / 109 MB, wide alias fan-out 2.2 s / 76 MB,
520 KB TOML/INI/.env under 0.7 s. Acceptable per file; the repo-level budget is P1-44's. Dockerfile heredoc/`SHELL`, `.env` multi-line and
TOML line best-effort are low: a heredoc body line starting `CMD` is read as an instruction, but only in the repo's own content.

## Mutants (item 5)

Fresh copies under `cpg/.ccg-work/gate-p1-11/` (deleted; `mut.sh` kept), foreground, `ulimit -v 1500000`, KILL 30-100 s. **Sampled from the
delegate's 52:** m01 role on first segment: killed (golden `.mcp.json`); m07 hash domain = file id: killed (independent HMAC); m13 declared set
empty: killed; m31 depth cap off: killed (alias chain); m41 merge order: killed. **My own:** o1 swap `env_var`/`default_text` in `default_fields`:
killed; o4 drop `container_name` from declared: killed; o5 silent hash-domain swap to a valid 32-hex `file_id`: killed; o3 context screen
removed: **survived (m2)**; o2 anchor path loses the child key: survives, argued equivalent (the dropped segment equals the alias's own last
segment, which W3 judges anyway; inferred, not proven). m30 is a hang, as the delegate said, and the test is real but weak (m1).

## What's solid
One derivation row per format, the enum read from spec3's §5.17 row, and a fixture per format; `bindValue`, `default`, `defaultHash`,
`command`/`args` and `valueHash` all come from `writer_rules` (spy tests plus the AST tripwire, run); the independent-HMAC check and
`audit_records(..., texts=)` over the fixtures; W3 alias/merge anchor judging; dense ordinals; D12 refusal and unencodable-drop; the
report's value-free shape; the `File.role` parity test with the normalizer; the compose `environment` list split.

## Open questions
1. `teco`: where does the link-local rule live (M3), and does P1-07's owner re-gate `writer_rules`?
2. `architect`: pin the published-versus-container port reading (m4), and the `service` wording for MCP servers at the spec update.

### Proposed REVIEW.md lines
Target: root `REVIEW.md`, "Always check" (source: B2, this review). *A parser of untrusted structured input is probed on every
expansion axis its format has (alias, merge key, anchor, include, nested document), each with a file under 2 KB at the memory cap; a cap on
one axis, or on file size, is a bound, not proof.*

## Pass 2

**Verdict: approve with suggestions** (no blocker; one new major to fold into the same fix pass). **CPG:** considered, not relevant — same files, read directly.
**Executed:** `test_config_extractor.py` + `test_writer_rules.py` + `test_valuehash.py` (679 passed) and `test_normalize_core.py` + `test_normalize_assemble.py` (277 passed); my planted probes (re-run of Pass 1, plus about 30 URL forms), the fixtures' config files, 4 mutants in fresh copies (deleted). **Read only:** the diffs below, `MemoryError` catch sites (`extract.py:97,111`), the m1/m2 tests.

| Finding | Disposition and evidence |
|---|---|
| B1 | **Fixed.** Re-ran: `['--password','hunter2']`, `['-p','hunter2lower']`, `['--api-key','SECRETKEY']`, string forms, `.mcp.json` flag-first command, Dockerfile `CMD` beside `ENTRYPOINT` (flag-first and program-first), shell form, flag-first `ENTRYPOINT`, Compose `command` beside `entrypoint`: no secret in any record, no `command` starts with `-`; e.g. `command:"<redacted>", args:["--api-key","-<redacted>"]`. `launch_argv_fields(..., program_known=)` gates `[REDACTED,*argv]`; `audit_records` now flags a flag-valued `command`. |
| B2 | **Fixed.** The 419-byte and 2,007-byte merge bombs: `skipped: nodes` in 0.00 s / 0.01 s, 26 MB, at `ulimit -v 1000000`; a recursive anchor `skipped: depth`; the alias bomb still `truncated` in 0.13 s. `MemoryError` is caught per file at parse and at record building (`extract.py:97,111`, read), counter `memoryErrors`. Side effect, accepted by the plan's bound: a 520 KB flow-map YAML (3.1 s) is now `skipped: nodes` where it was truncated at 10k entries; a legitimate file over 50k nodes loses all its entries. |
| M1 | **Fixed for `keyPath`.** NFD, leading/trailing space, NBSP and whitespace-only keys now validate in JSON, YAML, INI and TOML (`['café','ok']`, `['[""]','ok']`). **New N1 below.** |
| M2 | **Fixed.** Example `docker-compose.example.yml` with `container_name: metadata`: `declared_services == []`, `api_host: metadata` not bound; a primary compose still declares (`{'a_host':'api'}`). |
| M3 | **Fixed.** `build: /var/run` and `context: /etc` emit no `mount`; `./ok` does. 27 link-local forms withheld: dotted, with port/userinfo/trailing dot, bracketed and bare `fe80::/10`, `fec0::/10`, `fd00:ec2::254`, `100.100.100.200`, IPv4-mapped (`::ffff:a9fe:a9fe`), compatible (`::a9fe:a9fe`), NAT64, decimal/hex/octal/short IPv4; env defaults `${A_HOST:-169.254.169.254}` and `${B_URL:-http://0xa9fea9fe/}` withheld; `10.0.0.5` and `localhost` still bind. With the rule off (mutant p3) the dotted, IPv6, mapped, compatible, NAT64 and `100.100.100.200` forms bind; the decimal/hex/octal forms are withheld by the §1.7.2 grammar either way, so `_inet_aton` is defence in depth, pinned by direct unit tests. |
| m1 | **Fixed.** `test_config_extractor.py:615` asserts `Walk.visited == MAX_NODES + 1` (read; coder's mutant "walk budget off" now fails by assertion; I did not re-mutate it). |
| m2 | **Fixed.** The context table now has token-shaped segments (`./x7k2q9w4…`, `sk-…/svc`; read, `:901`); my old mutant o3 not re-run. |
| m3 | **Fixed.** `-p1-linking.md:217` carries the once-per-repo call convention. |
| m4 | **Open, routed to `architect`** (unchanged; not a gate blocker). |
| Nits | Oversize-decode and double-count not re-checked; the §5.17 quoting sentence stays a spec-update note. |

### N1 — major — `service` is not NFC-normalised, so one server or service name invalidates the unit (verified)
Same class as M1, now exposed because `keyPath` no longer fails first. `_service_name` (`launch.py:68`) returns the name as written; an `.mcp.json`
server `"café"` or a Compose service `café` fails `bundle.validate` with `service: string is not NFC (§8.1)` (a spaced name, an NFD
`args` element, command, env default and mount path all validate). **Fix:** `unicodedata.normalize("NFC", name)` (and strip) in `_service_name`, with the
W2 declared set built from the same normalised names; add NFD rows for `mcpServers.<n>` and `services.<n>`.

### Same-revision interactions
- **`program_known` versus detection:** consistent. The Dockerfile rule (`CMD` beside `ENTRYPOINT`) and the Compose rule (`command` beside `entrypoint`) pass `False`; a flag-first list is caught by the writer's `argv[0].startswith("-")`. Mutant p4 (Compose rule keyed on `entrypoint` instead of `command`) was killed. Trade-off, accepted: a program-first `CMD ["uvicorn",…]` beside `ENTRYPOINT tini` now has `command:"<redacted>"` and `uvicorn` in `args`; `--port 8300` stays readable. A `CMD` that inherits an `ENTRYPOINT` from a base image is not seen, and is covered only by the flag-first detection.
- **Link-local versus W2:** the address rule runs first in `_w2_ok`, so a declared service named `169.254.169.254` or `extra_declared_services=["169.254.169.254"]` still binds nothing (probed).
- **`SourceRefused` identity:** `errors.SourceRefused is core.SourceRefused is extract.SourceRefused` is `True`; a non-UTF-8 file raised from `extract_config` is caught by the `normalize.core` import; the normalize suites pass (277); the shell extractor imports the same class.
- **Golden change (`CMD` beside `tini` now redacted):** the fixtures' AC6-EX rows still derive. Re-run over `cpg/fixtures/system1`: EX-1 `.mcp.example.json` (role `example`, `command` fact `service:"fx-svc"`, `scripts/run-mcp.sh --port 8200` unchanged because the program is first), EX-2 and EX-4 environment entries with `example:true` and EX-4's `mount` fact `service:"svc", mountPath:".."`, EX-3 `.env.example`; the fixtures' Dockerfiles have no `ENTRYPOINT`, so their `uvicorn` commands are unchanged.

### Mutants (fresh copies under `cpg/.ccg-work/gate-p1-11/`, deleted)
p4 `program_known` keyed on the wrong Compose key: killed (golden compose). p3 link-local rule off: killed. p1 `_inet_aton` octal read as decimal, and p2 the short-form byte fill `256**(5-n)` changed to `256**(4-n)` (a silent argument): both survived the config test file the harness runs, and were **killed by `test_writer_rules.py`** (13 and 15 failures), so the gap is in the harness, not the suite. No test reaches them end to end, because the grammar already withholds those forms.

### Routing
`coder`/`tdd-engineer`: N1 (small), then close. P1-17 and P1-44 unblocked on P1-11. `architect`: m4 and the `service` wording at the spec update.
