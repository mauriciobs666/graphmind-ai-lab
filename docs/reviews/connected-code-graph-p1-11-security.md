# Config/Compose/.mcp.json/Kiro/Dockerfile extractors: security review (C-1001 P1-11)

> **Status:** active · **Owner:** `security-expert` · **Tracks:** C-1001 (P1-11)

Lens: code/app security (advisory, on request; not a gate). Reviewed: uncommitted `cpg/extractors/config/{__init__,formats,launch,extract}.py`,
against `docs/plans/connected-code-graph-p1-extraction.md` P1-11, `-bundle-spec3.md` §1.6/§1.7/§5.17, `-bundle-spec5.md` §9, the committed
`cpg/extractors/common/writer_rules.py` and `cpg/ccg/valuehash.py`, `cpg/ccg/bundle.py` gates. Root `REVIEW.md` applied (no `cpg/REVIEW.md`).
The analyst gate (`docs/reviews/connected-code-graph-p1-11.md`) was not touched.

CPG: not applicable — a new uncommitted Python module with no CPG built for it; traced by reading and by executing the real functions on throwaway inputs.

Method: ~60 tiny probes run with the real `extract_config`, foreground, `ulimit -v 1000000`, `timeout -s KILL 60`, `python3 -I`, scripts and inputs only
in the session scratchpad, made-up secrets, no network/FalkorDB/Docker. Everything below is **confirmed** (reproduced) unless marked theoretical.

**Verdict: needs changes.** Two blockers, both reproduced: a flag-first argv leaks a plain secret value, and a ~700-byte YAML merge-key file
exhausts memory (uncaught `MemoryError`). The W1-W4 / role / hash / keyPath machinery itself held up on every probe.

## Findings

### B1 (blocker, confirmed): a flag-first argv emits its flag's secret value in clear (Compose `command`, Dockerfile `CMD`, `.mcp.json` `command`)
`launch.py` `compose_drafts` (`command`/`entrypoint` loop), `dockerfile_drafts` (`ins in (st.entrypoint, st.cmd)`), `mcp_drafts` pass the element
list straight to `launch_argv_fields`, i.e. `bundle.gate_argv`, whose docstring says "argv[0] ... supplies no context". When the list does not start
with a program (a Compose `command:` that is the args of the image ENTRYPOINT; a Dockerfile `CMD ["--flag", ...]` beside an `ENTRYPOINT`; a CMD with
no program), argv[1] is gated with no context and a short value passes the token screen. Observed emitted records:

- Compose `command: ["--password","hunter2lower","--api-key","abcd1234xyz"]` -> `command: "--password"`, `args: ["hunter2lower", "--api-key", "<redacted>"...]`
  (`hunter2lower` in clear; also `--password hunter2lower` shell form; `-p hunter2lower`; uppercase `SECRETKEY` after `--api-key`).
- Dockerfile `ENTRYPOINT ["tool"]` + `CMD ["--password","hunter2lower",...]` -> `command: "--password"`, `args: ["hunter2lower", ...]`.
- `.mcp.json` `command: "--password", args: ["hunter2lower"]` likewise (less realistic, same path).

Contrast: the same flag after a real program (`python -m app --token ghp_...`) is withheld, so the gate is sound; the defect is feeding it a list without
an argv[0]. It is also a T-type confusion: the `command` field holds a flag, not a program.
Fix (route **coder**, test **tdd-engineer**; decision on the fact shape **architect**): when `elements[0]` is not a program (starts with `-`, or the
source is an args-only list: Compose `command` without a sibling `entrypoint`, Dockerfile `CMD` with an `ENTRYPOINT`), either gate `[<placeholder>, *elements]`
and emit the placeholder in `command` (a gate-accepted sentinel), or drop the fact; do not pass flag-first lists as-is. Add a planted-secret test: flag-first
Compose list, flag-first CMD, `-p value`, and `--flag value` where the value is lowercase/short (the existing planted secrets were all `--flag=value` or after a program).

### B2 (blocker, confirmed): YAML merge-key bomb, exponential memory, uncaught `MemoryError`
`formats.py` `parse_yaml.conv` (merge branch): `merged_items += [(mk, mv, mm or tgt) for ... in tgt.items]` copies the merged map's item list into every
merger, so a chain `m_i: &m_i {<<: [*m_{i-1}, *m_{i-1}], kN: v}` doubles per level. The alias-bomb defense (`Walk` 50k nodes) does not apply: the blow-up is in
`conv`, before walking. Measured (`ulimit -v 1000000`): 681 B -> 1.3 s/251 MB, 755 B -> 5.6 s/956 MB; a 15 KB, 200-level file -> `MemoryError` after 4.7 s at the 1 GB cap
(unbounded it would take the host); the `MemoryError` is not a `FormatError`, so it escapes `extract_config` and kills the whole unit, not just that file.
Plain alias bombs (billion laughs), 5000-deep nesting, 100k-deep JSON/TOML, 512 KiB scalars, 60k-entry flow maps, `!!python/object` tags (composed to plain scalars,
never constructed) all failed safe: bounded time (<2.4 s), <130 MB, fail-closed or truncated with counters, messages carry reason codes only.
Fix (**coder**): bound `conv` work: count nodes/merged items created in `conv` (shared counter against `MAX_NODES`) and raise `FormatError("nodes")`; flatten merges lazily
(keep `merged_from` references, expand in `Walk`, which is already bounded) rather than copying lists; additionally catch `MemoryError` at the per-file parse in `extract_config` as
a `skipped` reason so one file cannot abort the unit. Test: the 22-level chain above must yield `skipped: nodes` within a second at the stated `ulimit`.

### M1 (major, confirmed): `declared_services` (W2) is attacker-widenable by any compose-named file, including `example` role files and `container_name`
`extract.py` unions `compose_declared_services` of every Compose file in `files` (also `docker-compose.example.yml`, a file that does not run) plus every `container_name` value.
Probe: an example compose with `container_name: metadata` made `api_host: metadata`, `listen: metadata:80` and `endpoint: http://metadata/latest` emit as `bindValue`.
The single-label-host protection is therefore only "a name some compose file in this repo mentions". Same-repo is the spec's bar, so this is a hardening gap rather than a
violation: count only `primary`-role Compose files (`file_role`), and only keys of `services` (not free-text `container_name`) unless the spec says otherwise. Route **architect** (spec intent of "declared") then **coder**.

### M2 (major, confirmed; linker-side consequence): launchFact semantics a later linker must not trust
- `mount.mountPath` is the build context as written: `../../../../etc` and `/var/run` are emitted (the screen allows `.`/`..` segments and absolute paths). A P1-17 reader that joins it to the repo root escapes the repo.
- `service` is a shared namespace: a `.mcp.json` server named `a` (origin `explicit`) and a Compose service `a` (origin `compose`) both yield `command` facts with `service: "a"`; a hostile repo can name an MCP server after a Compose service.
- `bindValue` is emitted for any role-named key with a clean host/URL, including external or link-local targets (`url: http://169.254.169.254/latest` emitted). Correct under the spec (the bundle records, the linker decides), but nothing in the bundle says a value is a *client target* not a listen binding.
Fix: **architect** to state in the P1-17 plan that linking never dials or reads a path from these facts, must normalize and confine `mountPath` to the repo, and keys service identity on `(origin, path, service)`; **coder** may drop `..`/absolute contexts at the extractor (cheap, and consistent with `_safe_context`'s stated purpose).

### m1 (minor, confirmed): short secrets pass the token screen in non-argv string fields
`service` (MCP server name, Compose service) and `mountPath` are gated only by `screen_fails` (T1-T6, length/entropy heuristics), not the name rule: `srv/SECRETNAME-hunter2` was emitted as `service`
and `./secret-dir/sk-live-0123456789` as `mountPath`. Likewise a dotted host label such as `host2_host: sk-live-abc.example.com` is emitted as a `bindValue`. This is the spec's accepted heuristic residue; a human-chosen
server name carrying a short password is plausible. Suggest applying `bundle.name_fails` to `service`/`mountPath` segments too (cheap: withhold, W4). **coder**.

### m2 (minor, confirmed): partial emission on truncation is counted but not surfaced per file role
Caps fire (`truncatedFiles` counter, 10 000 entries / 50 k nodes): entries past the cap are silently absent; an attacker can pad a file so a real binding falls after the cap. Not a secret leak; note for the linker's confidence. No change required beyond the counter being consumed.

### n1 (nit): U+2028/U+0085 are not quoted
`render_key_path` quotes only `< 0x20`/`0x7F`; a key `x y_host` renders bare. Check the schema's keyPath pattern accepts it (not run: `bundle.validate` needs a header record; no schema file was exercised).

## Probed and held (evidence of what is solid)
1. **Credential-in-URL (>= 30 cases)**: `https://user:pw@h`, `?token=`, `?api_key=`, DSNs (`postgres://`, `redis://:pw@`, `mongodb://u:p@`, `ftp://`), `%40` userinfo, `host:pw@evil`, fragment, path tokens (`/v1/<32 hex>`, `/webhook/ghp_...`, `/sk-...`), 64-hex-prefixed FQDN, a bare secret in `host`/`bind`/`addr`/`hostname`, under role keys `url/uri/base/endpoint/server_host/redis_url/mongodb_url/host/listen/addr`: **all withheld** (value grammar + W1 + W2); only literal `host:port`, plain URLs and IPs emitted. Same in Compose list-form `environment`, Dockerfile `ENV`/`ARG` (incl. legacy `ENV K V`), `.mcp.json` `url`/`env`/`headers`, `${VAR:-https://u:pw@h}` env-defaults (default withheld, hash kept).
2. **Argv after a real program**: `--token=x`, `--token <v>`, `--header "Authorization: Bearer ..."`, `--db-url=postgres://u:pw@..`, `curl -u admin:pw`, `$(..)`/backtick/`;` shell metacharacters, `sh -c "<script>"`, a 5000-char arg (whole fact dropped at `MAX_ARG_CHARS`, 256-arg cap): withheld/redacted or dropped. Shell form is only `shlex.split`, never executed; no `$` survives the gate. Exception: B1.
3. **keyPath forgery**: keys `secret.host`, `api_key.host`, `token"].host`, `token[0].host`, `""`, `[`, `a]`, 600-char, control char, `k.a` vs nested `k`/`a`: quoted `["..."]` forms stay distinct from nested paths (no collision), and W1 sees the quoted segment, so `["secret.host"]` is withheld even though the role-token split sees `host` (E3 only admits). Over-long segments drop the entry.
4. **Parsers** (measured): YAML alias bomb (30 levels x 9) stops at 50k nodes in 0.13 s/31 MB; `!!python/*` tags composed, not constructed; duplicate keys last-wins; deep YAML/JSON/TOML -> `FormatError` reason codes; JSON `NaN`, 400-digit number handled; 500 KiB TOML array 1.2 s/126 MB; oversize files skipped before parsing; lone surrogate drops the entry (`unencodableEntries`), no hash; non-UTF-8 -> `SourceRefused` naming the path only. Worst measured legitimate 512 KiB YAML: 2.3 s / 106 MB.
5. **Hash path**: `value_hash_fields(key, rid, text)` with the entry's own 32-hex id as the domain, `entry_hmac` NFC/UTF-8 (read, not re-derived by test beyond the author's spy tests); no logging/`print` in the package (grep clean); report holds counters and path/reason codes only.

## Open questions
- B1: the intended `command` fact shape for an args-only list (`architect`): placeholder program, or no fact?
- M1: is a service in an `example`-role compose file "declared" per W2? Spec says "a declared Compose service of the same repository"; silent on role.

## Route
- **coder + tdd-engineer**: B1 (gate with a program placeholder / drop; new tests), B2 (bounded conv, lazy merge, `MemoryError` per-file catch; new test).
- **architect**: B1 shape, M1 declared-set rule, M2 linker contract for P1-17 (no dial, confine `mountPath`, service identity).
- Proposed `REVIEW.md` lines: none (a sweep over spec3 §1.6.1 callers for "list passed without argv[0]" is a one-off for B1).

## Pass 2

Re-checked live (same real-function probes as pass 1, plus new ones; `ulimit -v 1000000`, `timeout -s KILL 60`, made-up secrets) against the Round 2-3 code and the uncommitted `writer_rules.py`/`errors.py` follow-ups.

**Verdict: approve with suggestions.** Both blockers are closed; no new blocker. Residuals are the DNS-name and relative-path items below.

- **B1: fixed.** The pass-1 cases (`--password hunter2lower`, `-p value`, `--api-key SECRETKEY`, Dockerfile CMD beside ENTRYPOINT, `.mcp.json` flag-first `command`) now give `command: "<redacted>"` with the flag in `args` and the value redacted. New cases, none leaking (a scan of every launchFact for the planted string was empty; no `command` starts with `-`): attached `-p<secret>`, `--password=<secret>` after a flag-first element, `-H 'Authorization: Bearer …'`, `-H 'X-Api-Key: …'`, shell form `-p<secret> --token <s> -u admin:<s>`, CMD `-p<s>` beside `tini`. A real program (`x -H 'Authorization…'`) is unchanged and still redacted.
- **B2: fixed.** The 755-byte 22-level chain and the 7.8 KB 200-level chain, and a 4-way-merge 400-level chain, give `skipped: nodes` in 0.00-0.07 s (pass 1: 5.6 s / 956 MB and `MemoryError`). A legitimate merge still resolves (`svc.host` inherits, `svc.port` overridden).
- **M1: fixed.** An example-role compose file declares nothing (`declared_services` empty; `api_host: metadata` withheld); the same file as primary declares `a, b, c, metadata` and the host emits, as the spec intends. `container_name` still counts in a primary file (accepted: same-repo spec bar).
- **M2: partly fixed.** Absolute `mountPath` (`/var/run`) is withheld. A relative escape `../../../../etc` is **still emitted** (minor, linker-side: P1-17 must normalize and confine; suggest the extractor also drop `..` segments). Service-namespace sharing (`.mcp.json` `a` vs Compose `a`, both `service: "a"`) is unchanged: still a P1-17/architect contract item. Link-local: **fixed**. 38 address forms probed as `*_host` values: withheld for dotted, decimal (`2852039166`), hex (`0xa9fea9fe`, `0xa9.0xfe.0xa9.0xfe`), octal (`0251.0376.0251.0376`), short (`169.254.1`), trailing dot, `[::ffff:169.254.169.254]:80`, `::ffff:a9fe:a9fe`, `[::a9fe:a9fe]`, `[64:ff9b::a9fe:a9fe]:80`, `[0:0:0:0:0:ffff:a9fe:a9fe]`, `[fd00:ec2::254]:80`, `[fe80::1]`, `100.100.100.200`, and `instance-data`. Emitted by design: `localhost`, `127.0.0.1:8200`, `[::1]`, `10.0.0.5:80`, `[2001:db8::1]:80`. **Residual (confirmed, theoretical impact):** DNS names that *resolve* to link-local (`metadata.google.internal`, `169.254.169.254.nip.io`, `nip.example.com`) are emitted; an offline extractor cannot resolve them, so the P1-17 linker contract (no dialing from bundle values) is the only control. Pass-1 credential probes (p1/p2/p3 sets, 30+) re-run: unchanged, all still withheld.
- **m1: not fixed.** `srv/SECRETNAME-hunter2` is still emitted as `service` and `./secret-dir/sk-live-0123456789` as `mountPath` (`launch.py` still uses `screen_fails` only; no `name_fails`). Also `host2_host: sk-live-abc.example.com` still emits. Spec-accepted heuristic residue; unchanged suggestion.
- **m2: unchanged**, counter-only (`truncatedFiles`; new `memoryErrors` counter added, 0 here).
- **n1: not fixed.** `x y_host` is still rendered bare; Round 2 added NFC + strip of segments, which does not touch U+2028/U+0085. Still unverified against the schema pattern.
- **New, none found:** no regression in the keyPath-forgery probes (quoted `["secret.host"]` etc. still withheld); NFC/strip did not create a collision in the probed set (not exhaustively tested for NFC-equivalent distinct keys, which now share one keyPath and collapse last-wins: theoretical, low).
