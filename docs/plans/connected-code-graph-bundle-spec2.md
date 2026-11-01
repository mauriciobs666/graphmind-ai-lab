# Connected code graph — fact-bundle spec, pre-freeze amendments (v1 candidate)

> **Status:** superseded · **Owner:** `architect` · **Tracks:** C-1001 (M10) · **Extends:** `docs/plans/connected-code-graph-bundle-spec.md` · **Version:** 1.3 · **Superseded by:** `docs/plans/connected-code-graph-bundle-spec3.md` · **Reviews:** `docs/reviews/connected-code-graph-p0-impl.md`

*2026-10-29 (v1.0): S58 advisory dispositions, S10 gate rulings, S06 eval-protocol Q2; written before S15.*
*2026-10-30 (v1.1): re-gate round. Analyst BS-M1/M2, BS-m1–m9 and nits; graph-dba G-1–G-4, S-1, S-2; security Pass 2 P2-1–P2-8. One argv rule now closes BS-M1, BS-M2 and P2-1 together. T6 is adopted before the freeze. Two recommendations are declined in part: "no `@` after the first character" and "W5 writer-only". Dispositions are in §5.*
*2026-10-30 (v1.2): analyst Pass 2 (BS-P2-M1, BS-P2-m1–m3) and security Pass 3 (P3-1–P3-4).*
- *The context-loss major is fixed once: a redacted context supplier becomes a forcing sentinel (§1.6.1).*
- *AV-21 is kept by a closed single-dash allow-list, not by P3-1(a)'s plain-sentinel rule, which a validator could not check.*
- *`bindValue` and address-shaped argv values are screened per label.*
- *The §1.6.3 probe was run: 17,280 cells, 0 values in clear. Dispositions are in §5.1.*

*2026-10-30 (v1.3): analyst Pass 3 (BS-P3-M1, BS-P3-m1–m3).*
- *(v8) closes `NAME:secret`; the address exemption keeps `falkordb:6379`, `localhost:8000` and `keycloak:8443`. The separator axis is normative in §1.6.3.*
- *AV-21 corrected: my v1.2 check compared outputs by eye, not with the file.*
- *Allow-list widened to Go role and common JVM flags, each shown to pass only what its `--` form passes.*
- *(v7) gains `account`, `principal` and `identity`. The IPv6-hex residual and the internal-address over-redaction class are named.*
- *Observed: 17,280 + 6,480 cells, 0 in clear; transplant 675, bare-positional only; AV 68/68 and BV 28/28 parsed from this file. Dispositions are in §5.2.*

## 0. What this is, and how to read it

**Read with `docs/plans/connected-code-graph-bundle-spec.md` v1.3 ("the spec").** The spec controls
everywhere except where an amendment in §1 names the clause it replaces; there, §1 controls. Section
numbers below (§5.1, §5.18, …) are the spec's.

**Bundle schema v1 is the spec as amended here.** `bundleSchemaVersion` stays `"1.0"`. These
amendments change the v1 candidate before it freezes (spec §1), so they are not a version bump under
spec §8.6. S15 freezes the spec plus this document.

**Required conformance item: `-spec3`.** This item **replaces spec §1 bullet 3's successor name**
(`-spec2`, written by S16). The consolidated contract is
`docs/plans/connected-code-graph-bundle-spec3.md`:
- it carries `Supersedes:` both the spec and this document;
- it states the spec, as amended here and by any SL-7 delta, in one text;
- S16 writes it if S15 forces a change; otherwise `architect` writes it before P0 exit.

**P0 exit requires `-spec3` to exist and to be gated**, and `teco` checks this at P0 exit (§4.2). The
reason is that the spec's `Extended by:` pointer marks it authoritative, while about a dozen of its
clauses are not. Until `-spec3`, the contract is the two documents read together. Consolidating
them is the step that restores conformance, so it is a required item and not housekeeping (analyst
open question 1).

**Why a successor, and why `Extends:`.**
- **Root `AGENTS.md` collision rule 5 requires a successor.** The spec has been gated (S03 Passes
  1–3) and executed against (S09's `ids.py`; S10's schema, `bundle.py` and tests). So it stays intact.
- **Spec §8.6 is not a document-revision mechanism.** It versions `bundleSchemaVersion`, and its own
  "who writes a change" bullet defers to rule 5. And nothing here bumps the version (above).
- **`Extends:`, not `Supersedes:`, and where rule 5 does not fit.** Rule 5 gives `Extends:` to a
  successor that *adds to* an authoritative document. This one also replaces named clauses, which is
  a form the rule does not name (analyst, "Structural choices"). The repo already uses it: the egress
  spec over the proposal, and `p0-2.md`'s errata over the egress spec. Whether rule 5 should gain a
  clause-override form is the convention owner's question, routed by `teco`. Here the gap is bounded
  in two ways. Every replaced clause is named, in its §1 heading. And `-spec3` ends the arrangement
  (above). The amendments touch about a dozen clauses of a 1,400-line spec.
  A replacing document would re-emit the rest, golden vectors included, as an unreviewed
  re-transcription of gated text. A reviewer could trust it only after diffing all of it, and it would
  add no content. Each amendment below states exactly which clause it replaces. That is the
  precedent of the egress spec over the proposal, and of the successor plan's §2 D6 over the egress
  spec.
- **Cost, and where it is paid.** Until a consolidation, readers read two documents. S10's
  spec-derived tests parse the spec's markdown, so they must also read this document's replacement
  sections (§4.1, S10). `-spec3` ends this cost (above).

**What every amendment is tagged with.**
- **[pre-freeze]**: S10 must encode it before S15. It changes a field, a type, an id-bearing rule or
  what the validator accepts. Any of those after the freeze would be a major change (spec §8.6).
- **[writer]**: a rule for the P1 writers, which the validator does not check. It can be tightened
  after the freeze with no bump (§1.10).
- **[doc]**: wording only.

**Sources.**
- `docs/reviews/connected-code-graph-bindvalue.md` Pass 1 (S58, `security-expert`, advisory), with
  the stakeholder decisions of 2026-10-29:
  - FR-15's "never stores or shows a secret's value" covers the **graph and answers**, not the local
    bundle files;
  - internal hostnames in clear are acceptable.
- `docs/reviews/connected-code-graph-p0-impl.md` § `## S10 (P0-09)`: S10-m3, the nits, and
  "Questions answered" open items 1–3.
- `docs/plans/connected-code-graph-eval-protocol-ml.md` v1.1 §2.2 and §15 Q2 (S06).
- v1.1:
  - `docs/reviews/connected-code-graph-p0-impl.md` § `## Spec amendments (bundle-spec2,
    egress-spec2)` (analyst) and § `## Spec amendments — graph-dba check …`;
  - `docs/reviews/connected-code-graph-bindvalue.md` § `Pass 2 — dispositions check`.

CPG: considered, not relevant — the inputs are spec text and review findings. The one code question,
what S10's validator does today, was answered by reading `cpg/ccg/bundle.py` and the schema; no
`cpg_*` graph covers `cpg/`.

## 1. Amendments (normative)

### 1.1 Access-path grammar (replaces spec §5.1's grammar block and its quoting bullet) [pre-freeze]

```
argPath  = argHead *field                  argHead = "a" I | "recv" | "kw" field
defPath  = defHead *field                  defHead = "p" I | "recv" | "r"
usePath  = argPath | "ret" *field | ""
field    = "." IDENT | "[" JSONSTRING "]"   ; no "." before "["
I        = "0" | %x31-39 *DIGIT             ; 0-based, no leading zero
IDENT    = ( ALPHA | "_" | "$" ) *( ALPHA | DIGIT | "_" | "$" )     ; ASCII only
JSONSTRING = the key as Python's json.dumps(key, ensure_ascii=False) writes it
```

- **The bracket attaches with no dot.** `a1["content-type"]` and `kw.headers["a.b"]` are valid;
  `a1.["x"]` is not. The spec's own examples, its "one encoding per fact" rule and §5.17's
  `ports[0]` all use this form. The spec's grammar line, which allowed `.` before `[`, was the typo.
- **One encoding per key.** A key that is an `IDENT` is written `.key`. Every other key is written
  `["…"]`: an empty key, a non-ASCII one, or one containing `.`, `[`, a space or any other
  non-`IDENT` character. So `a1["method"]` is invalid, and a non-ASCII key is written `a1["café"]`.
- **`IDENT` is ASCII.** In Python's `re`, `\w` is Unicode. In ECMA-262, the JSON Schema dialect, it
  is ASCII. So a Unicode identifier class gives the schema and `bundle.py` two answers on `a1.café`.
  ASCII is the class on which both agree. The cost is only that non-ASCII keys are quoted, and the
  encoding is still unique. No golden vector has a non-ASCII path, so no id changes.
- **The `kw` head is `kw` plus one field**, the parameter name: `kw.prefix`. A non-ASCII parameter
  name, which Python allows, is `kw["café"]`.
- **The depth cap** (`fieldDepthCap`, 3) counts the fields **after** the head, in both forms.
  `kw.headers["a"]["b"]["c"]` has three. The head's own field is not counted.
- **§5.17's `keyPath` is unchanged.** It shares the bracket form, but it quotes only a segment
  containing `.` or `[`. The spec's sentence "This is §5.17's `keyPath` quoting" means the form, not
  the criterion. `keyPath` feeds `entryId` and the source-written `config` locators (§1.9), so it is
  not re-ruled here.

### 1.2 `nameNorm` (replaces the last sentence of spec §4.1 rule 10) [pre-freeze]

`nameNorm = NFC(casefold(NFC(name))).strip()`. Here `casefold` is Python 3.12's `str.casefold()`
(Unicode 15.0.0) and `strip` is §3.1's. The outer NFC is needed: for 26 code points, U+01F0 and
U+0390 among them, `casefold` of an NFC string is not NFC, and §8.1 would then reject the record.
The validator recomputes `nameNorm` and rejects a mismatch. It feeds no tuple.

### 1.3 Value-set nesting depth (clarifies spec §5.8 "nesting at most 4 deep") [pre-freeze]

An element's **depth** is the number of elements on the path from a top-level alternative down to
it, both ends counted. A top-level `{"const": …}` has depth 1. No element has a depth above 4, so
there are at most three nested `concat`/`template` containers above a leaf. A writer writes a
container that would sit at depth 4 as `{"unknown": "value-set-cap"}`. This is what S10's schema
already enforces (`element1`–`element4`, `elementLeaf`).

### 1.4 `controlledBy` (replaces spec §5.11's field sentence) [pre-freeze]

`point` is any point of the symbol: a call site, a `ret`, a `param` **or a `cond`**. `cond` is a
`cond` point, and `point ≠ cond`. A `cond`→`cond` record says that evaluating condition `point`
depends on condition `cond`, as in a nested `if`. Joern's CDG produces these. Without them, the
outer guard of a guarded call is unreachable by a `CONTROLLED_BY` traversal. Graph:
`(:Cond)-[:CONTROLLED_BY]->(:Cond)` is legal, so S17's edge-load anchors include `Cond` as a
**source** of `CONTROLLED_BY` (§1.11).

- **Writer (G-4).** The normalizer drops CDG self-edges and de-duplicates (`point`, `cond`) pairs
  before writing. A loop header's frontier can contain the header itself, and the validator's
  `point ≠ cond` check rejects the **whole bundle** on a single self-edge. A self-edge adds no
  reach, so dropping it loses nothing.
- **Cycles are expected (G-3).** A loop header controls its body's conditions, and an exiting
  `if c: break` controls the header back. So `CONTROLLED_BY` among `Cond`s has 2-cycles, and an
  anchor can re-enter its own reach set. Every traversal over it, the P1 impact/RCA BFS included,
  carries a visited set seeded with the anchor (graph note Q7, §6.1). Reach stays inside one
  function (distance 0), so it is bounded.

### 1.5 `import` (replaces spec §5.14's table) [pre-freeze]

Id tuple: `["import", fileId, ordinal]`. These are an assembly input: `library_api` consumption and
`{importRef}` resolution. No node.

| Field | Type | Req | Meaning |
|---|---|---|---|
| `fileId`, `line` | | yes | |
| `ordinal` | int | yes | the tuple's `ordinal`: 0-based per file, in source order. It is carried so that each record's id can be recomputed alone. A file's ordinals are exactly `0..k-1`, and their `line`s do not decrease in ordinal order. |
| `symbolId` | | no | present when the import is inside a function (a local import) |
| `module` | string | yes | the specifier as written: `./client`, `falkorchat.config`, `.` |
| `names` | array of `{name, alias?}` | yes | `[]` for a whole-module import; `*` for a star import. Strictly sorted by (`name`, `alias`), with an absent `alias` first (§8.1). **An exact repeat is written once:** `from m import a, a` gives `[{"name": "a"}]`. `a` and `a as b` are two entries. |
| `resolvedPath` | repo-relative path | no | when the normalizer resolves it to an in-repo file |
| `dynamic` | bool | yes | `import(…)`, `importlib.import_module(…)` or `require(<non-literal>)`. When `true` and the specifier is not a constant, the assembler emits `possible`/`dynamic-import` (AC-6). |

### 1.6 `launchFact` (replaces spec §5.18's `launchFact` paragraph and field list) [pre-freeze]

Id tuple: `["launchFact", repoId, path, kind, ordinal]`. These are explicit launch facts from
Dockerfiles, Compose, `.mcp.json`, Kiro agents and shell `exec` lines. Framework defaults, such as
uvicorn's 8000, come from packs at assembly, not from the bundle. Graph and `[P-later]` status are
as in the spec. The record's `repoId` is the header's.

| Field | Type | Req | Meaning |
|---|---|---|---|
| `path`, `line` | | yes | |
| `ordinal` | int | yes | the tuple's `ordinal`: 0-based per (`path`, `kind`), in source order. The density and line rule are those of `import` (§1.5). |
| `kind` | `command` \| `port` \| `env-default` \| `mount` | yes | |
| `command` | string | when `kind` is `command` | argv[0] after the argv gate (§1.6.1); `"<redacted>"` when it fails |
| `args` | array of string | when `kind` is `command`, possibly empty | argv[1…] **in argv order, never sorted**, each element after the argv gate |
| `port` | int 1–65535 | when `kind` is `port` | |
| `protocol` | string | when `kind` is `port` | |
| `envVar` | string | when `kind` is `env-default` | the variable name as written: `AGENT_TEAM_PORT` |
| `defaultHash` | 64-hex | when `kind` is `env-default` | HMAC of the default's text, by §1.7.5, with this record's `id` as the domain. It is always present, so **per-record change detection** works when `default` is withheld. It equals no other record's hash, so it supports no matching. Env binding uses `envVar`, the name, whether or not `default` is present (G-2). |
| `default` | string | **only** when `kind` is `env-default` and the default passes §1.7.2's value grammar, `envVar` passes §1.7.3's name rule, **and** W2 holds (a single-label host default must be a declared service, §1.7.4) | the default in clear: `AGENT_TEAM_PORT:-8200` gives `default: "8200"`. `${ROOT_PHRASE:-hunter}` gives no `default` (W2). |
| `mountPath` | string | when `kind` is `mount` | |
| `service` | string | no | the Compose service, when there is one |
| `origin` | `explicit` \| `env-default` \| `compose` | yes | |

A field of another `kind` is absent (S10's per-kind exclusivity).

#### 1.6.1 The argv gate

**One rule, validator-checked, before the freeze (v1.1).** This gate replaces v1.0's (g1)–(g3). It
closes analyst BS-M1 and BS-M2 and security P2-1 together, and both reviewers check it against the
same AV table. It is validator-checked because v1.0's gate already was. Under §1.10, a stricter
validator after S15 is a major change, so the fix lands now. It is not the writer-only W5 that P2-1
proposed, though P2-1 also judged a writer rule acceptable. A writer may still withhold more
(§1.10).

**The emitted sequence decides.** Every condition below reads `command` and `args` **as emitted**.
So the validator and the writer apply one rule to the same text, which closes BS-m5's
"the validator cannot see what the writer saw".

**v1.2: the context-loss defect, fixed once (analyst BS-P2-M1 = security P3-1 (a)–(d)).** In
v1.1, a redacted element that would have supplied context to its successor was written as a plain
`<redacted>`, so the successor was judged with no name. That covered:
- a single-dash long option (`-password hunter2`, `keytool -storepass`, `openssl -passout`);
- the forced redaction after a secret-named flag (`--no-auth --token abc123`);
- a writer withholding a flag.

v1.2 has **two kinds of sentinel**, and one rule decides which is written:
- **`-<redacted>` and `--<redacted>` force the next element.** One of these is written whenever the
  redacted element would, **read in clear, supply context**: it is flag-shaped (class 2 or 4), or
  it is a bare value containing a context word (below). `--` is used when the source element
  starts with `--`, and `-` otherwise.
- **`<redacted>` forces nothing.** It is written for every other redacted element.

**Element classes.** `args` is read left to right. Each element is exactly one of these, tested in
this order:
1. **A redaction sentinel:** `<redacted>`, `-<redacted>` or `--<redacted>`.
2. **A flag:** `--NAME`, with `NAME` matching `[A-Za-z][A-Za-z0-9._-]*`, or `-X`, exactly two
   characters with `X` from `[A-Za-z0-9]`, **or an entry of the closed single-dash allow-list**.
   A flag has no `=`. `NAME` admits `.`, so `--db.password` is a flag (BS-M2). The allow-list
   (v1.2, widened in v1.3):
   - **exact strings:**
     - `-jar`, `-cp`, `-classpath`;
     - Go role flags: `-port`, `-addr`, `-host`, `-listen`, `-bind`;
     - JVM flags: `-ea`, `-da`, `-server`, `-client`;
   - **closed shapes:**
     - `-Xmx<n>`, `-Xms<n>` and `-Xss<n>`, where `<n>` is digits plus an optional `k`/`m`/`g` in
       either case;
     - `-XX:+Name` and `-XX:-Name`, where `Name` is `[A-Z][A-Za-z0-9]{0,63}`;
     - `-verbose:gc`, `-verbose:class`, `-verbose:jni` and `-verbose:module`.

   An entry is a flag only if it also passes the token screen, and every `.`/`:`/`+`/`-`-separated
   part of it passes §1.7.3. So `-XX:+Token` and `-XX:+hunter2Secret9` are not flags: they are
   class 4 and get `-<redacted>`. An entry's **name** is its text after `-` (after `-XX:+`/`-XX:-`
   for that shape; before the `:` for `-verbose:`). `-XX:Name=value` contains `=`, so it is an
   assignment, whose name fails the name pattern, and it gets `<redacted>`.
3. **An assignment:** any other element containing `=`. Its **name** is the text before the first
   `=` with leading `-`s removed; its **value** is the rest. This covers `--k=v`, `-Dk=v`, `-e`'s
   operand `K=V`, `env K=V` and `--set a.b=v`, with or without a dash (BS-M2, P2-1(b)).
4. **A single-dash element:** `-` followed by two or more characters, not starting `--`, with no
   `=` and not allow-listed. Examples are `-phunter2`, `-uroot`, `-password` and `-storepass`. It
   has **two readings, and both must pass**:
   - an **attached short flag**, `-X` plus the attached value, whose context is the
     single-character name `X` (P2-1(a));
   - a **single-dash long option** whose name is the text after the `-` (P3-1(a)).
5. **A value:** anything else.

**Context.** A value's **context** comes from the immediately preceding element:
- a flag (class 2) supplies its `NAME`, its `X`, or its allow-listed name;
- **a bare value in clear that contains one of the 25 substring words** of §1.7.3 supplies
  itself, as a name. This covers `redis-cli AUTH hunter2`, `CONFIG SET requirepass hunter2` and
  `mysqladmin password hunter2` (P3-1(c)). The 11 whole-token words supply nothing, so AV-8's
  `ssh h` is unchanged;
- no other element supplies context. A sentinel supplies none, but a forcing sentinel forces.

An assignment's context is its own name plus the preceding element's context, if any.

**Forcing.** The element after a **secret-named flag** (its name fails §1.7.3), or after a forcing
sentinel, is **forced**. It is redacted whatever its class, and written as a forcing sentinel when
it would supply context, otherwise as `<redacted>`. So forcing propagates exactly as far as context
would have:
- `--no-auth --token abc123` gives `--no-auth`, `--<redacted>`, `<redacted>` (P3-1(d));
- `--token -x7hunter` gives `--token`, `-<redacted>`.

**What may be written in clear.** An element that is not forced passes iff its class passes:
- **A flag** passes iff it has 1–128 characters and passes the token screen (§1.7.1). A failing
  flag is written as its forcing sentinel.
- **A value, an assignment's value, or an attached value** passes iff all of these hold:
  - **(v1) Shape.** 1–128 characters from `[A-Za-z0-9._/:@+=,-]`, which fails `$NAME`/`${…}`,
    quotes and spaces. `[` and `]` are also admitted, **only** when the whole value matches
    §1.7.2's `hostport` or `url`, so `--host [::1]:8000` passes (v1.2, BS-P2-m2);
  - **(v2) Screen.** If the value is **address-shaped**, matching §1.7.2's `ipv4`, `ipv6`,
    `hostport`, `dnsname` or `url` or (v6)'s port mapping, it is screened by its production,
    label by label and segment by segment. Otherwise it passes the token screen as a whole
    string (§1.7.1's scope rule);
  - **(v3) Names.** Every name in its context passes the name rule (§1.7.3). An assignment's name
    is split at `.` into segments first, so `db.password` is judged on `password`. A failing name
    fails the value;
  - **(v4) No userinfo.** It has no `:` before an `@`, which fails `user:pass@…` in any position.
    `some-mcp@latest`, `@scope/pkg` and `git@host:org/repo` pass (BS-M1);
  - **(v5) URLs are URLs.** If it contains `://`, the whole value matches §1.7.2's `url`
    production. That excludes userinfo, queries, fragments and multi-segment paths (BS-M1);
  - **(v6) Single-character context.** If its context is a single-character name (`-p v`,
    `-pV`, `-a v`), it must match §1.7.2's `port`, `ipv4`, `dnsname` or `hostport`, or the
    port-mapping shape `[ipv4 ":"] port [":" port] ["/" ("tcp" / "udp")]`. A bare single label
    is not enough, so `-a hunter2` fails while `-p 8000` and `-p 8000:8000` pass (P2-1(c)). A
    one-letter flag gives no clue whether it takes a secret (`sshpass -p` versus `docker -p`),
    so the shape must. **(v6) fires when any name in the context is a single character**, an
    assignment's included (v1.2, BS-P2-m3). So `-e FALKORDB_HOST=falkordb` is redacted, because a
    single label is not address-shaped (AV-40), while `-e FALKORDB_PORT=6379` passes (AV-41);
  - **(v7) User context (v1.2, P3-1(b); widened in v1.3, BS-P3-m3).** If any context name has
    one of the whole tokens `user`, `login`, `basic`, `account`, `principal` or `identity`, a
    value containing `:` fails. That covers `--user admin:hunter2`, `--proxy-user u:p` and
    `--account admin:hunter2`. `docker --user 1000:1000` is over-redacted; it is not a binding
    input.
  - **(v8) `NAME:secret` (v1.3, BS-P3-M1).** A value that contains `:`, is **not**
    address-shaped (v2) and has no `://` fails when its text before the first `:` fails §1.7.3,
    after a split at `.`. This is BS-M2's `NAME=secret` class, with `:` as the separator. It
    covers `--header X-Api-Key:hunter2`, `pass:hunter2`, `Authorization:Bearer-hunter2` and
    `PIN:4821`. A failing value that contains a context word is written `-<redacted>`, as before,
    so the next element is forced.
    - **The address exemption is deliberate, and was checked by run.** `falkordb:6379`,
      `localhost:8000`, `keycloak:8443` and `[::1]:8000` are `hostport`s, so (v8) never reads
      their host as a name. `keycloak` contains `key`, and a non-exempt (v8) would have redacted
      it.
    - **Non-address values that still pass:** `falkorchat.app:app`, `X-Request-Id:abc` and
      `go run . -addr :8080`'s `:8080`, whose empty head passes.
- **An assignment** passes iff its name matches `[A-Za-z][A-Za-z0-9._-]*` and passes the token
  screen, and its value passes. Otherwise the whole element is `<redacted>`.
- **A single-dash element** passes iff its attached value passes under context `X`, its long name
  passes §1.7.3, and the whole element passes the token screen. Otherwise it is written
  `-<redacted>`, which forces the next element. `-p8000` passes. `-password`, `-storepass`,
  `-pin`, `-phunter2` and `-uroot` do not.
- **BS-m5's "flag after a flag"** still holds: `--verbose --token abc` keeps `--token` and redacts
  `abc`.

`command` (argv[0]) passes iff it meets (v1) and (v2), and is otherwise `<redacted>`. A failing
`args` element is written as its sentinel, which keeps the argv's arity.

**A writer that withholds an element (§1.10)** writes the sentinel this rule would write for it: a
forcing sentinel for a flag-shaped element or a context-word value. A plain `<redacted>` in place
of a context supplier would hide the context from the validator. So that is a writer defect, not
permitted withholding.

**The validator's view.** Every rule above reads the emitted sequence. A forcing sentinel tells the
validator that the next element must be redacted. A plain `<redacted>` gives the next element no
context, and the writer rule above makes that a correct reading.

**Accepted over-redaction.** A value after a boolean one-letter flag is redacted unless it is
address-shaped: `npx -y some-mcp` gives `-y`, `<redacted>` (AV-19). Launch binding resolves
**in-repo** targets (`python -m falkorchat.server`, `uv run --directory cypher-mcp …`), and those
pass. An external npx package name is not a binding input. The analyst's Pass 2 list, recorded
here:
- `-e FALKORDB_HOST=falkordb` (AV-40);
- every value after a flag whose name contains a substring word, such as `conn`, `cert` or `key`:
  `curl --connect-timeout 5` (AV-48);
- every element forced after a secret-named flag, such as `--auth-enabled --port 8000`;
- a value containing a substring word, which is written as a forcing sentinel (AV-16, AV-26);
- **(v1.3, BS-P3-m3) an internal address under an auth-, key- or session-named flag, or after a
  context-word value.** The stakeholder allowed these hostnames in clear, so this fails closed and
  costs binding evidence:
  - `--keycloak keycloak.corp.internal:8443` gives `--keycloak`, `-<redacted>` (AV-62);
  - `--auth-url http://auth-svc:8000` gives `-<redacted>` (AV-63);
  - `kubectl port-forward svc/session-db 5432:5432` loses its port mapping (AV-64).

  No cheap rule recovers these without re-admitting a value after a secret-named name, so they are
  recorded, not closed.

**The AV-21 trade-off (v1.2).** The analyst's fix alone, every redacted flag-shaped element forces,
would have cost `java -Xmx2g -jar app.jar` its `app.jar`. The JVM options are single-dash elements,
and so they would cascade. I kept `app.jar` with the **closed allow-list** in class 2 (`-jar`,
`-cp`, `-classpath`, `-Xmx/-Xms/-Xss<n>`). It is what the analyst offered as "the
over-redaction-only way". The list is exact strings and digit-only shapes, so it admits no
secret-bearing form.

I rejected security P3-1(a)'s alternative, "a single-dash element whose long name passes the name
rule is written plain". It keeps more in clear, but it is not validator-checkable: a plain
`<redacted>` cannot show whether its source name passed. That is exactly BS-P2-M1's failure mode.
Adding to the allow-list later is a loosening, so it is a major change (§1.10).

**v1.3, BS-P3-m2: the allow-list is widened now, not left as an accepted loss.**
- **Why.** Go is an FR-1 language, and its `flag` package is single-dash, so `-port 8080` and
  `-addr :8080` are exactly the explicit launch flags the proposal's §3.4 binds on. P2's Go recall
  would pay for the loss. The common JVM flags (`-XX:+…`, `-ea`, `-server`, `-verbose:gc`) cost
  `app.jar`. Widening after S15 would be a major change.
- **Why no added entry lets a secret through** (run 2026-10-30):
  1. Every entry is an exact string or a closed shape with no free value.
  2. An entry must itself pass the screen and the name rule, so `-XX:+Token` is not one (AV-61).
  3. An entry supplies its name as context exactly as `--NAME` does. For each of the 15 entries,
     the value after it is written identically to the value after its `--NAME` form.
  4. Every entry stays flag-shaped, so it forces when it is itself forced:
     `--auth-enabled -port 8080` gives `-<redacted>`, `<redacted>`.
  5. A grid of every entry as a prefix in front of §1.6.3's eleven placements, all 36 words × 3 cases
     × 4 suffixes, is 71,280 cells, with 0 values in clear.

  The only value an entry passes is one that its `--` form would also pass: the neutral-named-flag
  residual (AV-68).

**Vectors** (normative). AV-1 to AV-5 are from v1.0. AV-1 now uses a full synthetic key (BS-n1),
and its `some-mcp` is redacted under (v6). AV-6 to AV-12 are P2-1's must-redact lines. AV-13 to
AV-24 are BS-M1's transplants, BS-M2's placements and BS-m5's cases. **v1.2** changes AV-6,
AV-16 and AV-18, and adds AV-25 to AV-48:
- AV-25 to AV-32 are security P3-1's forms;
- AV-33 to AV-39 are analyst BS-P2-M1's;
- AV-40 to AV-48 are BS-P2-m2, BS-P2-m3, P3-2 and the host interaction cases.

All 48 were reproduced on 2026-10-30 by a throwaway scratchpad emulation, not a repo file.

| # | Source argv | Written (`command`, then `args`) |
|---|---|---|
| AV-1 | `npx -y some-mcp --api-key sk_live_FAKE-0000-EXAMPLE-0000-NOT-REAL` | `npx`, `-y`, `<redacted>`, `--api-key`, `<redacted>` |
| AV-2 | `server --port 8000 --token ghp_16C7e42F292c6912E7710c838347Ae178B4a` | `server`, `--port`, `8000`, `--token`, `<redacted>` |
| AV-3 | `worker --api-token=9f86d081884c7d659a2feaa0c55ad015` | `worker`, `<redacted>` |
| AV-4 | `uvicorn falkorchat.app:app --host 0.0.0.0 --port 8000` | unchanged |
| AV-5 | `sh -c "exec $CMD"` | `sh`, `-c`, `<redacted>` |
| AV-6 | `mysql -uroot -phunter2` | `mysql`, `-<redacted>`, `-<redacted>` (v1.2) |
| AV-7 | `redis-cli -a hunter2 ping` | `redis-cli`, `-a`, `<redacted>`, `ping` |
| AV-8 | `sshpass -p hunter2 ssh h` | `sshpass`, `-p`, `<redacted>`, `ssh`, `h` |
| AV-9 | `curl -u admin:hunter2 http://svc:8000/x` | `curl`, `-u`, `<redacted>`, `http://svc:8000/x` |
| AV-10 | `psql postgres://app:hunter2@db:5432/app` | `psql`, `<redacted>` |
| AV-11 | `docker run -e DB_PASSWORD=hunter2 img` | `docker`, `run`, `-e`, `<redacted>`, `img` |
| AV-12 | `helm --set db.password=hunter2` | `helm`, `--set`, `<redacted>` |
| AV-13 | `env DB_PASSWORD=hunter2 server` | `env`, `<redacted>`, `server` |
| AV-14 | `server --db.password hunter2` | `server`, `--db.password`, `<redacted>` |
| AV-15 | `x --database-url postgres://u:p@db/app` (BV-17's value) | `x`, `--database-url`, `<redacted>` |
| AV-16 | `x --url https://hooks.slack.com/services/T0/B0/XXXX` (BV-6's value) | `x`, `--url`, `-<redacted>` (v1.2: the value contains `hook`, a context word) |
| AV-17 | `x --verbose --token abc` | `x`, `--verbose`, `--token`, `<redacted>` |
| AV-18 | `x --token -x7hunter` | `x`, `--token`, `-<redacted>` (v1.2) |
| AV-19 | `npx -y some-mcp@latest @scope/pkg` | `npx`, `-y`, `<redacted>`, `@scope/pkg` |
| AV-20 | `x --log4jConfigurationFile1 conf.xml` | `x`, `--<redacted>`, `<redacted>` |
| AV-21 | `java -Xmx2g -jar app.jar` | unchanged (v1.3 row fix, BS-P3-m1: v1.2's allow-list already made this output; the row was stale) |
| AV-22 | `python -m falkorchat.server -p 8000` | unchanged |
| AV-23 | `x -e webhook=https://hooks.slack.com/services/T0/B0/XXXX` | `x`, `-e`, `<redacted>` |
| AV-24 | `x -e ADMIN_PIN=4821` | `x`, `-e`, `<redacted>` |
| AV-25 | `keytool -storepass hunter2` | `keytool`, `-<redacted>`, `<redacted>` |
| AV-26 | `openssl pkcs12 -passout pass:hunter2` | `openssl`, `pkcs12`, `-<redacted>`, `-<redacted>` |
| AV-27 | `consul-server -password hunter2` | `consul-server`, `-<redacted>`, `<redacted>` |
| AV-28 | `curl --user admin:hunter2 http://svc/x` | `curl`, `--user`, `<redacted>`, `http://svc/x` |
| AV-29 | `mysqladmin -uroot password hunter2` | `mysqladmin`, `-<redacted>`, `-<redacted>`, `<redacted>` |
| AV-30 | `redis-cli AUTH hunter2` | `redis-cli`, `AUTH`, `<redacted>` |
| AV-31 | `redis-cli CONFIG SET requirepass hunter2` | `redis-cli`, `CONFIG`, `SET`, `requirepass`, `<redacted>` |
| AV-32 | `server --no-auth --token abc123` | `server`, `--no-auth`, `--<redacted>`, `<redacted>` |
| AV-33 | `redis-server --no-auth --requirepass hunter2` | `redis-server`, `--no-auth`, `--<redacted>`, `<redacted>` |
| AV-34 | `x --auth-enabled --password hunter2` | `x`, `--auth-enabled`, `--<redacted>`, `<redacted>` |
| AV-35 | `x --log4jConfigurationFile1 --password hunter2` | `x`, `--<redacted>`, `--<redacted>`, `<redacted>` |
| AV-36 | `java -keystorePassword changeit -jar app.jar` | `java`, `-<redacted>`, `<redacted>`, `-jar`, `app.jar` |
| AV-37 | `server -db-password hunter2` | `server`, `-<redacted>`, `<redacted>` |
| AV-38 | `server -token abc123` | `server`, `-<redacted>`, `<redacted>` |
| AV-39 | `server -pin 4821` | `server`, `-<redacted>`, `<redacted>` |
| AV-40 | `docker run -e FALKORDB_HOST=falkordb img` | `docker`, `run`, `-e`, `<redacted>`, `img` (accepted over-redaction) |
| AV-41 | `docker run -e FALKORDB_PORT=6379 img` | unchanged |
| AV-42 | `x --host [::1]:8000` | unchanged |
| AV-43 | `app --opt x7k2:q9w4:m1p8:z3v6` | `app`, `--opt`, `<redacted>` (T6 over `:`) |
| AV-44 | `app --opt x7k2+q9w4+m1p8+z3v6` | `app`, `--opt`, `<redacted>` (T6 over `+`) |
| AV-45 | `docker run -p 192.168.100.200:8000:8000 img` | unchanged (address-shaped, screened per production) |
| AV-46 | `x --db mydb.c9akciq32.us-east-1.rds.amazonaws.com:5432` | unchanged (internal hostname, stakeholder 2026-10-29) |
| AV-47 | `x -p8000` | unchanged |
| AV-48 | `curl --connect-timeout 5 http://svc:8000/x` | `curl`, `--connect-timeout`, `<redacted>`, `http://svc:8000/x` (accepted over-redaction: `conn`) |
| AV-49 | `curl --header X-Api-Key:hunter2 http://svc/x` | `curl`, `--header`, `-<redacted>`, `<redacted>` (v8; forcing, because the value contains `key`) |
| AV-50 | `curl --header=X-Api-Key:hunter2` | `curl`, `<redacted>` |
| AV-51 | `x pass:hunter2` | `x`, `-<redacted>` |
| AV-52 | `x Authorization:Bearer-hunter2` | `x`, `-<redacted>` |
| AV-53 | `x PIN:4821` | `x`, `<redacted>` (v8 on the whole token `pin`; no context word, so not forcing) |
| AV-54 | `x --header X-Request-Id:abc` | unchanged |
| AV-55 | `x --account admin:hunter2` | `x`, `--account`, `<redacted>` (v7 widened) |
| AV-56 | `server -port 8080` | unchanged (allow-list) |
| AV-57 | `go run . -addr :8080` | unchanged (allow-list) |
| AV-58 | `java -XX:+UseContainerSupport -jar app.jar` | unchanged (allow-list) |
| AV-59 | `java -ea -jar app.jar` | unchanged (allow-list) |
| AV-60 | `java -verbose:gc -jar app.jar` | unchanged (allow-list) |
| AV-61 | `x -XX:+hunter2Secret9 -jar app.jar` | `x`, `-<redacted>`, `-<redacted>`, `<redacted>` (a shaped entry whose name fails is not allow-listed) |
| AV-62 | `x --keycloak keycloak.corp.internal:8443` | `x`, `--keycloak`, `-<redacted>` (accepted over-redaction) |
| AV-63 | `x --auth-url http://auth-svc:8000` | `x`, `--auth-url`, `-<redacted>` (accepted over-redaction) |
| AV-64 | `kubectl port-forward svc/session-db 5432:5432` | `kubectl`, `port-forward`, `svc/session-db`, `<redacted>` (accepted over-redaction) |
| AV-65 | `x --host falkordb:6379` | unchanged (address-shaped; v8 exempt) |
| AV-66 | `x keycloak:8443` | unchanged (address-shaped; v8 exempt) |
| AV-67 | `x 3f2a:9c1e:5b7d:4a6f:8e0c:2b4d:6a8f:e1c` | unchanged (named residual: a hex token as canonical IPv6) |
| AV-68 | `x -host hunter2` | unchanged (as `--host hunter2`: the neutral-named-flag residual) |

**Named residuals (v1.2).** None of these is closed by any argv rule. All of them are routed to
`graph-dba` with S17's G-1 decision, where P3-1 recommends carrying only role-relevant `launchFact`
fields and never the whole of `args`:
- **An unflagged positional secret**, such as `rabbitmqctl add_user admin hunter2` or
  `htpasswd -b f admin hunter2`. It was already accepted, because no argv rule can see it.
- **A value after a whole-token-only bare word**, such as `vault hunter2`. Those 11 words supply no
  context, to keep `ssh h` (AV-8).
- **A password-only userinfo with no scheme**, such as `hunter2@db:6379`, which (v4) admits
  (security Pass 3).
- **A letters-only random token of 16–23 characters** (P3-3), for example
  `--id kxqzmwvptryhbnjdlcqp`.
- **(v1.3, BS-P3-m3) A hex token written as canonical IPv6**, such as
  `3f2a:9c1e:5b7d:4a6f:8e0c:2b4d:6a8f:e1c` (AV-67). `ipv6` has no labels, so nothing in it is
  screened, and any hex rule would also fail real addresses like `2001:db8:85a3::8a2e:370:7334`.
  The same applies to `bindValue` (security Pass 3: "theoretical").
- **(v1.3) A secret after a neutral-named allow-listed flag**, `-host hunter2` (AV-68). It is the
  same, and no wider, than the long form `--host hunter2`, which was always the
  neutral-named-flag residual. The run below shows every entry behaves exactly like its `--` form.
- **A chunked token spread over host labels**, `x7k2.q9w4.m1p8.z3v6.example.com`. It is
  address-shaped, so it is screened label by label (§1.7.1's scope rule). The same text that is not
  host-shaped fails T6.

#### 1.6.3 The argv coverage probe (normative; S10 implements it verbatim; v1.2, separator axis v1.3)

This is the analyst's falsifiable stopping condition for §1.6.1 (Pass 2, narrowed in Pass 3), and
it is S10's test. §1.6.1 is done when the probe passes. A later re-gate of §1.6.1 is needed only
when a cell goes red, or when an element class, a sentinel kind or a **name/value separator** is
added.

- **Words `w`:** the 36 words of §1.7.3, the 25 substring words and the 11 whole-token words.
- **Cases:** `W0` is `w.upper()`, `w.lower()`, and `"my" + w[0].upper() + w[1:]`.
- **Suffixes:** `W = W0 + s`, for `s` in `""`, `"2"`, `"_2"`, `"1"`.
- **Placements**, eight of them, each a list of argv elements, with `v = "hunter2"`:
  1. `["--"+W, v]`
  2. `["--"+W+"="+v]`
  3. `["-"+W, v]`
  4. `["-"+W+"="+v]`
  5. `["-e", W+"="+v]`
  6. `[W+"="+v]`
  7. `["--x."+W, v]`
  8. `["--x."+W+"="+v]`
- **The separator axis** (v1.3, BS-P3-M1). Three more placements, on the same words, cases,
  suffixes and prefixes, form a separate count:
  9. `[W+":"+v]`
  10. `["--header", W+":"+v]`
  11. `["--header="+W+":"+v]`
- **Prefixes**, inserted after `argv[0] = "x"` and before the placement:
  - `[]`;
  - `["--auth-enabled"]`, a passing secret-named flag;
  - `["--log4jConfigurationFile1"]`, a screen-failing flag;
  - `["-Xmx2g"]`;
  - `["-q7zz"]`, a failing single-dash element.

  `-Xmx2g` is allow-listed in v1.2, so it passes, and `-q7zz` keeps the grid's intended case of a
  failing class-4 element in front.
- **Pass condition:** no emitted element contains `hunter2`, in any of:
  - the grid, 36 × 3 × 4 × 8 × 5 = **17,280** cells;
  - the separator axis, 36 × 3 × 4 × 3 × 5 = **6,480** cells.
- **BS-M1's transplant** is part of the probe. Take the 15 must-redact rows BV-6–BV-20, one value
  each. Place each value in the eight placements, with the flag name taken from the keyPath's last
  segment (`command` for an index), plus a bare positional, under the five prefixes. That is
  15 × 9 × 5 = 675 cells. **Pass condition:** no value in clear outside the bare-positional
  column.
- **The tables are part of the probe.** S10 parses AV-1–AV-68 and BV-1–BV-28 **from this
  document's text**: the input, and the expected output or outcome ("unchanged" meaning the input
  itself). It compares them mechanically. Expected values are never retyped into the test (BS-P3-m1).
- **Observed on 2026-10-30, against v1.3**, in the architect's scratchpad emulation, with the tables
  parsed from this file:

  | Check | Cells or rows | In clear or mismatched |
  |---|---|---|
  | Grid | 17,280 | **0** |
  | Separator axis | 6,480 | **0** |
  | Transplant | 675 | 2, both BV-11's `4821` as a bare positional (the accepted residual) |
  | AV-1–AV-68, parsed | 68 | **0** |
  | BV-1–BV-28, parsed | 28 | **0** |
  | Allow-list prefix grid (§1.6.1, BS-P3-m2) | 71,280 | **0** |

  Against v1.2, the analyst's Pass 3 found 2,592 of the 6,480 separator-axis cells in clear.

  **The v1.2 claim "AV-1 to AV-48 reproduce" was wrong for AV-21.** My v1.2 check printed the
  emulation's outputs and compared them with the rows by eye, not with the file. The emulation
  already implemented the allow-list, and the table row was never updated when the allow-list was
  added. The mechanical comparison above is the fix.

#### 1.6.2 What reaches the graph or an answer

- **Gated fields.** `command`, `args` and `default` passed the gates above (§1.6.1, §1.7.2,
  §1.7.3, W2), and `port` is an integer.
- **Ungated names and paths.** `envVar`, `mountPath`, `service` and `protocol` are not gated.
  They are names and paths from source, the same class as `keyPath`. Wherever they are shown they
  go through F-4's screen and F-3's fencing (BS-m3).
- **`defaultHash` is never rendered.**
- **No other text of the source line reaches the graph or an answer.**
- **Whether the graph carries the gated fields at all is S17's decision** (G-1; §1.11). Until it
  rules, binding evidence renders as `path`/`line` only.

### 1.7 `configEntry`: the `bindValue` gate and `valueHash` (replace spec §5.17's `valueHash` and `bindValue` rows)

`bindValue` is emitted iff the value passes the value grammar (§1.7.2), the entry passes the name
rule (§1.7.3), the argv exclusion (§1.7.4 W1) holds, and the writer conditions (§1.7.4) hold. The
rest of §5.17 is unchanged: the `valueHash` HMAC key, `example`, and `bindValue`'s absence from the
graph.

#### 1.7.1 The token screen [pre-freeze]

A string **fails** the screen iff any of these holds. A *run* is a maximal sequence of ASCII letters
and digits.
- **T1:** a run of ≥ 16 characters containing at least one letter and at least one digit;
- **T2:** a run of ≥ 24 characters;
- **T3:** a run of ≥ 12 characters, all of them hex digits (`0-9a-fA-F`);
- **T4:** a UUID shape, `8-4-4-4-12` hex in either case, anywhere in the string;
- **T5:** the string has ≥ 20 characters and, at its start or right after one of `=` `/` `:` `@`
  `,`, it carries one of these case-sensitive prefixes: `xoxa-` `xoxb-` `xoxp-` `xoxr-` `xoxs-`
  `ghp_` `gho_` `ghu_` `ghs_` `ghr_` `github_pat_` `glpat-` `sk-` `sk_live_` `sk_test_` `rk_live_`
  `pk_live_` `pk_test_` `AKIA` `ASIA` `AIza` `hvs.` `eyJ`;
- **T6 (v1.1, chunked tokens; P2-2; class widened in v1.2, P3-2):** a maximal substring of
  `[A-Za-z0-9._:+,-]`, with `.`, `_`, `:`, `+`, `,` and `-` removed, leaves ≥ 16 characters
  that are either all hex digits or have ≥ 6 letter↔digit transitions. So these fail:
  - `x7k2-q9w4-m1p8-z3v6` (15 transitions);
  - `3f2a9c1e-5b7d4a6f-8e0c2b4d` (24 hex);
  - `x7k2:q9w4:m1p8:z3v6` and `x7k2+q9w4+m1p8+z3v6`.

  `release-2024-notes` (2 transitions), `ipv4-addresses-2024` (3), `s3-bucket-v2-us-east-1` (5)
  and `falkorchat.app:app` (0) pass.

**Scope of the screen (v1.2, BS-P2-m1, with the P3-2 interaction).** The screen is a function of
one string. **Which string** is fixed per production:
- **Inside §1.7.2's grammar,** every `label` and every `seg` is screened **on its own**, and
  `bindValue` is never screened as a whole string.
- **In argv,** an address-shaped value (§1.6.1 (v2)) is screened the same way, by its production.
- **Step 8a** screens each path segment.
- **Every other argv value or flag** is screened as a whole string.

Without this rule, the `:`-widened T6 run over the whole string would fail legitimate addresses
(run 2026-10-30):
- `mydb.c9akciq32.us-east-1.rds.amazonaws.com:5432`, an internal hostname the stakeholder allowed
  in clear (2026-10-29);
- `10.100.200.250:65535`;
- `2001:db8:85a3::8a2e:370:7334`.

Under the rule, `falkordb:6379`, `localhost:8000`, `[::1]:8000`, the RDS endpoint with and
without a port, and `192.168.100.200:8000:8000` all pass, in `bindValue` and in argv (BV-21,
BV-22, BV-27; AV-42, AV-45, AV-46).

**The cost is one residual:** a chunked token spread across host labels,
`https://x7k2.q9w4.m1p8.z3v6.example.com/`, binds (BV-28). The other reading, the whole string,
would have redacted the RDS endpoints the stakeholder allowed.

**The widened class also applies to step 8a.** This is one shared screen, as decided for T6 in
v1.1. P3-2 proposed it for argv only. Because 8a is id-bearing, the widening lands before the
freeze. It changes a key only for a path segment whose `:`/`+`/`,`-joined chunk reaches T6's
thresholds, and it changes no SL-1/SL-2 key or golden vector.

**Why T6 is decided now, in the shared screen.**
- **It is id-bearing through step 8a (§1.8)**, so it is pre-freeze or never: added later, it
  would re-id every affected `cpId`.
- **One screen everywhere.** The alternative, T6 for 8a only and a writer-only T6 for argv and
  `bindValue`, would give `bundle.py` two screens and the validator a weaker one than the writer.
  The cost is over-redaction of a dash-chunked identifier with many letter↔digit alternations,
  which is rare in route segments and config hosts.
- **No golden vector or slice key moves.** GV-9's segments have no digit, and SL-1/SL-2 are
  unchanged. Every BV row keeps its outcome.

Over-redaction is intended. The screen is the advisory's SA-4(a) with two departures, both against
false positives. First, the length rules count runs, not whole strings, so `release-2024-notes`
passes. Second, a prefix counts only on a string of ≥ 20 characters, so `sk-SE`, `pkg` and `skus`
pass. Every token shape the advisory names is longer than 20 characters. The argv gate (§1.6.1),
the value grammar (§1.7.2) and the `http` key rule (§1.8) all use the screen. `bundle.py` implements
it once.

#### 1.7.2 The value grammar (replaces (a) and (c)) [pre-freeze]

`bindValue` is a closed, case-sensitive, anchored grammar. It applies to **the exact string
emitted**, after the format's own unescaping and §8.1's NFC. No URL library parses it. The value
has fewer than 512 characters.

```
bindValue = port / host / ipv6 / hostport / url  ; v1.1: bare ipv6 and hostport added
hostport  = host ":" port                        ; localhost:8000, falkordb:6379, [::1]:8000
port      = %x31-39 0*4DIGIT                     ; value 1–65535
host      = "localhost" / "host.docker.internal" / ipv4 / "[" ipv6 "]" / label / dnsname
ipv4      = dec-octet 3( "." dec-octet )         ; dec-octet = "0" / %x31-39 0*2DIGIT, value ≤ 255
ipv6      = 2*39( DIGIT / %x61-66 / ":" )        ; and equal to Python 3.12's str(ipaddress.IPv6Address(ipv6))
label     = 1*63( %x61-7A / DIGIT / "-" )        ; no leading or trailing "-"; passes the token screen
dnsname   = label 1*( "." label )                ; last label 2*63( %x61-7A ); at most 253 characters
url       = scheme "://" host [ ":" port ] [ "/" [ seg [ "/" ] ] ]
scheme    = "http" / "https" / "ws" / "wss" / "redis" / "rediss" / "postgres" / "postgresql"
          / "mysql" / "mongodb" / "amqp" / "amqps" / "nats" / "grpc"            ; v1.1: non-web schemes
seg       = 1*32( %x61-7A / DIGIT / "-" / "_" )  ; passes the token screen
```

- **v1.1, BS-m2: `hostport`, bare `ipv6` and non-web schemes are admitted before the freeze.**
  - **Why now.** Each adds no exposure class. `host` and `port` are already bindable alone, and a
    URL of any scheme under this grammar still has no userinfo, query, fragment or second
    segment, so `postgres://u:p@db/app` (BV-17) still fails.
  - **The forms are the proposal §3.4's common binding values,** and SA-4(a) had promised to keep
    `falkordb:6379`.
  - **Why not later.** Loosening a validator-checked rule after S15 is as major as tightening it
    (§1.10), so deferring to `bindParts` would have made these forms unbindable for all of v1.
  - **The scheme list is closed.** A further scheme is a major change, decided with F-7.
  - **S10 removes today's bare-IPv6 branch** and implements `ipv6` as above, bare and bracketed.

- **A single-label `host` contains at least one letter.** `99999` is neither a port nor a host.
- **`ipv6` is the RFC 5952 canonical text.** It is lowercase, has no `.` (so no embedded IPv4) and
  no `%` (so no zone index), and equals what Python 3.12's `ipaddress` writes for it. This closes
  S10-M1: `":"`, `"abc:"` and `"deadbeefcafe0123:9876543210abcdef"` fail. The two forbidden
  characters keep Python versions that print IPv4-mapped addresses differently in agreement.
- **IP literals of every class pass**, and so do internal DNS names: the stakeholder decision of
  2026-10-29 (SA-13).
- **The validator re-checks every emitted `bindValue` against this grammar**, the screen included.

#### 1.7.3 The name rule (replaces (b)) [pre-freeze]

**Folding.** `fold(x)` takes NFKD, drops every combining mark (category `Mn`), then takes NFKC and
Python 3.12's `casefold()`. Dropping the marks makes a bare `PRİVATE` (U+0130) fold to `private`
(P2-5); in v1.0 it failed only with a `key` after it (BS-m4).

An entry fails the name rule when any **named** `keyPath` segment (an `[i]` index is not one)
matches:
- **as a substring of the folded segment**, any of 25 words:
  - the spec's 15: `pass`, `pwd`, `secret`, `token`, `key`, `cred`, `auth`, `cookie`, `session`,
    `private`, `cert`, `dsn`, `conn`, `hook`, `sig`;
  - plus 10 moved from v1.0's whole-token list (P2-5): `pw`, `psk`, `jwt`, `hmac`, `totp`,
    `mnemonic`, `bearer`, `gpg`, `pgp`, `kms`;
- **as a whole folded token**, any of 11 words: `salt`, `pepper`, `seed`, `pin`, `otp`, `pat`,
  `license`, `nonce`, `encrypt`, `ssh`, `vault`.

The total stays 36 words. The 10 moved words almost never occur inside English identifiers, so
`cfg.adminpw` now fails. The over-redaction is accepted: `upward_limit` fails on `pw`. The 11 words
kept as whole tokens would otherwise hit `path`, `ping`, `shopping` and `asphalt`, so `cfg.rootpin`
and `githubpat` remain accepted residuals.

**Tokens.** Split the segment **before folding**:
- on `_`, `-` and `.`;
- at every lowercase-to-uppercase boundary, and before the last capital of a run of capitals that
  is followed by a lowercase letter (`HTTPServer` gives `HTTP` and `Server`);
- **(v1.1, BS-m4) at every letter↔digit boundary**, so `ADMIN_PIN1`, `PIN1` and `GithubPat2`
  fail.

Then fold each token.

**Where the rule applies.** It judges `launchFact.envVar` (§1.6) and every argv name (§1.6.1
(v3)), each as one segment, except that an assignment name is split at `.` first. The validator
checks it on `configEntry`, on `envVar` when `default` is present, and on every argv context.

#### 1.7.4 Argv exclusion and writer conditions

- **W1 [pre-freeze], argv arrays never bind.** `bindValue` is absent when a named `keyPath` segment
  `args`, `command` or `entrypoint` is directly followed by an `[i]` index
  (`services.app.command[2]`, `mcpServers.gh.args[3]`). Those values belong to `launchFact`, under
  the argv gate. The validator checks W1.
- **W2 [writer], single-label hosts are declared services.** A single-label `host` other than
  `localhost`, bare, inside a `url` **or inside a `hostport`** (v1.2, P3-4), is emitted only when
  the label is a Compose service name or
  `container_name` declared in a Compose file of the same repository at `builtCommit`
  (`FALKORDB_HOST: falkordb`, `falkor-chat/compose.yaml`). **W2 also governs `launchFact.default`**
  (v1.1, P2-3, BS-m9): `${FALKORCHAT_WS_ID:-acme}` emits no `default` unless `acme` is a declared
  service. That matters more than for `bindValue`, because `default` is renderable (§1.6.2). W2
  does not govern argv (v6 governs it).
- **W3 [writer], no interpolation, and aliases are judged at both ends.** The normalizer never
  resolves `${…}`, `$NAME` or `env_file:` for `bindValue`, and the grammar has no `$`. A YAML alias
  or merge-key value is bindable only if **both** the alias's `keyPath` and the anchor's `keyPath`
  pass the name rule.
- **W4 [writer], withholding is always allowed.** A writer may omit any `bindValue` or `default`
  (§1.10).

#### 1.7.5 `valueHash` and `defaultHash` [pre-freeze: field semantics]

```
valueHash = lowercase-hex( HMAC-SHA-256( key, UTF-8(id) || 0x00 || UTF-8(NFC(text)) ) )
```

- `id` is the record's own 32-hex id. `text` is the value as the format's own unescaping yields it:
  a quoted scalar's content, or a bare scalar as written (YAML `8200` gives `8200`). `key` is
  §5.17's system key (`~/.config/ccg/<systemId>.hmac`).
- `launchFact.defaultHash` is computed the same way, over the default's text, with the
  `launchFact` record's id.
- **The hash is domain-separated per entry**, so equal values in two entries give different hashes
  (SA-11). Per-entry change detection still works, because `entryId` is stable across snapshots.
- **No consumer matches hashes (v1.1).** graph-dba's ruling closes F-2: no graph query, linker or
  binding compares `valueHash` across entries. Binding uses clear fields and env-var **names**.
  So the assembler needs the key only to detect change. It never needs it to bind (BS-m1, G-2,
  P2-4).
- **A future cross-entry hash is a minor bump, and bundle-only.** If a future finding needs one
  (for example "production reuses the example secret"), it is an optional field. It **stays in
  the bundle and never becomes a graph property**, because every raw-query reader (FR-16, FR-25)
  would then hold the oracle.
- The validator cannot recompute the hash, because it holds no key.

#### 1.7.6 Gate vectors (normative; SA-12)

Every row was reproduced on 2026-10-29 by a throwaway emulation of §1.7.1–§1.7.4 (session
scratchpad, not a repo file), and again on 2026-10-30 under v1.1's screen, grammar and name rule,
with no outcome changed. "Validator" marks a row that S10's reader decides alone. A must-bind row
validates, and a must-redact row that carries a `bindValue` is rejected.

The last column lists **at least** one deciding rule. Several rows fail more than one (BS-n2): BV-7
also fails T2 and T3; BV-10 also T2 and T3; BV-13 and BV-14 also T2; BV-15 also T1 and T5; BV-16
also T5; BV-20 also T2 and T3. S10's tests assert the outcome, not the set of failing rules.

v1.1 adds these vectors:

| # | `keyPath` | value | outcome | decided by (at least) |
|---|---|---|---|---|
| BV-21 | `services.x.environment.FALKORDB_URL` | `redis://falkordb:6379` | **bind** | validator; W2 |
| BV-22 | `x.addr` | `localhost:8000` | **bind** | validator |
| BV-23 | `x.addr` | `::1` | **bind** | validator |
| BV-24 | `notify.url` | `https://ntfy.sh/x7k2-q9w4-m1p8-z3v6` | redact | validator: T6 |
| BV-25 | `cfg.adminpw` | `localhost` | redact | validator: name (`pw`) |
| BV-26 | `x.ADMIN_PIN1` | `4821` | redact | validator: name (`pin`, digit split) |
| BV-27 | `x.db_host` | `mydb.c9akciq32.us-east-1.rds.amazonaws.com:5432` | **bind** | validator (screened per label, v1.2) |
| BV-28 | `x.url` | `https://x7k2.q9w4.m1p8.z3v6.example.com/` | **bind** (accepted residual) | validator (screened per label, v1.2) |

| # | `keyPath` (format) | value | outcome | decided by (at least) |
|---|---|---|---|---|
| BV-1 | `mcpServers.falkor-chat-agent-team.url` (mcp-json) | `http://localhost:8200/mcp` | **bind** | validator |
| BV-2 | `services.server.environment.FALKORDB_HOST` (compose) | `falkordb` | **bind** | validator admits it; writer W2 (a declared service) |
| BV-3 | `mcpServers.cypher.env.FALKORDB_PORT` (mcp-json) | `6379` | **bind** | validator |
| BV-4 | `mcpServers.cypher.env.FALKORDB_HOST` (mcp-json) | `host.docker.internal` | **bind** | validator |
| BV-5 | `x.url` | `http://localhost:8000/mcp/` | **bind** | validator |
| BV-6 | `notify.webhook` | `https://hooks.slack.com/services/T0/B0/XXXX` | redact | validator: name (`hook`), grammar (multi-segment) |
| BV-7 | `datadog.api` | `3f2a9c1e5b7d4a6f8e0c2b4d6a8f0e1c` | redact | validator: T1 |
| BV-8 | `heroku.apiid` | `550e8400-e29b-41d4-a716-446655440000` | redact | validator: T3, T4 |
| BV-9 | `DB_PW` (env) | `correcthorsebatterystaple` | redact | validator: name (`pw`), T2 |
| BV-10 | `services.app.command[2]` (compose) | `9f86d081884c7d659a2feaa0c55ad015` | redact | validator: W1, T1 |
| BV-11 | `ADMIN_PIN` (env) | `4821` | redact | validator: name (`pin`) |
| BV-12 | `notify.url` | `https://ntfy.sh/x7k2q9w4m1p8z3v6` | redact | validator: T1 on the segment |
| BV-13 | `fn.url` | `https://abcdefghij0123456789klmnopqrstuv.lambda-url.us-east-1.on.aws/` | redact | validator: T1 on a host label |
| BV-14 | `x.url` | `https://ntfy.sh/abcdef0123456789xyzqwertyuiopas` | redact | validator: T1 |
| BV-15 | `auth_header_example.jwt` | `eyJhbGciOiJIUzI1NiJ9.eyJzdWIiOiIxMjM0In0.abc-def` | redact | validator: name (`auth`, `jwt`), grammar (upper case) |
| BV-16 | `aws.id` | `AKIAIOSFODNN7EXAMPLE` | redact | validator: grammar (upper case), T1 |
| BV-17 | `DATABASE_URL` (env) | `postgres://u:p@db/app` | redact | validator: grammar |
| BV-18 | `x.url` | `http://localhost:8000\abc` | redact | validator: grammar |
| BV-19 | `x.url` | `http://LOCALHOST:8000/mcp` | redact | validator: grammar (accepted over-redaction) |
| BV-20 | `monitor.endpoint_id`, a YAML alias of `deploy.token` | 32 hex | redact | validator: T1; writer W3 also |

**Residual risk, accepted for P1** (the advisory's own irreducible case): a word-only capability
under a neutral key passes, for example `https://ntfy.sh/deploy-alerts-team`. Only keyed binding
parts remove it. That is SA-4(b), deferred (§2).

**Accepted residuals added in v1.2:**
- **A letters-only random token of 16–23 characters** passes: `https://ntfy.sh/kxqzmwvptryhbnjdlc`
  binds. T1 needs a digit, T2 needs 24 characters, and T6 needs transitions (P3-3). F-7's
  `bindParts` removes it for `bindValue`, and S17's G-1 choice removes it for argv.
- **A chunked token spread across host labels** binds (BV-28; §1.7.1's scope rule).
- **A `hostport` or URL whose db index, dbname or vhost holds a value**, such as
  `redis://localhost:6379/hunter2`. No protocol carries a credential there (security Pass 3's
  probes).

**Accepted residuals added in v1.1:**
- **A base64 secret split by `/` into short letter-only pieces** can pass the screen piece by
  piece, in a URL path and therefore in step 8a. The example is
  `/wJalrXUtnFEMI/K7MDENG/bPxRfiCYEXAMPLEKEY`. The shape is rare, and recording it is P2-7's
  recommendation.
- **The concatenated whole-token names** `rootpin` and `githubpat` (P2-5).

### 1.8 `http` `keyNorm`: step 8a (inserted after spec §6 `http` step 8) [pre-freeze: id-bearing]

**(8a)** After step 8, each remaining **literal** segment that fails the token screen (§1.7.1) is
replaced: by `{?}` in a consumer key, and by **`{#}`** in a provider key (v1.1; v1.0 used `{}`).
Placeholders are not screened.

**`{#}` (v1.1, BS-m7)** is a provider placeholder meaning "a withheld literal". It matches only a
consumer `{?}`, and the match is always `possible`/`partial-key`. A consumer literal that passes the
screen never matches it, because the same literal on the provider side would have passed too.
v1.0's `{}` made the masked route a path parameter, so **every other** consumer literal in that
position matched it, presumably as `confirmed`. That made the bullet below true only for the masked
pair.

- **Why before the freeze.** `keyNorm` is in the `cpId` tuple, and it is a readable graph property
  (`ContractPoint.key`/`keyNorm`). So a code literal such as a webhook path reaches every answer
  that shows the contract point (SA-9(a)). A later fix would re-id every affected point.
- **Effect on linking.** A consumer `{?}` is a partial key (`possible`/`partial-key`, spec §6,
  step 8), and a provider `{#}` matches only `{?}`. So a masked segment links only as `possible`,
  and it never widens a provider's match set.
- **Departure from the advisory:** providers are masked too. A provider route literal can itself
  be a capability (`@app.post("/hooks/<token>")`), and `{#}` keeps a masked provider matchable by a
  masked consumer.
- **No golden vector moves.** SL-1's and SL-2's keys are unchanged (`shop`, `api`, `health` and
  `x` pass). GV-9 is unchanged. GV-10 is a hash vector of a given tuple, not a `keyNorm`
  derivation.
- **Example.** A consumer call to
  `https://hooks.slack.com/services/T024BE7LD/B01ABCDEFGH/EXAMPLEexampleEXAMPLEexample0000` gives
  `POST /services/T024BE7LD/B01ABCDEFGH/{?}`, the last segment failing T2. The advisory's literal
  `T0/B0/XXXX` is a placeholder and passes.

### 1.9 Locator types (adds two rows to spec §3.5's table) [doc]

| Anchor | Locator | Id it converts to |
|---|---|---|
| a configuration entry | `{repoId, path, keyPath}` | `entryId` = `h(["configEntry", repoId, path, keyPath])`, with `keyPath` by §5.17's grammar |
| a unit | `{repoId, rootPath, lang}` | `unitId` = `h(["unit", repoId, rootPath, lang])` |

Where each is used, and at which `keyPath` grain, is the eval protocol's
(`docs/plans/connected-code-graph-eval-protocol-ml.md` §2.2, §2.4). A `launchFact` has no locator in
v1: its `ordinal` cannot be written without counting.

### 1.10 Determinism and versioning (amends spec §8.1, and spec §8.6's "who writes a change" bullet)

- **§8.1 [doc].** "Arrays are sorted where a field says so (`targets`, `values`, `args`, …)": the
  `args` there is `decorator.args`. `launchFact.args` is in argv order (§1.6).
- **§8.6, validator versus writer [pre-freeze: a rule of the versioning policy].** A `1.n` validator
  enforces the rules this spec and the amendments here mark validator-checked. A writer may always
  withhold more than those rules require: it may omit an optional `bindValue` or `default`, or write
  an argv element as its sentinel. A withheld context supplier takes the **forcing** sentinel
  (§1.6.1, v1.2). **Tightening a writer-only rule is not a schema change and
  needs no version bump.** Tightening a validator-checked rule rejects bundles that were valid, so it
  is a major change. **Loosening one is major too (v1.1, BS-m2):** an older validator rejects the
  newly admitted bundles. That is why §1.6.1, §1.7.1–§1.7.3 and W1 land before the freeze, in
  both directions.
- **§8.6, "who writes a change" (v1.1, BS-m6).** The bullet says a minor or major version is
  written into a successor spec under rule 5. Before the freeze, this document is that successor,
  and `-spec3` consolidates it (§0). After the freeze, the bullet stands unchanged.

### 1.11 Mapping to the graph (amends spec §9.1) [pre-freeze for S17]

| Record | Change |
|---|---|
| `controlledBy` | `(point)-[:CONTROLLED_BY]->(:Cond)`, where `point` may be a `Cond` (§1.4). The edge-load anchors include `Cond` as a source. |
| `launchFact` | `defaultHash` is **not carried** and never rendered. `ordinal` is not carried, because it is in the id. **Routed to S17 (G-1):** spec §9.1 carries only `path`/`line`, so "may be rendered" (§1.6.2) has nothing to render from the graph. S17 either carries the gated fields on the `launchFact`-origin `ConfigEntry` (`launchKind`, `command`, `args`, `port`, `protocol`, `envVar`, `default`, `mountPath`, `service`), or records that binding evidence renders as file and line only. That is a graph-side decision with no bundle change. |
| `configEntry` | Unchanged: `bindValue`, `format` and `example` stay out of the graph. **`bindValue` is matching input only, and it is never rendered** in an answer (v1.1, P2-6). The SA-3, SA-4(b) and SA-7 deferrals rest on this. Until F-7, a renderer shows the provider's mount path, never a consumer's config URL. `ConfigEntry.valueHash` now has §1.7.5's per-entry semantics. **Routed to S17 (P2-8):** store the `ConfigEntry.keyPath` **property** with each screen-failing segment replaced by `[#…]`. `entryId` stays computed from the raw `keyPath`, so nothing re-ids, and raw queries are covered where F-4's render masking is not. |
| `Param`, `Ret`, `Cond` | **Routed to S17 (S-1):** `ddl_v1.cypher` creates `pointId` range indexes on all three **before** the load, and the post-load gate adds `UNIQUE NODE … PROPERTIES 1 pointId`. `Cond` is now both ends of an edge group. |
| `import` | Unchanged (no node). `ordinal` is in the id. |

### 1.12 Validity: the schema versus `bundle.py` (adds to spec §11's S10 row) [pre-freeze]

**A bundle is valid iff it satisfies the spec as amended here** (v1.1, BS-m8; v1.0 made
`bundle.py` normative, bugs included). `cpg/ccg/bundle.py` is the **reference validator**, and a
disagreement between it and the spec is a `bundle.py` defect. `bundle-v1.schema.json` is
**necessary, not sufficient**. It must never reject a spec-valid bundle. It checks
shapes and per-record field rules, as far as JSON Schema's ECMA-262 pattern subset can express them
without lookaround. `bundle.py` adds:
- every id recompute and every cross-record rule;
- the one-encoding rule for quoted keys (§1.1);
- `nameNorm` (§1.2);
- the canonical-IPv6 check, the token screen, the name rule and W1 (§1.7), wherever the schema
  cannot express them;
- the argv rule over the emitted sequence (§1.6.1).

The schema's top-level `description` and `cpg/README.md` say this. A schema-only consumer is
therefore not a validator. This answers S10's "one-encoding rule lives only in `bundle.py`" nit:
that placement is correct, and it must be declared.

### 1.13 Provisional status (amends spec §10)

- **[P-slice], added:** `nameNorm`'s recomputation (§1.2), and step 8a on the slice's keys (§1.8).
  SL-1 and SL-2 must come out unchanged.
- **[P-later], restated:** `configEntry.bindValue` and `launchFact` freeze as amended here, untested.
  P1 is their first writer.

## 2. Dispositions

"BV" is `docs/reviews/connected-code-graph-bindvalue.md` Pass 1. "S10" is
`docs/reviews/connected-code-graph-p0-impl.md` § `## S10 (P0-09)`. "EP" is
`docs/plans/connected-code-graph-eval-protocol-ml.md` v1.1. The follow-up column names the
implementer and the change; §4 groups them by step.

**How the stakeholder decision of 2026-10-29 was applied.** FR-15 covers the graph and answers, not
the local bundle files. So none of the bundle-side findings is an FR-15 violation, and severities
are read on the graph/answer path. SA-1 is still adopted in its writer-side form, for a design
reason, not a compliance one. A single gate that the validator re-checks makes "only gated values
can be rendered" a property of the record, not a duty of every renderer, and it matches what BF8
already chose for `configEntry`.

| Item | Source | Decision | Landed | Implementer follow-up |
|---|---|---|---|---|
| SA-1 `launchFact` default/args in clear | BV SA-1 | **Adopt now** (pre-freeze), in the advisory's writer-side form. Adds `defaultHash`, a conditional `default`, the argv gate and `"<redacted>"`. Also the §9.1 rendering statement. | §1.6, §1.6.1, §1.6.2, §1.11 | **S10** (`tdd-engineer`): schema and `bundle.py`, `default` iff the gate passes, `defaultHash` required, each `args`/`command` element passes §1.6.1 or is its sentinel, AV-1–AV-68 and §1.6.3's probe as tests (v1.3). **P1 writer:** emits it. |
| SA-2 open host class | BV SA-2 | **Adopt now.** The shapes are validator-checked. The declared-service condition is a writer rule (W2). | §1.7.2, §1.7.4 W2 | **S10:** grammar. **P1 writer:** W2. |
| SA-3 role allow-list; positional names | BV SA-3 | **Split.** The argv-array exclusion is **adopted now** (W1, validator-checked). The role allow-list and the per-format `keyPath` derivation table are **adopted later** under §8.6, as a writer-only rule (§1.10), because they need P1's config-extractor design for Dockerfile, Kiro and Compose list forms. | §1.7.4 W1; later: P1 | **S10:** W1. **`architect`, P1 plan:** the role rule and the `keyPath` table, before P1 relies on `bindValue` (F-6). |
| SA-4 capability URLs | BV SA-4 | **(a) Adopt now** (the token screen), with two false-positive departures. **(b) Adopt later** under §8.6, as an additive optional `bindParts` field (a minor bump). | §1.7.1; later: P1 | **S10:** the screen. **`architect`, P1:** `bindParts` (F-7). |
| SA-5 normalization and grammar | BV SA-5 | **Adopt now**: a closed, case-sensitive grammar on the emitted string, re-checked by the validator. | §1.7.2 | **S10:** replaces the current classes; closes S10-M1. |
| SA-6 name-list misses | BV SA-6 | **Adopt now**: substring words kept, whole-token words added, NFKC plus `casefold`. | §1.7.3 | **S10:** the rule, plus a probe over all 36 words. |
| SA-7 numeric secrets | BV SA-7 | **Adopt now in part.** `pin`, `otp` and `totp` are whole-token names, which closes BV-11. The full fix rides SA-3's role rule (P1). | §1.7.3 | as SA-3 and SA-6 |
| SA-8 aliases and interpolation | BV SA-8 | **Adopt now**, as a writer rule (W3). The grammar already excludes `$`. | §1.7.4 W3 | **P1 writer.** |
| SA-9(a) code literals in `http` `keyNorm` | BV SA-9 | **Adopt now** (id-bearing). Departure: provider literals are masked too, to `{#}` (v1.1; v1.0's `{}` was a wildcard, BS-m7). | §1.8 | **S12** (`coder`): step 8a in `assemble.py`. **`data-scientist`:** EP §13.4's `normalize` applies 8a to key items (F-8). |
| SA-9(b) `Cond.code` literals | BV SA-9 | **Adopt later**, as a writer-only rule: no field changes, so no bump (§1.10). The slice graph has no LLM consumer. | P1 normalizer | **`graph-dba`, P1:** mask failing string literals in `cond.code` as `"…"` (F-9). |
| SA-10 secret in `keyPath` | BV SA-10 | **Reject the id-bearing form.** HMAC'ing a segment would make `config` locators unwritable from source: EP §2.2, M2's premise. It would also re-id entries, for an uncommon shape. **Mitigation adopted later** at render time, with no id change (F-4). | — | **`architect`, S33:** render `keyPath` and `keyNorm` through the token screen (F-4). |
| SA-11 `valueHash` equality oracle | BV SA-11 | **Adopt now**: per-entry domain separation. Decided by reversibility: equality can be added later as a minor field, but removing it after the freeze is major. | §1.7.5 | **P1 writer.** S10: description only. |
| SA-12 gate vectors | BV SA-12 | **Adopt now**: BV-1–BV-20 are normative, and AV-1–AV-5 are added. v1.1 extends them to BV-26 and AV-24. | §1.7.6, §1.6.1 | **S10:** every BV row and AV row as a test. |
| SA-13 internal topology | BV SA-13 | **Reject**, per the stakeholder decision of 2026-10-29: internal hostnames, and IP literals, are acceptable in clear. | §1.7.2 | — |
| BV Q1: FR-15 and the bundle | BV open question 1 | **Answered** by the stakeholder: FR-15 covers graph and answers, not bundle files. | §2 preamble | — |
| BV Q2: bundle at rest | BV open question 2 | **Follow-up.** | §3 F-1 | **`devops`** (F-1) |
| BV Q3: cross-entry `valueHash` equality | BV open question 3 | **Closed at v1.1** by graph-dba's ruling: no equality; any future cross-entry hash stays bundle-only. | §1.7.5 | **`graph-dba`** with `architect` (F-2) |
| BV Q4: hostnames in LLM context | BV open question 4 | **Answered** (SA-13). | — | — |
| BV Q5: prompt injection via free text | BV open question 5 | **Follow-up.** | §3 F-3 | **`architect`** (S33) and **`cobb`** (F-3) |
| §5.1 field-path form | S10, open item 1 | **Ruled: the no-dot form `a1["x"]`.** The grammar line was the typo. | §1.1 | **S10:** none for the form (already enforced). |
| S10-m3.1 nesting depth | S10-m3 | **Ruled: the leaf counts**, so at most three containers. | §1.3 | **S10:** none; a boundary test if one is missing. |
| S10-m3.2 `controlledBy.point` = `cond` | S10-m3 | **Ruled: allowed**, with `point ≠ cond`. | §1.4, §1.11 | **S10:** keep accepting it; reject `point == cond`. **S11:** emit the CDG's cond→cond. **S17:** `Cond` as a `CONTROLLED_BY` source. |
| S10-m3.3 repeated import names | S10-m3 | **Ruled: the normalizer de-duplicates**, and the sort stays strict. | §1.5 | **S10:** none. **S11:** de-duplicate. |
| S10-m3.4 `nameNorm` | S10-m3 | **Ruled: `NFC(casefold(NFC(name))).strip()`**, Python 3.12 `casefold`. | §1.2 | **S10:** recompute and reject a mismatch. **S11:** emit by the formula. |
| S10-m3.5 `IDENT` | S10-m3 | **Ruled: ASCII** `[A-Za-z_$][A-Za-z0-9_$]*`. The `kw` head is `kw` plus one field. | §1.1 | **S10:** `_IDENT` becomes ASCII; the `kw` arm takes the bracket form under the one-encoding rule; tests for `a1.café` (rejected), `a1["café"]` and `kw["café"]` (accepted), `kw["prefix"]` (rejected). **S11:** quote non-ASCII keys. |
| Missing `ordinal` on `import`/`launchFact` | S10, open item 2 | **Ruled: added as a required field.** The density and line checks stay. | §1.5, §1.6 | **S10:** required in the schema; ids recomputed from the field. **S11:** emit it. |
| Quoted-identifier rule: schema vs `bundle.py` | S10 nit 1 | **Ruled: `bundle.py` is the validator, and the schema is necessary, not sufficient.** Declared in both places. | §1.12 | **S10:** the schema `description` and `cpg/README.md`. |
| `bindValue` IPv6 widening | S10-M1 | **Superseded by the grammar.** S10's in-flight stricter IP check should match §1.7.2's `ipv6` rule. | §1.7.2 | **S10:** align the in-flight fix to RFC 5952 canonical form, no `.`, no `%`. |
| Locator types `config` and `unit` | EP §15 Q2 (F18 in the successor plan) | **Adopted.** | §1.9 | none. EP §2.2 already uses them. |

## 3. Follow-ups with owners

| # | Follow-up | Owner | When | Default until decided |
|---|---|---|---|---|
| F-1 | **Bundle-at-rest handling** (BV Q2; stakeholder decision 2026-10-29): file mode, work-area location, retention, and whether any LLM-backed tool ever reads a bundle directly. The bundle may hold source literals in clear (`valueSet`/`constant`), which FR-15 does not forbid there. | `devops` | before P1's first non-proving-repo build | P0 bundles live only in the gitignored work area |
| F-2 | **Closed (v1.1):** no consumer needs cross-entry `valueHash` equality (graph-dba ruling). A future cross-entry hash is a minor, bundle-only field (§1.7.5). S17 records the ruling. | `graph-dba` (S17) | S17 records it | no equality |
| F-3 | **Prompt injection through free text from source** (BV Q5): `keyPath`, `Cond.code`, `calleeName`, `qname`, emitted `launchFact` args. Fence them as data wherever an LLM reads them. | `architect` (S33, the answer spec) for rendering; `cobb` for the headless agents' prompts | S33; S52 | — |
| F-4 | **Render free-text identifiers through the token screen** (SA-10 mitigation): `ConfigEntry.keyPath`, a `config_key`/`http` key, and `launchFact.mountPath`/`service`/`envVar` shown in an answer (BS-m3). The store-time half is P2-8, routed to S17 (§1.11). | `architect` (S33) | S33 | — |
| F-5 | **`-spec3`, a required conformance item (§0).** One gated document that supersedes the spec and this one. **P0 exit requires it.** | `architect`; `teco` checks it at P0 exit | S16 if it runs; otherwise before P0 exit | read both |
| F-6 | **SA-3 role allow-list and per-format `keyPath` derivation** (Dockerfile, Kiro agent, Compose list forms), writer-only. | `architect` | P1 plan, before P1 relies on `bindValue` | W1 plus the name rule |
| F-7 | **SA-4(b) keyed binding parts** (`bindParts`), an additive optional field, a minor bump. | `architect` | P1 | clear `bindValue` under §1.7 |
| F-8 | **Step 8a in the eval protocol's key normalizer** (EP §13.4), so writers are never asked to screen. | `data-scientist` | the next EP revision (S06 v1.2 hand-offs) | — |
| F-9 | **SA-9(b) `Cond.code` literal masking** in the P1 normalizer. | `graph-dba` | **before P1's first build of a non-proving repo** (P2 Pass 2 condition) | verbatim, as in the spec |
| F-10 | **P1 writer obligations** (BS-m9): W2 (with `default`), W3, W4, the writer side of SA-1, §1.6.1's argv rule, G-4's self-edge drop, and the per-entry HMAC (§1.7.5). These go into the P1 plan as one writer done-condition. | `architect` (P1 plan) | P1 plan | — |
| F-11 | **S17's graph-side items:** G-1 (carry the gated `launchFact` fields, or render file and line only; security P3-1 recommends carrying `port`, `mountPath`, `service`, argv[0] and the role-relevant pairs only, never the whole of `args`, which closes §1.6.1's named residuals), P2-8 (screened `keyPath` property), S-1 (`pointId` indexes on `Param`/`Ret`/`Cond`), and F-2's ruling (§1.11). | `graph-dba` (S17) | S17 | — |
| F-12 | **Cycle-safe traversal over `CONTROLLED_BY`** (G-3): a visited set seeded with the anchor. | `architect` (S20/S33 note), P1 BFS implementer | S20; P1 | — |
| F-13 | *2026-10-30, routing row added after acceptance; the normative text lands in `-spec3`.* **The `http` `keyNorm` step order is to be reordered to RFC 3986 §6.2.2's order:** spec §6's step 6 (percent-decode the unreserved characters, uppercase the other escapes) runs **before** step 5 (collapse `/` runs, resolve `.`/`..`). The current order is not idempotent: `GET /a/%2E%2E/b` → `GET /a/../b` → `GET /b`. `keyNorm` is in `cpId`, so the change is pre-freeze. Step 8a is unaffected; it still runs after step 8. Source: `docs/plans/connected-code-graph-eval-protocol2-ml.md` §4 Q3, and its review's Pass 2 P2-m3. **No step may encode the old order:** S12's `http` rule and the shared 8a function, S35/S39 contract keys, the S40/S52 writer briefs, and the eval normalizer (K42 expects `GET /b`). | `architect` (`-spec3`, S16 or before P0 exit); `teco` makes S15's record list F-13 as part of the frozen contract if `-spec3` is not yet written | before S15's freeze record | RFC order, as above. The old order is never implemented. |

## 4. Downstream obligations

### 4.1 Implementer follow-ups, by step

- **S10, `tdd-engineer`.** One follow-up round after its current fix round, **before S15**. The
  in-flight IP fix should target §1.7.2's `ipv6` rule directly.
  1. `import` and `launchFact` gain a required `ordinal`; ids recomputed from it (§1.5, §1.6).
  2. `launchFact`: `defaultHash` required for `env-default`; `default` conditional; **the v1.1
     argv rule over the emitted sequence, as of v1.3**:
     - element classes, with v1.3's widened class-2 allow-list and class 4's two readings;
     - forcing versus non-forcing sentinels, and the context-word bare values;
     - (v1)–(v8), with (v1)'s bracket rule, (v2)'s per-production screen, (v7)'s six tokens and
       (v8)'s `NAME:secret` check;
     - the forcing rule.

     Tests: **§1.6.3's probe verbatim**:
     - 17,280 grid cells plus 6,480 separator-axis cells, zero values in clear;
     - the 675-cell transplant;
     - AV-1–AV-68 and BV-1–BV-28 **parsed from this document's text** (§1.6).
  3. `bindValue`: the v1.1 grammar (`hostport`, bare `ipv6`, the closed scheme list), the screen
     with **T6**, using v1.2's widened class and §1.7.1's per-production scope, the v1.1 name rule
     (NFKD mark-stripping, 25 substring and 11 whole-token words, the letter↔digit split) and W1.
     These replace today's classes, **including the removal of today's bare-IPv6 branch**. Tests:
     BV-1–BV-28, plus the SA-6 probe over all 36 words with the suffix axis `""`, `2`, `_2`, `1`
     (BS-m4, BS-m9).
  4. `IDENT` becomes ASCII, and the `kw` head takes `kw` plus one field (§1.1).
  5. `nameNorm` recompute (§1.2).
  6. `controlledBy` rejects `point == cond` (§1.4).
  7. ~~The validity statement~~: **already done**. The schema's `description` and `cpg/README.md`
     say "necessary but not sufficient" (BS-m9). Only §1.12's v1.1 wording, "the spec is
     normative; `bundle.py` is the reference", needs a README touch.
  8. **The spec-derived tests read this document too.** `import` (§1.5) and `launchFact` (§1.6) are
     now tables, so `launchFact` moves from the prose transcription to the table parser. Amended
     clauses are read from here.
- **S11, `graph-dba`.** Emit:
  - `nameNorm` by §1.2;
  - `import.ordinal`, with repeated names de-duplicated (§1.5);
  - cond→cond `controlledBy` from the CDG (§1.4);
  - ASCII-`IDENT` quoting of field keys (§1.1);
  - **G-4:** drop CDG self-edges, and de-duplicate (`point`, `cond`) pairs, before writing
    (§1.4). Verify against the slice CPG.

  `launchFact` and `configEntry` are [P-later], so the slice does not emit them.
- **S12, `coder`.** The `http` `keyNorm` step 8a in `assemble.py`, with T6 and the provider `{#}`
  placeholder, which matches only `{?}` and is always `possible` (§1.8). SL-1 and SL-2 must come
  out unchanged.
- **S15, `analyst` + `graph-dba`.** The freeze covers the spec **as amended here**. The gate checks
  that every [pre-freeze] item of §1 is encoded by S10, and the "frozen, untested" list includes
  §1.6 and §1.7.
- **S17, `graph-dba`.**
  - `Cond` as a `CONTROLLED_BY` source; `defaultHash` not carried (§1.11);
  - F-2 recorded as closed (no equality; any future cross-entry hash is bundle-only);
  - G-1's disposition;
  - P2-8's screened `keyPath` property;
  - S-1's `pointId` indexes (F-11, S-2).
- **S33, `architect`.** G-1's rendering consequence (file and line only, unless S17 carries the
  fields), F-3, F-4 and F-12.
- **S08, `devops`.** Nothing from this document.

### 4.2 Successor-plan done-conditions this changes

To be folded into `docs/plans/connected-code-graph-p0-2.md` when `teco` sends the §14 step. This
document does not edit that plan.

| Plan location | Change |
|---|---|
| §2 D5 | "successor `-spec2` via S16" becomes **`-spec3`**. The frozen contract is the bundle spec plus `-spec2`. |
| §2 D17, §3 S58 row, §5.3's S58 row, K-P0-16 | S58's dispositions are recorded **in `-spec2`** (§2 and §5 here), before S15, not "in S16's successor, else a §8.6 note". K-P0-16 is mitigated. (v1.0 said "§8's S58 row"; it is §5.3's, BS-m9.) |
| §4.0, the deliverable (b) row ("bundle spec (+ `-spec2` if changed)") | It becomes "bundle spec + `-spec2`; `-spec3` required before P0 exit" (BS-m9). |
| §9.2 P0 exit | Add **"`-spec3` exists, is gated, and `Supersedes:` the bundle spec and `-spec2`"** as a P0 exit item (§0, F-5). |
| §3 S03 row | The contract is the bundle spec v1.3 **plus** `-spec2` v1.1. |
| §3 S10 row | Add the §4.1 follow-up round, before S15, with its gate (analyst). |
| §3 S11 row | Add §4.1's emitter items. |
| §3 S12 row | Add step 8a. |
| §3 S15 row | "Freeze iff SL-1–SL-7" is unchanged. The record names the spec plus `-spec2` as the contract, and lists §1.6/§1.7 as "frozen, untested". |
| §3 S16 row | Output `docs/plans/connected-code-graph-bundle-spec3.md`, with `Supersedes:` the spec **and** `-spec2`. If S16 does not run, `architect` writes it before P0 exit (F-5). It is not optional. |
| §3 S11 row (in addition) | G-4 (self-edges and de-duplication). |
| §3 S17 row | Add F-11: §1.11's graph items, F-2's closure, G-1, P2-8 and S-1 (S-2). |
| §3 S33 row | Add F-3, F-4, F-12 and G-1's rendering consequence. |
| P1 plan (when written) | F-10's writer obligations, F-6, F-7 and F-9's deadline. |
| §15 F18 | **Resolved** by §1.9. |

## Ready to implement (summary)

Amendments: `docs/plans/connected-code-graph-bundle-spec2.md`, read with the bundle spec v1.3.
`bundleSchemaVersion` stays `1.0`.
- **S10 (before S15):** the eight items of §4.1, chiefly:
  - `ordinal` on `import`/`launchFact`;
  - `launchFact`'s argv gate and `defaultHash`;
  - the closed `bindValue` grammar (with `hostport`), screen (with T6) and name rule, with 28 BV
    vectors, 68 AV vectors and §1.6.3's probe (17,280 + 6,480 cells, plus the transplant) plus the
    SA-6 name probe;
  - ASCII `IDENT`;
  - the `nameNorm` check.
- **S11:** `nameNorm`, import `ordinal` and de-duplication, cond→cond without self-edges, ASCII
  quoting.
- **S12:** `http` step 8a with T6 and `{#}`.
- **S17:** F-11.
- **P0 exit:** `-spec3`.
- **No golden vector changes.**

## 5. v1.1 dispositions (re-gate checklist)

One row per finding. **Fixed** cites where it landed; **declined** gives the reason; **routed**
names the owner. "A" is analyst's § `## Spec amendments (bundle-spec2, egress-spec2)`; "G" is
§ `## Spec amendments — graph-dba check …`; "P2" is the advisory's `Pass 2 — dispositions check`.

| Finding | Disposition |
|---|---|
| A BS-M1 argv URL userinfo, capability URLs | **Fixed**, §1.6.1 (v4), (v5); AV-9, AV-10, AV-15, AV-16, AV-23. Departure: `@` is kept for package specs (`some-mcp@latest`), as A asked, and only `:`…`@` fails (see P2-1). |
| A BS-M2 `NAME=V` without a dash; dotted flags | **Fixed**, §1.6.1 classes 2 and 3 and (v3); AV-11–AV-14; the two-axis coverage probe (2,592 cells, none in clear). |
| A BS-m1 `defaultHash` and env binding | **Fixed**, §1.6 row, §1.7.5: change detection only, and env binding uses `envVar`. Key holder: the assembler needs the key only for change detection. |
| A BS-m2 `host:port`, non-web schemes, bare IPv6 | **Fixed** by admitting them before the freeze, §1.7.2. Loosening later is major, §1.10. Removal of the bare-IPv6 branch is in §4.1 item 3. |
| A BS-m3 §1.6.2's "each one passed a gate" | **Fixed**, §1.6.2 (gated versus ungated fields), and F-4 widened. |
| A BS-m4 digit-suffixed names; `PRİVATE` | **Fixed**, §1.7.3: letter↔digit split, and NFKD mark-stripping. The mark-stripping goes further than A's "leave as is", following P2-5. |
| A BS-m5 `(g3)` undefined cases | **Fixed**, §1.6.1: the emitted sequence decides; flag sentinels; flag after flag; AV-17, AV-18, AV-20. |
| A BS-m6 spec §1's S16 bullet amended unnamed | **Fixed**, §0 ("Required conformance item"), §1.10 (§8.6 bullet named). |
| A BS-m7 provider `{}` is a wildcard | **Fixed**, §1.8 `{#}`. |
| A BS-m8 `bundle.py` normative | **Fixed**, §1.12. |
| A BS-m9 follow-up gaps | **Fixed**, §4.1 items 3 and 7 (item 7 already done), F-10, W2 over `default` (§1.7.4), §4.2 rows for §4.0 and §5.3. |
| A BS-n1 AV-1's `…` | **Fixed**, §1.6.1 AV-1. |
| A BS-n2 incomplete "decided by" | **Fixed**, §1.7.6: the column is headed "at least", and the extra rules are listed. |
| A open question 1, rule 5 versus `Extends:` | **Fixed** for this document: `-spec3` is a required P0-exit item (§0, F-5). **Routed** for the convention: `teco` takes it to the convention owner. |
| A open question 2, `host:port` now or later | **Decided: now** (BS-m2). |
| G-1 `launchFact` render permission versus §9.1 | **Routed**, `graph-dba` S17 (§1.11, F-11), with S33's rendering consequence. |
| G-2 `defaultHash` wording | **Fixed**, §1.6 row. |
| G-3 cycles | **Fixed** as a stated property, §1.4. **Routed**: the visited set goes to the P1 BFS and S20/S33 (F-12). |
| G-4 CDG self-edges | **Fixed**, §1.4 writer bullet; S11 in §4.1. |
| G S-1 `pointId` indexes | **Routed**, `graph-dba` S17 (§1.11, F-11). |
| G S-2 follow-up completeness | **Fixed**, §4.1 (S11, S17, S33) and §4.2. |
| P2-1 argv credential forms (W5) | **Fixed**, §1.6.1, as one rule shared with BS-M1/BS-M2; AV-6–AV-12 adopted verbatim. **Declined in part, with reasons:** (a) **writer-only W5 is declined**: the gate was already validator-checked, so the fix lands before the freeze as a validator rule, which P2-1 also permits; (b) **"no `@` after the first character" is declined** in favour of "no `:` before an `@`". That keeps `some-mcp@latest`, which A asked to keep, and still fails every userinfo password. A user-only `name@host` is not a secret, and a token before `@` fails the screen. |
| P2-2 T6 chunked tokens | **Fixed**, §1.7.1 T6, **before the freeze, in the one shared screen**: it is id-bearing through step 8a, and a split screen would leave the validator weaker than the writer. BV-24. |
| P2-3 `default` skips W2 | **Fixed**, §1.6 row and §1.7.4 W2. |
| P2-4 `defaultHash` claim | **Fixed**, as BS-m1. |
| P2-5 name-rule edges | **Fixed**, §1.7.3: 10 words moved to the substring list, and marks stripped. The (g3) reading is §1.6.1's post-secret-flag rule. `rootpin` and `githubpat` are recorded as accepted residuals. |
| P2-6 `bindValue` never rendered | **Fixed**, §1.11 `configEntry` row. |
| P2-7 slash-split base64 | **Fixed** as an accepted residual, §1.7.6. |
| P2-8 SA-10 store-time masking | **Routed**, `graph-dba` S17 (§1.11, F-11). |
| P2 SA-9(b) condition (F-9 before the first non-proving-repo build) | **Fixed**, F-9's deadline. |

### 5.1 v1.2 dispositions

Sources are the analyst's Pass 2, "A2" (review § `## Spec amendments (bundle-spec2, egress-spec2)` → `### Pass 2`), and security's Pass 3, "P3" (advisory § `## Pass 3 — v1.1 check`).

| Finding | Disposition |
|---|---|
| A2 BS-P2-M1 = P3-1(a) and (d): a redacted context supplier drops the successor's context | **Fixed once**, §1.6.1: forcing versus non-forcing sentinels, class 4's two readings, and forcing that propagates through context suppliers. AV-25, AV-27, AV-32–AV-39. Probe §1.6.3: 17,280 cells, 0 in clear. **The trade-off:** the analyst's fix would change AV-6, AV-18 and AV-21. AV-6 and AV-18 are adopted as changed. AV-21 keeps `app.jar` through a closed single-dash allow-list, which is the analyst's own over-redaction-only option. **Declined:** P3-1(a)'s "write plain `<redacted>` when the long name passes", because a validator cannot check it. |
| P3-1(b) `--user admin:hunter2` | **Fixed**, §1.6.1 (v7); AV-28. |
| P3-1(c) a secret-named bare word before the value | **Fixed** for the 25 substring words (context-word values, forcing when redacted); AV-29–AV-31. **Residual:** the 11 whole-token words supply no context (`vault hunter2`), to keep AV-8. It is named and routed to S17 with G-1. |
| P3-1 (d) a secret flag directly after a secret flag | **Fixed**: forcing propagates through flag-shaped elements. AV-32–AV-35. |
| P3-1 unflagged positional secrets | **Residual**, already accepted, now named in §1.6.1. Routed to `graph-dba` S17 with G-1: P3-1's recommendation is to carry only role-relevant fields and never all of `args`. |
| A2 BS-P2-m1 the screen's scope in `bindValue` | **Fixed**, §1.7.1 scope rule: per label and segment, never the whole `bindValue`; the same in argv for address-shaped values. BV-27 (RDS) binds. BV-28 (labels chunking a token) is an accepted residual. |
| A2 BS-P2-m2 `[::1]:8000` in argv | **Fixed**, §1.6.1 (v1); AV-42. |
| A2 BS-P2-m3 (v6) and an assignment after `-e` | **Fixed**: (v6) fires on any single-character context name; AV-40 (redacted, over-redaction) and AV-41 (clear). |
| A2 accepted over-redactions to record | **Fixed**, §1.6.1's "Accepted over-redaction" list. |
| A2 stopping condition | **Fixed**, §1.6.3: the probe is normative for S10. Observed counts: 13,824 cells over the analyst's four prefixes and 17,280 over five, all with 0 in clear. The transplant gives 675 cells, with only the bare-positional residual in clear. |
| P3-2 `:`, `+`, `,` chunking | **Fixed**, §1.7.1 T6 class widened, in the shared screen (8a included, pre-freeze). Checked against `falkordb:6379`, `localhost:8000`, the RDS endpoint, `10.100.200.250:65535` and an IPv6 literal, which all pass by the scope rule. AV-43–AV-46. |
| P3-3 letters-only tokens of 16–23 characters | **Recorded** as a residual, §1.7.6 and §1.6.1. Removed later by F-7 and S17/G-1. |
| P3-4 W2 and `hostport` | **Fixed**, §1.7.4 W2. |
| P3 accepted residual `hunter2@db:6379` | **Recorded**, §1.6.1 named residuals. |
| P3 routing of P3-1's G-1 recommendation to `graph-dba` | **Routed**: S17, with F-11. |

### 5.2 v1.3 dispositions

The source is the analyst's Pass 3 (review § `## Spec amendments (bundle-spec2, egress-spec2)` → `### Pass 3`).

| Finding | Disposition |
|---|---|
| BS-P3-M1 `NAME:secret` in clear | **Fixed**, §1.6.1 (v8), adopted as proposed. It is checked against the address exemption: `falkordb:6379`, `localhost:8000`, `keycloak:8443` and `[::1]:8000` still pass (AV-65, AV-66); `falkorchat.app:app`, `X-Request-Id:abc` and `:8080` still pass. The separator axis is normative in §1.6.3: 6,480 cells, 0 in clear. AV-49–AV-55. |
| BS-P3-m1 stale AV-21 | **Fixed**, the row now reads "unchanged". **Root cause:** the v1.2 emulation did implement the allow-list, but my check printed its outputs and compared them with the rows by eye, and the row was never updated. **Process fix:** every row is now compared mechanically with expected values parsed from this file (68/68 AV, 28/28 BV), and §1.6.3 requires S10 to do the same. The analyst's `v12.py` agrees with my v1.3 emulation on all 48 v1.2 AV inputs. |
| BS-P3-m2 allow-list sized to AV-21 | **Fixed by widening now**, §1.6.1 class 2: Go role flags `-port`, `-addr`, `-host`, `-listen`, `-bind`; JVM `-ea`, `-da`, `-server`, `-client`, `-XX:[+-]Name`, `-verbose:{gc,class,jni,module}`. Each entry is shown to admit only what its `--NAME` form admits (15/15 identical), and the 71,280-cell entry-prefix grid has 0 in clear. AV-56–AV-61, AV-68. |
| BS-P3-m3 `--account admin:hunter2` | **Fixed by rule**, cheaply: (v7) gains `account`, `principal` and `identity` (AV-55). |
| BS-P3-m3 hex token as canonical IPv6 | **Recorded** as a named residual, §1.6.1 (AV-67): no hex rule separates it from a real address. |
| BS-P3-m3 internal addresses over-redacted after auth/key/session names | **Recorded** as accepted over-redaction, §1.6.1 (AV-62–AV-64): recovering them would re-admit values after secret-named names. |
| Pass 3 stopping verdict | **Met**: grid 17,280 and separator axis 6,480, with 0 in clear; transplant 675, with only the bare-positional residual; AV and BV reproduce from the file. The narrowed re-gate rule (a new class, sentinel kind or separator) is in §1.6.3. |
