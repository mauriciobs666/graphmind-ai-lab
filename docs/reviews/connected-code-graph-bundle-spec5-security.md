# Connected code graph — `bindParts` (`-bundle-spec5.md`) security advisory

> **Status:** active · **Owner:** `security-expert` · **Tracks:** C-1001 (P1)

## Pass 1

### Scope and verdict

- **Lens:** code/app security at design altitude (index `docs/plans/connected-code-graph-p1.md`
  N-ADV, step P1-13 gate), advisory to `architect`, who keeps final say.
- **Reviewed:** `docs/plans/connected-code-graph-bundle-spec5.md` (uncommitted, read in full)
  against its base text `-bundle-spec3.md` §1.7, §1.10–§1.12, §5.17, `-bundle-spec4.md`, the SA-4(b)
  design and threat model in `docs/reviews/connected-code-graph-bindvalue.md` (Pass 1 threat model,
  SA-4, SA-7, SA-11; Pass 2 P2-6), index D4, and P1-02/P1-07/P1-11/P1-14/P1-17/P1-20 in the P1
  sub-plans. One code fact was read from `cpg/ccg/bundle.py` at HEAD.
- **Questions asked:** what a repo-content writer (a malicious config file or URL) can cause, and
  what a reader of a bundle, answer, graph or audit can infer.
- **Method:** static reasoning only. Nothing was exploited, and nothing was written to any FalkorDB
  instance. I reproduced EH-1, EH-3, EH-14, EH-24 and EH-25 with Python 3.12.3 `hmac`. They match.
  I also probed `ipaddress` and `socket.inet_aton` behaviour locally (S5-3, S5-9).
- **CPG:** not applicable. This is a contract spec, not code. The one code fact (S5-1) is a single
  line of `bundle.py`.
- **Growth-stops check (root `REVIEW.md`):** not applicable. This is the document's first version,
  a new `Extends:` successor, not a revision.
- **Verdict: approve with suggestions.** There is no blocker. The parse grammar is tight, and the
  HMAC construction is sound. One **major** (S5-1) needs a normative sentence before P1-14 encodes
  the rejection paths. Several minors are cheapest to take now, before `"1.1"` freezes the field
  semantics.

| # | Severity | Topic | Confidence |
|---|---|---|---|
| S5-1 | major | Rejection paths echo the value, and `Unit.failReason` is a graph property | code line confirmed; `failReason` content inferred |
| S5-2 | minor (pre-freeze) | Host-independent `pathHmacs` cross W6's own trust line | traced |
| S5-3 | minor (pre-freeze) | IPv4-mapped IPv6: `H` depends on the interpreter's patch level | 3.12.3 verified; newer behaviour from upstream |
| S5-4 | minor | E4 is a confidentiality control for the clear `port`, and the validator does not check it | traced |
| S5-5 | minor (theoretical) | `external`→declared-name matching admits dotted names: a planted confirmation oracle | traced |
| S5-6 | minor (route to `devops`) | Key custody: inside the repo tree, beside the bundles, at the container's uid | traced |
| S5-7 | minor | `bindKeyCheck`: no new oracle, but the published test key is not refused | reasoned |
| S5-8 | nit | §10 overclaims that the oracle never leaves the bundle | traced |
| S5-9 | nit | Missing vectors, among them the only check of §5.3's refusal | traced |
| S5-10 | nit | Clear `scheme`/`port` and `len(pathHmacs)` on `external` are an unstated residual | reasoned |

### Findings

#### S5-1 (major) — the validator's rejection paths echo the value, and the reason reaches the graph

*The `bundle.py` line is confirmed. That `failReason` carries the exception text is inferred,
because no plan fixes its content.*

- **The spec's rule.** §6.3 says `bindParts`, including the clear `host`, and
  `header.bindKeyCheck` never appear "in an answer, a `raw_query` result, the broker audit or a log
  line".
- **What the spec does not fix.** It does not say what a **rejection** may print.
- **What the code does today.** `cpg/ccg/bundle.py:905` raises
  `BundleError(f"line {n}: bindValue {r['bindValue']!r} under keyPath …")`. That puts the
  rejected value in the message, and the value is rejected exactly because it failed the token
  screen or the name rule. `_check_schema` (`bundle.py:168-172`) forwards jsonschema's
  `err.message`, and for a `pattern`/`enum`/`const` failure that message embeds the offending
  instance.
- **Where the message goes.** P1-20 keeps a failing unit's last bundle as
  `stale-kept {built commit, failed at, reason}` (`-p1-linking.md` P1-20 (1)). The graph carries
  `Unit.failReason` as a readable property (`-bundle-spec3.md`, the "Not carried / carried"
  list at line 2062, and `docs/plans/connected-code-graph-graph.md` line 335).

**Why it matters.** Validation exists to catch a writer defect that emits a screen-failing value.
At that moment the catch copies the value out of the bundle, which the stakeholder ruled outside
FR-15, into a log and probably the graph, which is inside FR-15. That graph is raw-query readable
and reaches LLM-backed tools. For `bindParts` the new shapes are BK-27 (a T1 token as a `service`
`host`), BK-30 (T1 on `scheme`) and BK-9 (an `example.com` host). §5.3's refusal "naming
`bindKeyCheck`" is ambiguous in the same way: it could print both checks. A key fingerprint is low
value, but printing it contradicts §6.3 as written.

**Suggested improvement** (one normative sentence in §6.3 or §6.4, plus a P1-14 behaviour):
- *"A rejection or refusal message names the record line, the field and the rule (for example
  `line 7: configEntry.bindParts.host: §6.4 (4) service class`). It never includes the value of
  `bindValue`, any `bindParts` field or either `bindKeyCheck`. `bundle.py` replaces jsonschema's
  message for these fields with that form."*
- Ask P1-20 to store `failReason` as a closed reason code (or that value-free message), never as
  raw exception text.
- **Test that can fail:** feed BK-27 and BK-30 to `validate` and assert that the planted token
  `x7k2q9w4m1p8z3v6` is absent from `str(BundleError)`. Current `bundle.py:905` turns the same
  assertion red for a `bindValue` mutant such as BV-12's value. So the existing defect is the
  positive control, and the fix covers it in the same step.

#### S5-2 (minor, cheapest pre-freeze) — host-independent path hashes reveal capability paths across W6's line

*Traced from §2, §6.2 and §10.*

`pathHmacs[i] = entry_hmac(key, BIND_PATH_TAG, prefix)` does not depend on the host. So equal path
prefixes under **different hosts** give equal hashes. W6 keeps a `loopback`/`service` URL's
`bindValue`, path included, in clear. Its stated reason is that "a path word is not a remote bearer
capability" there. W6 withholds the path for `external`, because there it can be one.

**Example of the leak.** A dev config `http://localhost:8080/deploy-alerts-team` (a local ntfy for
testing) and a prod `https://ntfy.sh/deploy-alerts-team` in the same system give the same
`pathHmacs[0]`, EH-21. The first has `bindValue` in clear. A bundle reader therefore learns the
prod topic, which is a bearer capability. §10 says "only words that already pass in clear can be
revealed". That is literally true, but the word passes in clear only in the context W6 judged
harmless, and the reveal lands in the context W6 judged a capability. The severity is minor: the
leak is bundle-side only, and a bundle reader usually holds the repo (advisory threat model). It is
still the residual F-7 was meant to remove.

**Suggested improvement: scope the path hash to the host.** Set
`pathHmacs[i] = entry_hmac(key, BIND_PATH_TAG, H + "/" + "/".join(segs[0..i]))`. The concatenation
is unambiguous, because `H` contains no `/`.
- **§7 still works unchanged.** The linker compares paths only after the host is decided. For
  `loopback`/`ip`/`service`, `H` is the clear `host`. For `external`, `H` is the matched name `s`.
- **What it costs.** Mount hashes are computed per consumer host rather than once per mount:
  O(consumers × mounts) HMACs at proving-repo scale, which is negligible.
- **Spellings no longer collapse in the hash.** Two loopback spellings reaching one provider
  (`localhost` and `127.0.0.2`) would hash differently. The linker uses the consumer's own `H`, so
  binding is unaffected.
- **What changes in the spec.** The table in §2 and the formula in §5.3 change, EH-14 to EH-23 are
  recomputed with their host, and the BL rows are unchanged in outcome.
- **When.** Only before `"1.1"` ships. Afterwards it needs a new tag and a field change.

#### S5-3 (minor, pre-freeze) — IPv4-mapped IPv6 makes `H` and validity depend on the Python patch level

*Python 3.12.3 behaviour was verified locally. The newer behaviour comes from CPython gh-87799,
backported to maintenance branches (PR 135085, a later fix merged into the 3.9 branch). Which
3.12.x patch first ships it is not established here.*

§3 step 3 sets `H = "[" + str(IPv6Address(inner)) + "]"` after checking `inner` against
`[0-9a-f:]{2,39}`. That class keeps dots out of the **input**, but not out of the **output**:
- Python 3.12.3 prints `::ffff:7f00:1` as `::ffff:7f00:1` (verified);
- after gh-87799, CPython prints an IPv4-mapped address with a dotted tail, `::ffff:127.0.0.1`.

`cpg/pyproject.toml` has `requires-python = ">=3.12"`, so "Python 3.12" in the spec does not pin a
patch level. On an interpreter with the change:
- (a) `hostHmac` for `http://[::ffff:7f00:1]/` differs from the 3.12.3 result, which breaks
  §5.3's determinism claim;
- (b) the emitted `host` holds a `.`, so §6.4 (4)'s `ip` check (§1.7.2's no-dot canonical form)
  **rejects the writer's own output**;
- (c) semantically the address is loopback, yet it is classed `ip`.

`-bundle-spec3.md` §1.7.2 claimed the no-dot rule "keeps Python versions … in agreement". That
claim was already false for this one input, because the older printer accepts it and the newer
one rejects it. The validator is the one place that cannot tolerate this.

**Suggested improvement.**
- §3 step 3, bracketed: `none` if `IPv6Address(inner).ipv4_mapped is not None`, or if the address
  is in `::/96` other than `::1` (IPv4-compatible) or in `64:ff9b::/96`.
- Mirror that in §6.4 (4)'s `ip` class.
- Add BP vectors `http://[::ffff:7f00:1]:80/` and `http://[::7f00:1]/`, both → `none`.

This rejects by address range, not by string shape, so every printer agrees.

#### S5-4 (minor) — E4 protects the clear `port`; "not a confidentiality control" is inaccurate

*Traced from §6.1, §6.4 and the advisory's SA-7.*

§6.1 says E4 (the name rule) "is defense in depth, not a confidentiality control: `bindParts`
emits nothing in clear outside §1.7.2's host and port classes". But the port class is exactly
where SA-7 placed numeric secrets. The name rule is what catches them for `bindValue`: BV-11
(`ADMIN_PIN: 4821`) and BV-26 are rejected **by the validator**.

Under spec5, `x.pin_url: http://localhost:4821` fails E4 only at the writer. The validator does not
check the name rule on an entry that carries only `bindParts`, so §6.4 admits `port = 4821` there.
That is an asymmetry: the validator rejects a PIN as `bindValue` but admits the same PIN as
`bindParts.port`. §6.1 also plans to loosen E4 for `auth`. Loosening it is free on the `auth`
substring, but it would not be free if it ever reached the whole-token words (`pin`, `otp`, `pat`,
`seed`).

**Suggested improvement** (either option):
- (a) **Make the name rule a validator check on any `configEntry` carrying `bindParts`**, as it
  already is for `bindValue`. It costs nothing now, because the validator sees `keyPath`. Later
  loosening of the substring list becomes a major change, which is what a security loosening
  should be.
- (b) **Keep E4 writer-only, but correct the sentence**: *"E4 guards the clear `port` and `host`.
  A loosening may drop substring words only; the whole-token words of `-spec3` §1.7.3 stay."*

I recommend (a).

#### S5-5 (minor, theoretical) — dotted declared names turn §7's `external` match into a planted confirmation oracle

*Traced from §7 and P1-17 (5). It needs an actor outside the advisory's threat model (an
adversarial contributor to one repo of a multi-repo system).*

For `external`, §7 compares `hostHmac` against **every** Compose service name or `container_name`
declared anywhere in the system. A match "proceeds as `service` `s`", and the link's `BOUND_BY`
evidence renders the provider's mount path. Links and evidence are in the graph, so they reach
raw-query readers and LLM tools.

Docker names may contain `.`: container names allow `[a-zA-Z0-9][a-zA-Z0-9_.-]`, and Compose
service names allow `.` (inferred from Docker's documented name classes, not tested here). So a
contributor to repo A can declare `container_name: ntfy.sh` and give that unit many mount paths, a
dictionary of candidate topics. The linker then binds repo B's withheld
`https://ntfy.sh/deploy-alerts-team`, and the answer shows a confirmed link to `/deploy-alerts-team`
from B's config entry. One build confirms the guess. A single-label host can be confirmed the same
way, but a declared single label is in-system topology, which is binding's purpose. A **dotted**
match is never real Compose-network resolution in practice. It exists only to reach `dnsname`
externals, which is where capability hosts live.

**Suggested improvement.**
- In §7, compare `external` hashes only against declared names that match §1.7.2's single
  `label`. A dotted `H` then cannot match.
- Add a BL vector: consumer BP-6, provider `service=ntfy.sh` → no match.
- This loses nothing that §4's own writer rule admits, because a dotted name is never `service`
  at the writer.

#### S5-6 (minor, route to `devops`) — key custody: inside the repo tree, beside the bundles, at the container's uid

*Traced from P1-02(d), `-bundle-spec3.md` §5.17, `cpg/.gitignore` and
`cpg/infra/toolchain/README.md` "Run".*

The brief asks whether the custody rule is enough. For a bundle reader, the hashes are only as
strong as the separation between bundle access and key access, and `bindParts` raises the stakes.
Under a key compromise, every word-only external path (EH-21) and every dictionary-word external
host (EH-3, `ntfy.sh`) is recoverable by dictionary in seconds, because the HMAC adds no work
factor once the key is known. High-entropy segments stay safe, because the prefix hash forces a
full-token guess.

Today's custody:
- P1-02(d) puts the key at `cpg/.ccg-work/keys/<system>.hmac`, **inside the repo working tree**
  and under the same `.ccg-work/` as the bundles (`cpg/.ccg-work/bundles/<system>/`).
- `-bundle-spec3.md` §5.17 still says the key is "kept outside the graph and the repo"
  (P0: `~/.config/ccg/<systemId>.hmac`). Spec5 §2 cites P1-02(d), so it inherits the
  contradiction.
- The commit barrier is a single `.gitignore` line (`.ccg-work/`); I confirmed it is present.
- `0600` separates nothing from a container whose user is uid 1000, "the same uid as the lab's
  host user" (toolchain README). Any `-v` mount of a `.ccg-work` ancestor, for example a `$OUT`
  chosen under the work area, hands the key to the indexer sandbox.
- Positive: the toolchain build pipes its context on stdin (`build.sh:119-121`), so no image build
  can capture the key.

**Suggested improvement** (for `devops`, who decides):
- move the key outside the repo tree, as §5.17 says (`~/.config/ccg/`), so that no repo-scoped read
  grant, mount or archive reaches it; or
- keep the location, but add a P1-02 test that refuses any sandbox `-v` source that is an ancestor
  of `keys/`.

Either way, `-bundle-spec3.md` §5.17 and P1-02(d) should agree.

#### S5-7 (minor) — `bindKeyCheck` adds no oracle, but the published test key is not refused

*Reasoned.*

**What publishing `bindKeyCheck` costs.** `bindKeyCheck = HMAC(k, "ccg.bindParts.keycheck.v1" ‖ 0x00)`.
Publishing it gives an attacker a way to verify a guessed key. Every clear-host entry already gives
the same: `host=localhost` beside `hostHmac` is a known plaintext/MAC pair. Under a random 256-bit
key and a PRF, neither leaks anything. The design is fine, **conditional on the key being
unguessable**:
- P1-02(d) says only "generated once";
- and §9's test key `00…1f` is published together with its check value (EH-24).

A test key leaking into a real run (a fixture default, a fallback in `key_path`) would leave every
hash open to dictionary attack by anyone who reads this spec.

**Suggested improvement.**
- P1-02(d) states the generation: 32 bytes from `secrets.token_bytes`.
- `ccg.atrest` refuses to load a key whose `bind_key_check(key)` equals EH-24, except in the test
  harness. Doing it at key load covers `"1.0"` bundles and `valueHash` too.
- The test: load the §9 key outside the harness and expect a refusal. The positive control is a
  freshly generated key.

#### S5-8 (nit) — §10: "the hashes never leave the bundle, so no raw-query reader holds the oracle"

*Traced.* The hashes do not leave the bundle, but the **result of comparing them** does. Every
link bound by parts tells a graph reader that a consumer's withheld host equals service `s` and
that its path has prefix `M`, which is rendered. That is binding's purpose and acceptable. But §10
should say so, because S5-5 is exactly that channel.

**Suggested rewording:** *"The graph learns, per bound link, the equality its binding asserts
(host = a declared service, path prefix = a provider mount) and nothing else. An unbound consumer
reveals nothing."*

#### S5-9 (nit) — vectors that are missing

- **§5.3's refusal has no vector.** The assembler refuses a bundle whose `bindKeyCheck` differs
  from its own, and that refusal is the guard that keeps two keys' hashes from being compared.
  Without a vector, nothing shows it can turn red (root `REVIEW.md`: every check can fail and
  pass). Add an assembler vector: a bundle written under a second key → `stale-kept`, with the
  same bundle under the system key as the positive control.
- **WHATWG and `inet_aton` IPv4 forms.** Under §3 these are `dnsname`, so `external`. Real
  resolvers read them as IPs (verified: `inet_aton("1.2.3.0x4")` gives `1.2.3.4`, and
  `0x7f.0.0.0x1` gives `127.0.0.1`). The outcome is harmless: the host is withheld, and binding is
  under-bound rather than mis-bound. Pin it with BP rows `http://1.2.3.0x4/` and
  `http://0x7f.0.0.0x1/` → `external`, so that a later "fix" does not start emitting them as clear
  `ip`. The same applies to `foo.localhost` (RFC 6761 loopback), which is `external`.
- **S5-3's and S5-5's rows.**
- **An invariant row.** Every url-form `bindValue` a writer emits comes with `bindParts`. E1–E4
  are implied by `bindValue`'s own conditions, so W6 can never be bypassed by an entry that keeps
  `bindValue` without parts. That holds by inspection (`url` ⊂ `parse_bind_url`'s domain, the same
  role and name rules), but only a test keeps it true.

#### S5-10 (nit) — residual not stated: clear `scheme`/`port` and segment count on `external`

An `external` entry still shows `scheme`, `port` and `len(pathHmacs)` in clear. Together with a
known provider shape (`https`, 4 segments) this can fingerprint a third-party service (a Slack-style
webhook under a neutral key) without revealing the credential. That is acceptable. Add one line to
§10's residuals so it is a decision rather than an omission.

### No finding — checked and sound

These are recorded so they are not re-litigated:
- **Linker injection.** Every `bindParts` input the linker reads is closed-shape:
  - 64-hex hashes, an enum and an integer port;
  - a scheme pattern;
  - a `host` in a validator-fixed grammar;
  - mount paths normalized to `[A-Za-z0-9._~-]` segments, with any other mount matching nothing
    (BL-12).

  No value is interpolated anywhere. One condition holds this up: the linker must consume only
  bundles that passed `bundle.validate`, and §7 should say so in one clause.
- **Prefix semantics.** `d == 0` matches every consumer by design. Prefix-only matching never
  renders a consumer's deeper segments. Hashing prefixes rather than single segments blocks
  position-shifted equality, and under a key compromise it still forces a full guess of a
  high-entropy segment.
- **Domain separation.** Every tag contains `.`, and every record id is 32 hex, so no `bindParts`
  hash can equal a `valueHash`. The `domain ‖ 0x00 ‖ value` split is unique.

### What is solid

- **§3 step 1's character rule removes the WHATWG differential class in one stroke.** It admits
  only `0x21`–`0x7E` and bans `@ ? # ; \ % $`. So there is no userinfo confusion, no
  backslash-as-slash, no tab or newline stripping, no percent-decoding, and no IDNA or homograph
  mapping: `ｌｏｃａｌｈｏｓｔ` returns `none`.
- **No URL library parses the value.** The "last label not all digits" rule closes WHATWG's
  decimal IPv4 shorthand.
- **The bracketed IPv6 class keeps zones and embedded IPv4 out of the input.**
- **`host` is never in clear for `external`, and W6 withholds the clear `bindValue` copy for
  `ip`/`external`.** Together these remove BV-24, BV-28 and the word-only residual, which is the
  goal of F-7.
- **§6.3 extends never-rendered to every `bindParts` field, the clear `host` included,** and the
  answer schema's closed `launch` hop cannot represent the fields anyway.
- **The validator holds no key.** The key-epoch check stops cross-key comparison from failing
  silently.
- **The vectors are reproducible.** I independently recomputed five of them.

### Open questions

1. **For `architect` and `devops`: where does the config writer run?** It needs the key to emit
   `bindParts`. If it runs inside the extraction sandbox beside third-party indexers, the key
   enters a container that also executes target-repo tooling. Running the config writer on the
   host, or handing the sandbox only a derived per-build subkey, would avoid that. This decides
   how much S5-6 matters.
2. **For P1-20: what is `failReason`?** If it is a code, S5-1 shrinks to the log path.
3. **For the stakeholder, via `architect`: is the adversarial-contributor actor of S5-5 in scope**
   for multi-repo systems? The advisory's threat model is the accidental committer. If it is out
   of scope, S5-5 is a cheap hardening rather than a required one.

### Proposed REVIEW.md lines

Target: root `REVIEW.md`, heading "Always check". Source: S5-1. A never-rendered rule was stated
for success paths while `bundle.py:905` already echoes a gated value on its failure path.

- Addition: *"A rule that a field is never rendered, logged or stored outside its file is checked
  on the failure paths too: every validator, refusal and exception message that can carry the
  field's value, and wherever that message is stored (a graph property, a report, an audit)."*

No prune: no test, CI job or hook enforces this today.
