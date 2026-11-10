# python-web-quirks KB: raw-capture list, coverage and gap ruling (S7)

> **Status:** active · **Owner:** `cobb` · **Tracks:** — · **Last updated:** 2026-11-10

Output of S7 of `docs/plans/python-web-quirks-kb.md` (D11, "Raw captures and duplicates"); cited by
the coordination ledger. Derived **2026-11-10** by read-only queries on `ws:agent-team` (482 `Document`
nodes at derivation; nothing written or deleted). S13b re-derives it before H1 and again at S14's start;
any capture newer than that re-derivation is excluded by id at S14.

**Method and limit.** (1) `CONTAINS 'python-web-quirks'` over every document's title and text. (2) Keyword
sweeps over the text for each of the skill's 42 gotchas (API names, flags, error shapes), read against
the skill's section text. This is a lexical sweep, not a semantic one: a restatement phrased with none of
the keywords could be missed, which is why S13b re-derives. Producer is the `INGESTED_BY` agent.

## 1. Captures that mention the skill (4)

All four also restate a gotcha, so none is "path-only".

| Id | Producer | Title (abridged) | Ruling |
|---|---|---|---|
| `38cd49cddb004207a5c7abd10181a4eb` | architect | BackgroundTasks runs one request's tasks sequentially | restates §889; states a gap, now closed (§3) |
| `d0b47b2ccc874f1ebdda87b977d9dd1e` | architect | `http.client.InvalidURL` is an `HTTPException`, not a `ValueError` | restates §901 |
| `9a5e835d82e14bb1b3705b6ee22cb33e` | teco | same-length mutant in the same mtime second keeps the stale `.pyc` | restates §803 |
| `14745ca484644f8db040438b6f3932bb` | tdd-engineer | FastMCP `Annotated` Field bound violations are `isError` results | restates §922 |

## 2. Captures that restate a gotcha without naming the skill (13)

| Id | Producer | Title (abridged) | Restates (section heading, first words) |
|---|---|---|---|
| `1bfe74233c1a438dafafb2ed57f12cc8` | analyst | `--deselect` with absolute node IDs matches nothing | `pytest --deselect` |
| `a792f59aa1cc43a389266c01471ae49a` | teco | `-k 'not _live'`; `--deselect` absolute ids match nothing | `pytest --deselect` (the `-k` half is falkor-chat-specific) |
| `3cab6295c3994cdab3b2eb4cd06befed` | analyst | same-size mutants run a stale `.pyc`; fresh `PYTHONPYCACHEPREFIX` | `A same-second edit…` and `PYTHONPYCACHEPREFIX=<fresh…` |
| `3ab71d8271e04dd093ecc270a800ff13` | analyst | `python -I` ignores `PYTHONPYCACHEPREFIX` | `PYTHONPYCACHEPREFIX=<fresh…` (the skill states the `-I` caveat) |
| `2d0168efb3db4552bff3952f28c543d3` | coder | `pytest -q -q` prints no "N failed" line | `pytest -q -q` |
| `c590ed3adacc41f8a10366bb8d568e14` | coder | fd 0 is `/dev/null` under capture | `Under pytest's default capture, fd 0…` (also touches `-q -q`) |
| `95ab714455f146fbab7ffb137c158762` | analyst | `Popen.wait()` after `killpg` leaves a TERM-ignoring descendant | `Popen.wait() after os.killpg…` |
| `23d09f2bd0f34d3eb925fb2b65bf2326` | analyst | FastMCP tool args are pydantic-validated | `FastMCP validates tool arguments…` |
| `7fde1fc1f1924749b89dbad8ccf147bc` | analyst | list-element / Enum constraint under `items` / `$ref` | same FastMCP section (the schema-position half) |
| `60aba8dbe7bb48b28bda2a4c8cdb419c` | analyst | "Correction": `PYTHONPATH` does not override an editable install | `A private copy for mutation testing…`; **contradicts** the skill's reading, see note |
| `6ce4a2321bcb44989d382e2c762edd0f` | `u1` (unattributed) | `PYTHONPATH` plus a non-package cwd | same section; supplies the cwd condition |
| `629325c220234d9b94cea8109f032bde` | coder | mutant copy needs `results/`; `PYTHONPATH=<copy>` | same section (the skill's source) |
| `35b0029bb9404f3e9e27fa2b95a01340` | architect | LM Studio answers an unknown route with 200 + error JSON | `An OpenAI-compatible local server can answer a missing /v1…` |

**Note on `60aba8db…` / `6ce4a232…`.** `60aba8db…` (created first) says `PYTHONPATH` alone does not
override the editable install; `6ce4a232…` shows it does unless the cwd is the real package directory,
which is the likelier explanation of the first result. The skill's §990 text says `PYTHONPATH=<copy>`
is needed and does not state the cwd condition. That condition is not in the skill, so it is a small
unpromoted detail. It is recorded in the kaizen item (§4), and neither capture is a reason to hold H1.

## 3. Coverage of the 42 sections, and the gap ruling

**Sections with at least one restating capture (11 of 42):** §803 stale `.pyc`; §889 BackgroundTasks
sequential; §901 `InvalidURL`; §910 `--deselect`; §922 FastMCP; §940 `-q -q`; §949 `PYTHONPYCACHEPREFIX`;
§966 fd 0; §977 `killpg`; §990 mutation copy; §396 `/v1` 200 + error envelope.
**No restating capture (31 of 42):** every other section. That is expected: the captures are the
leftover raw stream, and the skill's other gotchas were written by `cobb` directly or promoted from
captures that are already gone. The ruling for these 31 is none needed; no capture is owed a deletion
or a promotion, and nothing in them is lost when the skill is migrated.

**Gap ruling for `38cd49cddb004207a5c7abd10181a4eb`.** Its text closes with a gap statement: the skill
"covers the 40/100-token limiter bound but not this per-request serialization". The gap is **already
closed**. `docs/reviews/team-learnings-distillation3-sweep.md` row 11 records it as promoted and verified
(`inspect.getsource`); `skills/python-web-quirks/SKILL.md` §889 carries it (verified 2026-11-06), and the
migrated copy `claude/knowledge/python-web-quirks.md` carries the same section (the line
"The bounded-pool entry above is about *total* parallelism; this is the per-request half."). Ruling:
**no new promotion text is needed beyond that section**; the kaizen item (§4) records the ruling and the
pointers so the stakeholder can delete the capture at H1 with the gap shown to be closed.

**Captures not covered by the skill (keep; not on the H1 list).** Four python gotchas appear in the
workspace and in none of the 42 sections (grep of the skill for `basicConfig`, `worktree`, `reloader` returns 0):
`e7f3f1195dd8490d939c896311ff28ba` and `a767dde2eb154f09923e8e80eaabef03` (FastMCP's constructor calls
`logging.basicConfig`; two captures, analyst and devops), `ab271eae5eb045b9a59d8ef1244e60e4` (a `git
worktree` does not isolate an editable install; cobb), `a2cf142e9ed0466aac2abcb9a629f44b` (stopping
`uvicorn --reload` by listening PID kills only the worker; devops). They are candidates for a new
section through the FR-10 path (raw capture, then `cobb`), not duplicates.

**Reviewed and ruled unrelated** (title or keyword matched but the gotcha differs from every section):
`781c055d…` (`attest` blocks on `input()`; the skill's `input()` section is the pytest `OSError`),
`8be79de8…` and `c6bad36b…` (race-testing technique, not §759), `baec4750…` (parametrized-collection
shrinkage, not §512), `572eea93…` and `fd243705…` (extract-and-execute), `12eb7b6c…` (flaky suite),
`681f7f73…` (falkor-chat logging config) and the LM Studio probe documents.

## 4. H1 delete list (ids; the stakeholder deletes by `curl`, no agent deletes)

The 17 captures of §1 and §2, by full id:

```
38cd49cddb004207a5c7abd10181a4eb  d0b47b2ccc874f1ebdda87b977d9dd1e  9a5e835d82e14bb1b3705b6ee22cb33e
14745ca484644f8db040438b6f3932bb  1bfe74233c1a438dafafb2ed57f12cc8  a792f59aa1cc43a389266c01471ae49a
3cab6295c3994cdab3b2eb4cd06befed  3ab71d8271e04dd093ecc270a800ff13  2d0168efb3db4552bff3952f28c543d3
c590ed3adacc41f8a10366bb8d568e14  95ab714455f146fbab7ffb137c158762  23d09f2bd0f34d3eb925fb2b65bf2326
7fde1fc1f1924749b89dbad8ccf147bc  60aba8dbe7bb48b28bda2a4c8cdb419c  6ce4a2321bcb44989d382e2c762edd0f
629325c220234d9b94cea8109f032bde  35b0029bb9404f3e9e27fa2b95a01340
```

Each is named in a distillation sweep and `claude/cobb/scripts/kb-claim-manifest.json` (git grep, 2026-11-10),
so its disposition is recorded. Requirements decision log entry "The four raw captures mentioning the
skill?" names only the four of §1; the §2 thirteen are the plan's "restates a gotcha" half of D11 and
are listed for the stakeholder's call. If the stakeholder wants only the four, delete §1.
Re-derive at S13b before H1: a capture ingested since is not on this list.
