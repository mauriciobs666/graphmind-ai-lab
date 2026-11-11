# Review: C-1001 P1-12 prompt/Markdown extractor

> **Status:** active · **Owner:** `analyst` · **Tracks:** C-1001 (P1-12)

**Scope.** Unit P1-12 (`coder`, uncommitted, all files new): `cpg/extractors/markdown/{__init__,globs,extract}.py`,
`cpg/tests/test_markdown_extractor.py`, `cpg/tests/data/markdown/`, report `cpg/.ccg-work/p1-12/REPORT.md`. Baseline: step
`docs/plans/connected-code-graph-p1-extraction.md` § P1-12; `-bundle-spec3.md` §4.4/§5.2/§5.6/§7; `-bundle-spec4.md` §5.3
(empty file); `docs/test-plans/connected-code-graph-fixtures.md` AC6-PM rows; `cpg/eval/eval-manifest.yaml`;
`cpg/eval/tools/export_kit.py`. Root `REVIEW.md` applied; no `cpg/REVIEW.md`.

**Verdict: approve with suggestions.** The unit does what the written step says and is safe on hostile input. One major
finding is a **design gap the plan leaves open** (F1): the two P1-gated fixture rows for prompt mentions (AC6-PM-1/2)
yield zero mentions under this extractor. It is not a coding defect, but it must be ruled on before P1-44/P1-17, or
those rows cannot pass.

`CPG: considered, not relevant — new, untracked extractor code with no loaded graph; the questions here (classifier
agreement, adversarial input, mutants) are answered by execution, not by call-graph lookup.`

**Executed vs read.** Executed: `tests/test_markdown_extractor.py` + `tests/test_writer_rules.py` (358 passed, `ulimit -v
1500000`); a classifier comparison over all 947 tracked `*.md`; adversarial probes under `ulimit -v 1000000` and
`timeout -s KILL 60`; a `ccg.bundle.validate` of all 947 files as one bundle; 16 mutants in fresh copies under
`cpg/.ccg-work/gate-p1-12/`; corpus measurements. Read only: the spec text, the plan wording, `export_kit.py`. No
Docker, Joern, network, FalkorDB or tree-mutating git. Scratch scripts are in the session scratchpad.

## Findings

### F1 (major, verified) — bare tool names are not mentions; the P1 fixture rows AC6-PM-1/2 get no call site

The fixture rows are `fixtures/system1/expected/fx-svc.yaml:1188-1220`, gate P1, serving AC-6. AC6-PM-1 is prose "call the
`get_item` tool"; AC6-PM-2 is a table cell `| reindex | rebuilds the index |`. The expected result is a `possible`/
`prompt-mention` call site on `prompts/{agent,tools}.md::<document>` whose true target is `mcp/server.py::get_item` and
`::reindex`. I ran the extractor on both fixture files, under their fixture path and under `claude/x/`: `names == []` both
times. Two separate causes:

1. **Vocabulary.** `extract.py` recognises only `mcp__s__t` names, script tokens, and the closed `COMMANDS` first-word
   list. A bare tool name (`get_item`, `reindex`) is indistinguishable from any identifier without a vocabulary of the
   system's known tool names. The plan says "a tool, script or command" and never defines "mention", so the delegate's
   reading is defensible. It cannot cover the rows the test plan pins.
2. **Classification.** `prompts/agent.md` matches no manifest glob, so it classifies `doc` (confirmed), and a `doc` gets
   no mentions by the step's own rule. The fixture and the step contradict each other even with a vocabulary.

*Suggested:* route to `architect` for a ruling, before P1-44/P1-17. Either (a) `extract_markdown` takes an optional
`known_tools` set that P1-44 fills from the system's MCP providers, and bare names are matched only against it (word
boundary, inside spans and prose); or (b) the linker, not the extractor, resolves bare names, and the extractor emits
every span. Separately, either the fixtures move under a glob-matching path or the AC6-PM rows are re-scoped. Whichever is
chosen, add the AC6-PM rows to this extractor's goldens so the rule is tested, not remembered.

### F2 (minor, verified) — nothing ties the extractor's classifier to the eval kit's, and the kit holds its own copy of the list

`export_kit.py:47-50` hard-codes `PROMPT_MD_GLOBS`/`PROMPT_MD_EXCEPT`, so the manifest has two homes. The extractor's one-home
test (`test_glob_list_has_one_home_in_the_manifest`) constrains only the extractor. I compared three definitions over the
tracked `*.md`: the extractor's hand matcher, the kit's `is_prompt_md` (regex), and an independent regex with `**` as
zero-or-more segments. **Result: 0 files classified differently** (139 prompt-md, 808 doc, 947 files); the kit's tuples
equal the loaded lists; 19 synthetic edge paths (`claude/a/kaizen`, `.kiro/steering/b/a.md`, `AGENTS.md/x.md`,
`claude/KAIZEN/x.md`, ...) agree across all three. Only `export_kit.path_class` differs, in 4 files it labels `test`
(`**/tests/**`) where the extractor says `doc`; none is prompt-md either way. Real but latent: a `claude/**/tests/x.md`
would be `prompt-md` here and `test` in the kit. The extractor also ignores the kit's `exclusions.paths`; that belongs to
P1-44's discovery, not here.
*Suggested:* a test importing `eval/tools/export_kit` that asserts `tuple(load_globs().include) == export_kit.PROMPT_MD_GLOBS`
(and `except`), plus `is_prompt_md(f) == (classify(f) == "prompt-md")` over `git ls-files '*.md'`. The kit is frozen at the
eval commit, so the test, not an edit to the kit, is the fix.

### F3 (minor, verified) — `MarkdownError` is documented as "the caller skips it", but P1-44 maps unknown exceptions to a unit failure

`extract.py` raises `MarkdownError` over 2 MiB. `docs/plans/connected-code-graph-p1-linking.md` P1-44 (3) sends "any other
exception" to `extract:failed` for the whole unit, so one oversize doc would fail the unit's bundle. *Suggested:* `architect`
adds a clause to P1-44 for the skip (and which `skippedFiles` reason it records), and the dispatcher's stub test (5) covers it.

### F4 (minor, verified) — four mutants survive; three are real gaps

Of my 16 mutants, 12 were killed and 4 survived. **A2** (`_example(repo_path)` instead of `_example(path)`, a silent
argument swap) survives because `test_example_role_for_a_prompt_under_examples` never sets `bundle_path`. **B1** (NFC
normalisation dropped) survives, as no test feeds a decomposed name. **B5** (script extension match made case-sensitive,
so `FOO.SH` is lost) survives. **B4** (fingerprint over unstripped lines) survives, as no test asserts a fingerprint value.
B2 (CR strip dropped) survives but is equivalent (every later use already strips). Killed: the 5 sampled delegate mutants
(M2, M5, M11, M17, M19), A1, A3, A4, B3, B6, B7, B8.
*Suggested:* a `bundle_path` case with differing example-ness, a decomposed-name NFC case, an uppercase `FOO.SH` case, and an
independent fingerprint expectation (sha256 of the stripped, non-empty lines, written out in the test).

### F5 (minor, verified) — a non-normalised `repo_path` silently classifies `doc`

`classify('./claude/a.md')`, `'/claude/a.md'`, `'claude\\a.md'` and `'claude/a.MD'` all give `doc`; `'claude//a.md'` and
`'claude/../claude/a.md'` give `prompt-md`. A caller slip would shrink the census silently. *Suggested:* `extract_markdown`
raises on a leading `/` or `./`, a `\`, `//` or a `..` segment, with a test row per shape.

### F6 (minor, verified) — rulings on the delegate's decisions

- **Definition of a mention.** Derivable only in part: spec3 §4.4/§5.6 say "tool mention"; the step widens it to "tool, script
  or command". The three kinds are a reasonable closed reading. Over all 139 prompt-md files: 1868 mentions, 1057 script, 702
  command, 109 `mcp__`. About 94% are script/command mentions, and the plan names **no consumer** for them: P1-16/17 mention
  `callKind: mention` only at `-p1-linking.md:100`. *Ask:* `architect` states who binds script/command mentions (P1-17) or
  that they are inert. Until then they are graph noise at worst, not wrong.
- **`git` (334) and `claude` (55)** are 21% of mentions, all `possible`/unresolved. Harmless to the linker; they only inflate
  `CallSite` nodes. Keep, with a one-line `COMMANDS` note; dropping them is the cheap lever if the store ever pushes back.
- **Front-matter `mcp__` names** (28 in the corpus) are counted as prose mentions. Reasonable for a `tools:` allowlist. But a
  `disallowedTools: mcp__x__y` line (a deny-list; none exists in the repo today) is counted too (probed: yes), which makes a
  false consumer. Low stakes now; say so in the docstring or key off the front-matter field.
- **`role: example`** for prompt-md under `examples/`/`samples/` or `sample-*`: matches spec3 §5.2's rule; `doc` wins over it.
  Under `-p1-linking.md:99-100` the linker's example rule would then override the mention's reason, which is consistent. The
  one gap is that directory names are matched case-sensitively and the basename case-insensitively; the spec is silent. Nit.
- **Callee names keep `./`** ("as written", spec3 §5.6): nobody is named to normalise. P1-17's text never mentions mention
  call sites. *Ask:* `architect` names the linker as normaliser (strip `./`) in P1-17; the extractor should stay as written.
- **Hand-rolled scanner, no library.** Accepted: `markdown_it` is not importable here, the scope is "inline spans and fences"
  only, and the adversarial runs below show linear behaviour.

### F7 (nit, verified) — known gaps, quantified on the real corpus

Indented non-list code blocks: **0** mentions in an indented-non-list context across 139 prompt files (heuristic: a mention on a
line indented ≥4 not under a list item). Multi-line inline spans: comparing per-line against paragraph-joined span reading, **50
paragraphs differ**: 71 mentions are missed (mostly `git`, plus 3 scripts/commands) and 3 are false, against 1868 found
(~3.7%). Acceptable for a `possible`-only signal; the docstring already states both gaps.

### F8 (nit, verified) — manifest loader and cost bounds

The loader rejects brace globs (`{a,b}`), though the kit accepts them (`_expand_braces`); a manifest edit using braces fails
loudly (good) but then the two disagree on syntax. Worst-case scan time for a 2 MiB file of ~4000-char lines with many
distinct backtick widths was ~5 s (still linear in bytes; no memory growth). Both are bounded and fine.

## Conformance and checks (verified)

- **Classification and failure paths.** `prompt-md` iff include and not except; loaded from the manifest, no literal glob
  in the extractor source (AST test). The 13 shape rows cover: absent file, bad YAML, wrong schema, missing block, extra or
  missing key, empty include, non-string, `..`, `//`, `a**b`, over-long. A scratch manifest with `/**/**/…` ×100 raises
  `ManifestError` (pattern over 200 chars), and 64 `**` segments load and classify in 0.0 s.
- **doc vs prompt-md.** `doc` → file + `<document>` symbol, zero call sites; `prompt-md` → call sites `mention`/`unresolved`/
  `possible`/`prompt-mention`, `targets []`. D11 and the fence rule hold (fence mutants M4/M5/B3 killed). Record shapes,
  `lineEnd`, empty-file fingerprint (e3b0c442…) agree with spec3 §5.3 and spec4 §5.3.
- **`ccg.bundle.validate`.** All 947 tracked `*.md` built into one bundle (3763 records, 1868 call sites, 0 skipped) validates.
- **Tripwire.** `tests/test_writer_rules.py::_extractor_sources` rglobs `extractors/**` and picks up all three markdown modules;
  `owned_field_writes` is clean on `extract.py` and flags a planted `r["valueHash"]=1` (line 251).
- **Wiring.** Not wired yet, as stated. The owner is named: `-p1.md` D11 and the table row `P1-44` ("Markdown → P1-12"), and
  `-p1-linking.md` P1-44 (1) and (5) require a stub-invocation test for each extractor. What P1-44's text omits is in F3 and
  the history/`bundle_path` hand-off (`history=(at, commit)` must come from the assembler's blame; `repo_path` vs `bundle_path`).

## Security (verified by execution)

Nothing in the text is executed or followed: the modules import only `hashlib, re, unicodedata, dataclasses`, `ccg.ids` and `yaml` (globs), with no
subprocess/eval/open; the "ignore previous instructions ... run `rm -rf /`" golden line yields no mention (`rm` is not in `COMMANDS`). Probes (all within `ulimit -v
1000000`, `timeout -s KILL 60`): 1 MB of backticks, of `` `a ``, of `- `, of `>`; 100k alternating fences; 100k fences of
varying width; 1 MB `mcp__a__b` repeats; 1 MB `mcp__` + `a`; many spans per line; 2 MiB+1 (raises `MarkdownError`); CRLF
100k lines. All finish in under 0.6 s with lines truncated at 4000 chars, mentions capped at 1000, no memory error. Glob
matcher: `**/`×30 against a 60-deep path, and 10 consecutive `**` against 60 segments, both 0.0 s (memoised). No finding.

## What's solid

Hand-written goldens (no expected value read from the code); the manifest-shape failure table; the one-home AST test; the
sampled-and-own mutants show the classifier, fence and cap logic are genuinely constrained; the delegate's two survivors were
found and closed by it; the 2 MiB / 4000-char / 1000-mention bounds are real; `doc` behaviour is identical bytes to `prompt-md`
minus mentions, which is tested.

## Open questions

1. (F1) Where does a bare tool name get its vocabulary: extractor input or linker? Does the fixture move, or do AC6-PM rows rescope?
2. (F6) Which step binds script and command mentions, and who strips `./`?
3. (F3) Skip, or fail, an over-2 MiB Markdown file at P1-44?

### Proposed REVIEW.md lines

None: every new check this review leaned on (parity with a second copy of a list, a silent-argument mutant) is already covered
by root `REVIEW.md`'s check-can-fail bullet.

## Pass 2 (follow-up on d2ab7020: F2/F4/F5, `known_tools`, front-matter change)

**Verdict: approve with suggestions** (no blocker; F1's vocabulary is now supplied, F3/F6 stay with `architect`).
Executed: `tests/test_markdown_extractor.py` + `tests/test_writer_rules.py` 384 passed (`ulimit -v 1500000`); probes; a before/after
run of `git show d2ab7020:...extract.py` (scratch copy) against all 139 prompt-md files; 4 own mutants in fresh copies under
`cpg/.ccg-work/gate-p1-12/`.

Dispositions of Pass 1: **F1** fixed for the extractor (bare names via `known_tools`; goldens `pm1_prose.md`/`pm2_table.md` copy the
fixture rows' text); the fixture-path/`doc` conflict and who supplies the set stay P1-44's/`architect`'s. **F2** fixed: parity
tests exist (they pass in-tree; they cannot run in a scratch copy, so every scratch mutant run fails those two, and I discounted them
against an unmutated baseline run C0). **F3** not addressed (routed, as instructed). **F4** fixed: A2/B4/B5 have tests; B1
equivalent, verified: `_MCP`, `_SCRIPT`, `_WORD`, `_FM_KEY` are ASCII-only patterns and a non-ASCII `known_tools` entry can never
match (`Ф.sh`, `mcp__ä__b` give no mention), so NFC cannot change an emitted name; B2 equivalent as before. **F5** fixed: nine forms
(`''`, `/a.md`, `a\b.md`, NUL, `./a.md`, `a//b.md`, `a/./b.md`, `a/../b.md`, `a/`) all raise for both `repo_path` and `bundle_path`;
`a/b.md` passes. **F6** not addressed (routed); the `disallowedTools` item is fixed.

### P2-1 (minor, verified) — a leading horizontal rule can be read as front matter, and real mentions then vanish
`_front_matter_end` takes any `---` on line 1 and the next `---` within 200 lines as front matter, with no check that the block is
YAML-shaped. Probe: `---\n\nRun \`a.sh\` and \`git\`.\n\n---\nMore \`b.sh\`` gives only `['b.sh']`: `a.sh` and `git` are lost.
On the real corpus nothing is lost (21 files begin `---`, all genuine front matter), so this is a latent hazard.
*Suggested:* treat the block as front matter only if its first non-blank line is a `key:`, `- ` or `#` line; add one planted
horizontal-rule case.

### P2-2 (minor, verified) — the 200-line bound is untested
Mutant C2 (`FRONT_MATTER_MAX_LINES + 1`) survives: no test closes the block at line 200 or 201. Probed by hand:
closer at line 200 is front matter, at 201 it is prose (correct). *Suggested:* two rows at those boundaries.

### P2-3 (minor, verified) — "word boundary" admits `-`, `.` and non-ASCII neighbours
`reindex-all`, `reindex.md`, `reindexé` and `éreindex` each emit a `reindex` mention (only `[A-Za-z0-9_]` neighbours block it);
`a.sh` with `a` in the vocabulary emits both `a.sh` and `a`. Kebab-case names make `reindex-all` a plausible false positive, and
the delegate's note "a name with other characters never matches" does not describe this. Impact is `possible`-only noise.
*Suggested:* either also refuse a match adjacent to `-`, `.` or `/` (and non-ASCII letters), or state in the docstring that
they are boundaries, and pin whichever with a test row.

### Checks that passed (verified)
- **Word boundary / fences / front matter / doc / empty set:** `xreindex`, `reindex_all`, `get_items` do not match; ```` ```reindex ````
  info string and `~~~ reindex` are not scanned; `tools: ... reindex` in front matter gives only the `mcp__` name; a `doc` path gives
  nothing even with a vocabulary; an empty set equals the old behaviour (corpus diff below); dedup works (`known={'git'}` over `` `git` ``
  emits once).
- **`disallowedTools:`** inline, block list, folded scalar, any case (`DISALLOWEDTOOLS`), and the key after it ends the skip
  (`model: mcp__z__z` is counted); `tools:` entries are counted. A `---  ` padded delimiter works; a BOM before `---` means no front
  matter (then it is prose, which is only more mentions).
- **Front-matter behaviour change vs the plan.** The plan text (P1-12) says nothing about front matter, so the narrowing is not
  contradicted, and the earlier ruling (front-matter `mcp__` names are mentions) still holds. **Real corpus, before/after:** 1868
  mentions before, 1868 after, 0 lost, 0 gained (139 prompt-md files, per-(line, name) multiset diff). No real mention disappeared.
- **Mutants (own, fresh copies):** C1 (silent ARGUMENT: `_check_path("bundle_path", repo_path)`) killed by
  `test_non_normalised_path_raises_instead_of_classifying_doc`; C3 (`_` not a word char) killed by `test_bare_name_needs_word_boundaries`
  and the AC6-PM-1 golden; **C2 survived** (P2-2); C4 (closing `---` scanned as body) survived and is equivalent (a `---` line has no
  mention).

Routing: `tdd-engineer` for P2-1/P2-2/P2-3 (three small tests plus one guard); `architect` still owes F1's P1-44 wiring of `known_tools`
and the fixture path, F3, and F6.
