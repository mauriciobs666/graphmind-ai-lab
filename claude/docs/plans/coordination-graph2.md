# Work-item scheme IDs in the coordination-graph backlog parser — plan extension

> **Status:** active · **Owner:** `architect` · **Tracks:** — · **Extends:** `claude/docs/plans/coordination-graph.md`

Extends `claude/docs/plans/coordination-graph.md` (the base plan stays authoritative). States only
the change: the backlog parser must accept work-item scheme IDs (`GAL-F0042`,
`GAL-I03-F0012-S01-T01`, `GAL-I02-K0001`; grammar `skills/work-items/docs/SPEC.md` §1) as well as
legacy `K-004` / `C-310`. Parser opacity rule: `docs/plans/cross-initiative-coordination-ids.md` FR-4(b).

## Changes against the base plan

- **§3.6 item heading** is replaced by
  `^### (?P<id>(?:[A-Z]+-\d{3,}|[A-Z]{2,6}(?:-[IFSTDK]\d{2,})+)) — (?P<title>.+)$`.
  The pattern checks shape only; canonical widths and reserved keys are `wi resolve`'s call. `FR`/`AC`
  path segments are not item segments and never match.
- **Opaque token (§3.6, §3.7):** apart from this heading, every ID token (`Depends on` refs, history
  ids, ledger citations) is split on commas, trimmed and compared as-is — never regex-validated.
- **§3.6 `Depends on` row:** refs may be legacy or scheme IDs, bare or `<namespace>:<ID>`.
- **§3.7 delivery matching:** the id is matched as an opaque token; add the example
  `## … — GAL-F0042 delivered`, and a scheme ID among the slash-joined siblings.
- **§3.6 one-time reformat:** the "series past the highest id" and `K-001` rules apply to
  legacy-ID backlogs only; a backlog already using scheme IDs has new IDs allocated by `wi new`.
- **§2.1 sources table:** the Agent plans row reads "legacy `### K-0NN — …` or scheme-ID headings";
  the `C-NNN` (cpg) cell is unchanged.
- **§5 backlog-parser tests:** add a table test of the heading pattern (positives: `K-004`,
  `C-310`, `K-0001`, `GAL-F0042`, `GAL-I03-F0012-S01-T01`, `GAL-I02-K0001`; negatives: `GAL-`,
  `gal-F0042`, `GAL-F4`, `K-04`, `GAL-X0001`, `GAL-F0042-FR2`); and a test that a scheme-shaped
  `Depends on` token is split and compared without shape validation.

## S1

S1 (`coordgraph/parse/backlog.py`, `history.py`) uses the pattern above. Read in full, S1's row
conflicts with nothing: it names no ID regex, and its done-condition (offline parser tests, history
coverage probe) is met by the added tests.
