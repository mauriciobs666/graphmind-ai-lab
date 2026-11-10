# work-items — History

> **Status:** active · **Owner:** `coder` · **Tracks:** GAL-F0001

Dated entries only, newest last.

## 2026-11-10 — Work-item numbering delivered (GAL-F0001)

Shipped the `work-items` skill and `wi` CLI (plan steps 1-12 plus 6b/7b/8b, then the step-10/11
consumer edits and the close-check fixes): grammar and FR-25 legality, journal + tracked
registry with flush and reseed, allocator, resolve/move/promote, legacy minting, links and
attributes, header sync, rebuild from registry history (`--floor`, `--accept-reissue`, `rebuilt`
marker), `check`, shared keys, `SKILL.md`/glossary with drift test, agent-prompt and root
`AGENTS.md` updates, GAL bootstrap (key claimed, registry initialized with `I01`/`I02` and
`GAL-F0001`, legacy namespaces/paths declared, registry committed). The skill is linked at
`~/.claude/skills/work-items` (done). Suite at the end of this run: 505 passed
(`model-bench/.venv/bin/python -m pytest -q tests` and `cpg/.venv/bin/python`, from `skills/work-items/`).

- **Requirements / plan / reviews:** `docs/requirements/work-item-numbering.md` (`bc326daf`), `docs/plans/work-item-numbering.md` (`947e81fd`; revisions `b6b8f730`, `556a26c1`, `d2caf4c7`), `docs/reviews/work-item-numbering.md` Passes 4-11.
- **Implementation commits, oldest first:** `2353ab33` (step 1), `5ca9ccc8` (2), `59a1911c` (3), `c28471f4` (4), `2e8635fd` (5), `28ce49c5` (Pass 7 fixes), `5e8076e9` (6), `99345898` (6b), `42e46051` (7), `0dd7fd0b` (8), `c321abea` (Pass 8 fixes), `20db8176` (7b), `e5fa1fe1` (9), `d70103b2` (10: tico/teco/architect/agent-maintenance prompts, flush rule in `claude/AGENTS.md`), `9da81acb` (11: root `AGENTS.md` header fields), `2cc4c4ee` (12: GAL bootstrap), `f343891a` (Pass 9 fix), `36b7c392` (Pass 10 close-check fixes S-1, C-1, C-2). Coordination-ledger commits carry the `coordination:` subject prefix and are omitted.
- **Live spec:** `skills/work-items/docs/SPEC.md` written whole (sections 1-11), pointed to from root `AGENTS.md` and `skills/README.md`.
- **Open follow-ups:** P9-2 (the rebuild guard runs after the reseed reads stale ref tips); P9-4 (bare `K-` names can never resolve); N5 (a public `Store.read_events()` for `check.py`); step 13 handoffs (`tico` amends team-engagement AC-1; the cross-initiative plan author decides whether `needs` cites scheme IDs and the parser treats them as opaque); the real journal event `GAL-I02-K0001`, from another session, awaits its owner's `wi flush` and commit.
