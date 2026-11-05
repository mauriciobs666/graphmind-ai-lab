# Work-item numbering and terminology — Feature Requirements

> **Status:** Ready for design · **Owner:** `tico` · **Tracks:** — · **Last updated:** 2026-11-04

## Intent
One common vocabulary and one numbering scheme for every unit of work, so that any item can be cited, found and related to another without knowing which component or project it came from. It is the standard that the coordination work (overlap checks, parallel runs, the coordination graph) and green-field projects in other repos rely on.

## Problem & current state
- Identifiers are per component and inconsistent: `C-` (CPG backlog, root `docs/BACKLOG.md`), `K-` (falkor-chat backlog, also used for kaizen), few or no numbered items in `mcp-monitor`, `opencode`, `model-bench`; each agent's `kaizen/plan.md` has its own `K-001…` sequence, so a bare `K-004` is ambiguous across the repo.
- Requirements documents are identified by slug only; `Tracks:` points at a backlog ID or `—`.
- FR/AC numbers are local to one document.
- No shared vocabulary for the levels of work.
- A number is assigned by reading a backlog and taking the next free one, which breaks under concurrent agents or branches.
- Cross-initiative coordination (`docs/requirements/cross-initiative-coordination.md`) records the units a unit needs, including in another ledger, so each unit must be nameable unambiguously from another document; green-field projects in other repos (`claude/docs/requirements/team-engagement-in-other-repos.md`) feed the same shared pool and must not collide with this repo's items.

**Context for design (not decided here):** the rules (glossary, ID shape, allowed parents, citing, legacy names) may suit a skill under `skills/`, which would also reach green-field repos automatically (`claude/docs/requirements/team-engagement-in-other-repos.md`, FR-6, FR-7). Allocation, lookup and checks (FR-5 to FR-9) need real tooling a skill can only point to, and the list of project keys (FR-18) must be shared across projects.

## User stories
- As the stakeholder, I want every epic, feature, story, task, requirement and defect to have one ID and one name for its level, so that I can cite and find it anywhere in the repo.
- As `tico`, `teco` or `architect`, I want to relate one item to another by ID across components and projects, so that overlap and dependency checks work.
- As the stakeholder starting a green-field project, I want its items to be unique alongside this repo's, so that the shared pool and graph never confuse the two.

## Functional requirements
- FR-1: A glossary defines the levels of work — initiative → feature → story → task — each with one formal term, as are the other item types (requirement, defect, kaizen); "epic" is an alias of feature, and any synonym is listed as an alias and never used in IDs.
- FR-2: Every item has an ID made of a project prefix followed by one typed segment per level, e.g. `GAL-F0012`; the whole string is unique across projects.
- FR-2a: Initiatives, features, defects and kaizen items are counted per project per type, so a segment such as `F0012` identifies one item in the whole project whatever its parent. Stories and tasks are counted locally from `01` under their parent (`S04` is the 4th story of its feature), so they are unique only as part of their full path.
- FR-3: A feature with no initiative skips that level (e.g. `GAL-F0012`); the type letters of each segment (one or two) name its level. A story or task always has a parent: one-off work with no feature is a kaizen item or a defect.
- FR-4: The path is the identity: when an item changes parent, it gets a new ID and the old ID stays as an alias.
- FR-5 (no duplicates): Two agents creating items concurrently never receive the same ID — in one working copy or on separate branches, and including two items added under the same parent.
- FR-6 (no reuse): An ID is never handed out again, even if its item is deleted, abandoned or renamed; every old ID resolves to the item or to an alias for it.
- FR-7 (cheap allocation): An agent obtains the next ID without scanning every backlog and requirements document.
- FR-8 (single lookup): One place answers what an ID is, where it lives and what it was called before.
- FR-9 (checkable): A check can find an item with no ID, a malformed ID, or one ID on two items, among the items in the scheme.
- FR-10: FR and AC are numbered under the same scheme as path descendants of the feature they belong to (e.g. `GAL-I03-F0012-AC3`, or `GAL-F0012-AC3` for a feature under no initiative), segment letters `FR` and `AC`; they are counted per feature and keep the numbers they already have in the requirements document, so none is renumbered. They are renamed with the feature (FR-24).
- FR-11: Defects (letter `D`) live for life under the reserved Defects initiative (`GAL-I01-D0015`). The feature, story or task a defect concerns is a recorded link, empty until diagnosed and changeable without renaming the defect; a defect is never a child of what it concerns. The fix is linked back to the defect.
- FR-12: The scheme applies to new work from the day it ships. An existing `K-`/`C-` item receives a `GAL-` ID when an item under the scheme first links to or cites it, or when work on it is first opened under the scheme. Minting is idempotent: one legacy item never has two `GAL-` IDs, and a second request returns the first. Its old name, qualified by the component or agent where it lived (e.g. `architect:K-004`), is an alias resolving to it; an unambiguous bare name (e.g. `C-310`) resolves too, and an ambiguous one is reported as ambiguous. Existing documents are not rewritten and nothing is bulk-migrated. FR-6 and FR-9 apply to items already in the scheme.
- FR-13: A lightweight improvement ("kaizen") is an item type of its own, letter `K`, living under the reserved Kaizen initiative (`GAL-I02-K0015`). When it grows into a feature it is renamed (e.g. `GAL-F0040`, under an initiative chosen at promotion or none) and the old `K` ID stays as an alias, per FR-4.
- FR-14: Every project reserves initiative numbers `I01` (Defects) and `I02` (Kaizen), so these two have the same ID in every project; other initiatives are numbered from `I03`.
- FR-15: Containment is expressed by the ID path; every other relation between items (e.g. the feature, story or task a defect concerns) is a recorded link, not part of the path.
- FR-16: An item's milestone is a recorded attribute, never part of its ID; changing it renames nothing.
- FR-17: There is one project key per repo, shared by all its components; this repo's key is `GAL`.
- FR-18: A new project's key is chosen with the stakeholder in the project's first session, the agent proposing one; a key already used by any project that feeds the shared pool is never proposed or accepted; one list of the keys in use, shared by all projects, is consulted.
- FR-19: Each project keeps a fixed list of components; every item has exactly one home component (where its documents live) and any number of touched components, recorded as links and never part of the ID.
- FR-20: The lookup of FR-8 is the authority for aliases, including those of pre-existing `K-`/`C-` items. An item's own header also lists its former IDs for human readers (`Aliases:`), and the aliases can be recovered from those headers if the lookup is lost. Following an alias resolves to the item's current ID and reports it.
- FR-21: Numbers are zero-padded to a minimum width — initiative 2 digits, feature, defect and kaizen 4, story and task 2 — and grow a digit on overflow without renaming anything; FR and AC numbers are unpadded and counted per feature.
- FR-22: Moving a story or task to another parent renames it, to a parent FR-25 allows, with the next free local number under the new parent; the old ID stays as an alias (FR-4).
- FR-23: A feature's ID and its topic slug resolve to each other through the lookup of FR-8, and every document of the feature's family carries the feature ID in its header, assigned when the requirements document is created; filenames are unchanged.
- FR-24: When an item is renamed (FR-4, FR-13, FR-22), its stories, tasks and other descendants are renamed with it, each old ID staying as an alias; their local numbers do not change.
- FR-25: Allowed parents: an initiative has the project; a feature has an initiative from `I03` up, or none, and never `I01` or `I02`; a story has a feature only; a task has a story, a feature, a defect or a kaizen item; a defect has `I01` only; a kaizen item has `I02` only; FR and AC have a feature. Any other parent is refused.

## Out of scope
- How IDs are allocated, stored or looked up (registry, allocator, checks): design, for the architect.
- Rewriting existing documents, or migrating existing `K-`/`C-` items to new IDs (FR-12).
- Status, priority, estimate or workflow rules for items; the `Status:` vocabulary for documents is unchanged.
- Legacy / existing-code projects (deferred in `claude/docs/requirements/team-engagement-in-other-repos.md`).
- Renaming files to carry IDs.
- Amending the root `AGENTS.md` conventions this standard touches (header fields, `K-` numbering); carried out when the feature closes.

## Acceptance criteria
- AC-1 (FR-1 to FR-3, FR-21): Given initiative `I03`, when a feature, a story under it and a task under that story are created, then their IDs are `GAL-I03-F<4 digits>`, `…-S01`, `…-T01`; a feature created under no initiative is `GAL-F<4 digits>`.
- AC-2 (FR-5): Given two agents creating items concurrently — two features, or two tasks under one story, in one working copy or on separate branches that are then merged — then all IDs are distinct.
- AC-3 (FR-4, FR-20, FR-22, FR-24): Given a feature moved to another initiative, or a story moved to another feature, then it has a new ID, every story and task under it is renamed with it keeping its local numbers, each old ID resolves to the current one and reports it, and a second move still resolves from the first ID.
- AC-4 (FR-6): Given a deleted or abandoned item, then its number is never issued again and its ID still resolves.
- AC-5 (FR-7): Given any number of backlogs and requirements documents, then obtaining a new ID requires reading none of them.
- AC-6 (FR-8, FR-12): Given `architect:K-004` and `devops:K-004`, two different items, when each is first cited by an item under the scheme, then each receives one `GAL-` ID (a second citation returns the same ID) and its qualified old name resolves to it; bare `K-004` is reported as ambiguous and bare `C-310` resolves; no existing document is edited. The same holds when the legacy item is first cited and when work on it is first opened.
- AC-7 (FR-9): Given a set containing an item without an ID, one with a malformed ID and two items sharing an ID, then the check reports all three.
- AC-8 (FR-11, FR-15): Given a new defect, then it lives under `GAL-I01`; when the feature, story or task it concerns is set and later changed, its ID does not change.
- AC-9 (FR-13, FR-24): Given a kaizen item with tasks promoted to a feature, then it has a new feature ID, its tasks follow, and the old `K` ID and the old task IDs resolve to the current ones.
- AC-10 (FR-14, FR-17, FR-18): Given a new project, then its key is proposed to and agreed with the stakeholder in its first session, a key already in use is refused, and `I01`/`I02` are Defects/Kaizen in it.
- AC-11 (FR-16, FR-19): Given an item, then it has one home component from the project's list and any number of touched components; changing its milestone or its home component changes no ID.
- AC-12 (FR-20): Given the lookup lost, then every alias recorded in an item header can be recovered and resolves as before.
- AC-13 (FR-21): Given a feature whose 100th story is created, then it is `S100`, and no earlier ID changes.
- AC-14 (FR-23): Given a feature's ID, then the lookup returns its slug, and given the slug, its ID; the ID appears in the header of each family document.
- AC-15 (FR-1): Given the glossary, then it defines each level and type with one formal term and lists "epic" as an alias of feature.
- AC-16 (FR-2a, FR-3, FR-25): Given two features each with stories, then each feature's stories start at `S01`; every parent/child pair FR-25 allows is accepted and every other pair is refused, including a feature under `I01`, a story under a defect or kaizen item, and a story or task with no parent.
- AC-17 (FR-10, FR-24): Given a feature's requirements document, then each FR and AC is citable by its full path with the number it already has in the document; when the feature moves to another initiative, each is renamed with it and the old citation resolves to the current one.

## Open questions
None.

## Decision log
2026-11-04 — What triggered this? → Several at once: ID confusion, coordination needing a stable key, wanting a repo-wide view, and the green-field requirement needing a robust team/coordination standard (Intent, Problem).
2026-11-04 — Which kinds of item get a standard ID and name? → Hierarchy of work levels, plus FR, AC and defects/backlog items under the same scheme (FR-1, FR-10).
2026-11-04 — How is an ID made unique across repos? → A project prefix, then a composable hierarchical string, one segment per level (FR-2).
2026-11-04 — Is identity permanent, or the path? → The path is the identity: a move renames the item and the old ID stays as an alias (FR-4).
2026-11-04 — How is an item with no natural parent identified? → The missing level is skipped (FR-3).
2026-11-04 — Which numbering guarantees are required? → All five: no duplicates under concurrency, no reuse, cheap allocation, single lookup, checkable (FR-5 to FR-9).
2026-11-04 — Which levels, and which word? → Initiative → feature → story → task; "feature" is the formal word, "epic" an alias (FR-1).
2026-11-04 — Counter scope: per parent or per project? → Unique across the project, per type (FR-2a).
2026-11-04 — Where do defects sit? → Attached to the introducing feature, parentless until diagnosed, fix is a task under the defect (FR-11).
2026-11-04 — What happens to existing `K-`/`C-` IDs? → Alias only; the new scheme is for new work; old documents are not rewritten (FR-12).
2026-11-04 — Is a kaizen item a type, and what happens when it becomes a feature? → A type of its own (`K`); on promotion it is renamed and the old K ID stays as an alias (FR-13).
2026-11-04 — Standing initiatives for defects and kaizen, and how is a defect tied to its feature? → Two reserved initiatives, I01 Defects and I02 Kaizen, each the permanent home of its items; the introducing feature is a link, not a path segment. Overturns the entry "Where do defects sit?" as to placement (FR-3, FR-11, FR-13, FR-14, FR-15).
2026-11-04 — Is milestone part of the ID? → No, a recorded attribute (FR-16).
2026-11-04 — One project key per repo or per component? → One per repo; this repo's key is `GML` (FR-17).
2026-11-04 — How is a new project's key chosen? → Together with the stakeholder in the project's first session, the agent suggesting (FR-18).
2026-11-04 — Is the key `GML`? → No, `GAL` (graphmind-ai-lab); overturns the entry "One project key per repo or per component?" as to the key only (FR-17).
2026-11-04 — How is a component represented? → Fixed list per project; one home component and any number of touched components, as links (FR-19).
2026-11-04 — Where do aliases live? → The lookup is the authority; the item's header also lists them; lookup rebuildable from headers; resolves to the current ID (FR-20).
2026-11-04 — Are story and task counters project-wide or local to the parent? → Local from 01 under the parent; initiative, feature, defect and kaizen stay project-wide; a story or task always has a parent (FR-2a, FR-3, FR-5, FR-22). Overturns the entry "Counter scope: per parent or per project?" for story and task.
2026-11-04 — Segment widths and separator? → Hyphen; initiative 2 digits, feature, defect and kaizen 4, story and task 2, growing on overflow (FR-21).
2026-11-04 — Do slugs and IDs resolve to each other? → Both ways, with the feature ID in each family document's header; filenames unchanged (FR-23).
2026-11-04 — Consult: analyst on readiness review (`docs/reviews/work-item-numbering.md`) → Pass 1 needs changes, Pass 2 needs changes (narrower); findings folded into FR-1, FR-5, FR-10 to FR-13, FR-18, FR-20, FR-22 to FR-25, AC-2, AC-3, AC-6, AC-8, AC-9, AC-12, AC-15 to AC-17.
2026-11-04 — Legacy `K-`/`C-` items: ID on first touch or separate namespace? → A `GAL-` ID on first touch, old name qualified by where it lived becomes an alias; overturns the entry "What happens to existing `K-`/`C-` IDs?" in part (FR-12, AC-6).
2026-11-04 — Do descendants follow a renamed parent? → Yes, renamed with it, old IDs aliased, local numbers unchanged (FR-24, AC-3, AC-9).
2026-11-04 — Does rename-on-move still hold with project-unique feature numbers? → Yes, one rule for every type: the full string is the identity; a feature moving initiative gets a new string and an alias (FR-4).
2026-11-04 — Can a defect be a child of a feature or story, e.g. when a whole story is missing? → No: link-only; this widens the link target of the entry "Standing initiatives for defects and kaizen…" (feature only) — the link may target a feature, story or task, and a missing story is created under the feature and linked as the fix (FR-11, FR-15).
2026-11-04 — Parent legality, incl. a task directly under a feature and no story under a defect or kaizen? → Confirmed table (FR-25, AC-16).
2026-11-04 — Are FR and AC path descendants (renamed with a moving feature) or cited by the feature's stable segment? → Path descendants, one rule (FR-10, FR-24, AC-17).
2026-11-04 — Legacy ID minted when an item under the scheme first links to or cites it, or work on it is first opened? → Yes (FR-12).
2026-11-04 — Readback of the final document confirmed by the stakeholder; analyst Pass 3 verdict approve with suggestions, folded in → Status flipped to Ready for design.
