# Kaizen — Improvement Plan: frontend-engineer

> Forward-looking backlog for the `frontend-engineer` agent.
> Status: 🔵 proposed · 🟡 in-progress · ✅ done (then moved to history.md) · ⚪ rejected/deferred
> Last reviewed: 2026-09-02

## Active


### K-001 — First real-run shakedown

- **Added:** 2026-07-09
- **Status:** 🔵 proposed
- **Priority:** medium
- **Rationale:** the prompt is untested against a real task; first runs usually surface routing gaps (vs. coder) and missing repo specifics.
- **Proposed change:** delegate a real UI task (the `salesperson-ui` storefront SPA is the first one in flight), observe where the agent hesitates or drifts out of scope, and fold fixes back into the prompt.
- **Notes:** watch especially whether the coder↔frontend-engineer efficiency boundary routes correctly from teco.

> Summary: First real-run shakedown on a UI task in this repo

### K-002 — Visual verification tooling

- **Added:** 2026-07-09
- **Status:** 🔵 proposed
- **Priority:** low
- **Rationale:** "verify in the running UI" currently relies on dev-server output and console checks; real screenshot/browser-drive tooling (Playwright, a browser MCP server) would make visual claims evidence-backed.
- **Proposed change:** once the lab standardizes a browser-automation tool, wire it into the prompt's verification step (and consider an `mcpServers` frontmatter entry).
- **Notes:** keep it optional — Streamlit apps may only need `streamlit run` + manual checks.

> Summary: Visual verification tooling (screenshots/browser automation)

## Parking lot / ideas

### GAL-I02-K0012 — `cpg_salesperson` three-site rot: fixed by deletion, not by a fresher pointer (2026-09-02)

- **Status:** ⚪ rejected — record, not work (settled 2026-09-02)

The 2026-08-24 C6 lint kept `:18`'s concrete `cpg_salesperson` pointer
and `:20`'s "the running UIs are Streamlit apps" under the *scope above rot* test, and recorded a
watch note that the pair plus the frontmatter `description` were a **three-site** update which
would become a deletion candidate on its third rot. That rot is the `salesperson-ui` coordination:
the Streamlit app is retired, the `salesperson/` name is reassigned to a React storefront, and
`cpg_salesperson` now holds a retired codebase under a name that suggests the new one. All three
sites were rewritten (see `history.md`, 2026-09-02) to statements that cannot rot on a rename —
*the CPGs here are Python-only, so no front-end source is in one* — rather than to new pointers.
**The finding to carry forward:** *scope above rot* justified keeping a concrete repo fact, but it
did not justify keeping one whose truth depended on a **path**. A repo fact anchored to a durable
property survives; one anchored to a directory name is a countdown.

### GAL-I02-K0013 — Anti-trigger clause and "every UI state" bullet kept after the C6 lint

- **Status:** ⚪ rejected — record, not work (judged and kept 2026-08-24)

The anti-trigger function of the old "`falkor-chat/` may grow a web
front-end — check its docs" clause is preserved in `:20`'s replacement ("Different products,
different bars — never take one as the stack precedent for the other" plus the retired-Streamlit
anti-precedent); and **"Every UI state is a requirement" (`:75`)** vs. step 3 vs. the data-fetching
bullet — three touches, kept, for the reason given in 2026-08-24: cutting one principle bullet for
~17 w would be a structural change out of step with every other agent's principles list.

### GAL-I02-K0014 — "Python-native UIs" section may be dead weight (watch, 2026-09-02)

- **Status:** 🔵 proposed — parking-lot origin; owner confirms status

With the repo's
only Streamlit app retired, `:9`, the frontmatter `description`'s "Python-native UIs like
Streamlit" clause, and the `:51` Streamlit-fluency section are pure discipline breadth with no
live surface in this repo. Not cut in U6 (a correction pass, not a redesign) and arguably correct
to keep — the agent is not repo-locked and Streamlit is a real front-end skill. Revisit only if a
token budget forces a choice, or if the description's Streamlit clause is ever observed
*misrouting* a dispatch.

### GAL-I02-K0015 — Perishable framework-quirks resource file (`frontend-quirks.md`)

- **Status:** ⚪ rejected — record, not work (delivered 2026-09-16, see `history.md`)

The file this bullet anticipated
was created 2026-09-16: `claude/frontend-engineer/frontend-quirks.md`, seeded from 7 distilled `kaizen_team`
entries (React/TS/Vite/TanStack Query/i18next/RTL gotchas). See `history.md`, 2026-09-16.

### GAL-I02-K0016 — Design-system/token conventions section if the lab adopts one

- **Status:** 🔵 proposed — parking-lot origin; owner confirms status

Design-system/token conventions section if the lab adopts one.
