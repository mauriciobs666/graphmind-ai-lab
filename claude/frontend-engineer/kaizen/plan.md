# Kaizen — Improvement Plan: frontend-engineer

> Forward-looking backlog for the `frontend-engineer` agent.
> Status: 🔵 proposed · 🟡 in-progress · ✅ done (then moved to history.md) · ⚪ rejected/deferred
> Last reviewed: 2026-07-09

## Active

| ID | Added | Priority | Status | Summary |
|------|------------|----------|--------|---------|
| K-001 | 2026-07-09 | medium | 🔵 | First real-run shakedown on a UI task in this repo |
| K-002 | 2026-07-09 | low | 🔵 | Visual verification tooling (screenshots/browser automation) |

### K-001 — First real-run shakedown
- **Status:** 🔵 proposed
- **Priority:** medium
- **Rationale:** the prompt is untested against a real task; first runs usually surface routing gaps (vs. coder) and missing repo specifics.
- **Proposed change:** delegate a real UI task (e.g. a salesperson Streamlit change or the future falkor-chat front-end), observe where the agent hesitates or drifts out of scope, and fold fixes back into the prompt.
- **Notes:** watch especially whether the coder↔frontend-engineer efficiency boundary routes correctly from teco.

### K-002 — Visual verification tooling
- **Status:** 🔵 proposed
- **Priority:** low
- **Rationale:** "verify in the running UI" currently relies on dev-server output and console checks; real screenshot/browser-drive tooling (Playwright, a browser MCP server) would make visual claims evidence-backed.
- **Proposed change:** once the lab standardizes a browser-automation tool, wire it into the prompt's verification step (and consider an `mcpServers` frontmatter entry).
- **Notes:** keep it optional — Streamlit apps may only need `streamlit run` + manual checks.

## Parking lot / ideas
- **Judged and kept, do not re-litigate (2026-08-24, C6 lint).**
  - **`today that's concretely `cpg_salesperson` for `salesperson/chatbot.py`` (`:18`)** and **"In *this*
    repo the running UIs are Streamlit apps" (`:20`)** — kept under the plan's **finding 10**, whose
    test is *scope above rot*: `devops`'s repo example was deleted because that agent is
    **user-scoped**, making a one-repo snapshot a false anchor everywhere else. `frontend-engineer`
    is project-scoped, so a repo fact is a true anchor, and neither line carries a standing
    "refresh it" obligation of the kind finding 10 warns about.
  - **"`falkor-chat/` may grow a web front-end — check its docs before assuming a stack for it"
    (`:20`)** — reads as speculation but functions as a class-1 **anti-trigger**: without it the
    agent's nearest precedent (salesperson) would have it assume Streamlit for falkor-chat.
  - **"Every UI state is a requirement" (`:75`)** vs. step 3 vs. the data-fetching bullet — three
    touches, kept. Cutting one principle bullet for ~17 w would be a structural change out of step
    with every other agent's principles list.
- **Watch note — `cpg_salesperson` now lives in three places that rot together (2026-08-24, C6 lint).**
  The frontmatter `description` ("`cpg_salesperson` today"), `:18`, and the surrounding repo fact at
  `:20`. Different readers justify it (routers read the description, the agent reads the body), so it
  stands today — but if `salesperson/chatbot.py` ever stops being this repo's Streamlit entry point
  it is a **three-site** update, which is precisely the shape finding 10 says becomes a deletion
  candidate on its third rot. Recorded so the count is visible next time rather than rediscovered.
- A perishable "framework quirks" resource file (like graph-dba's `falkordb-quirks.md`) if the lab settles on one web framework and version-specific gotchas accumulate.
- Design-system/token conventions section if the lab adopts one.
