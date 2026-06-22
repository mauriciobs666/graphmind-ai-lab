# Kaizen — Improvement Plan: saul

> Forward-looking backlog for the `saul` agent.
> Status: 🔵 proposed · 🟡 in-progress · ✅ done (then moved to history.md) · ⚪ rejected/deferred
> Last reviewed: 2026-06-21 (dossier root → `$AGENT_WORKDIR/saul/dossies/`; `Bash` granted)

## Active

| ID | Added | Priority | Status | Summary |
|------|------------|----------|--------|---------|
| K-001 | 2026-06-21 | med | 🔵 | Activate connected areas (Consumidor/CDC, Família e Sucessões, Trabalhista) as the user opts in |
| K-002 | 2026-06-21 | med | 🔵 | Add WebFetch domain allowlist for legal sources (planalto, stf, stj, tjsp, al.sp) to settings.json so source-verification fetches don't prompt |
| K-003 | 2026-06-21 | low | 🔵 | Seed a small reusable library of minuta templates (inicial de cobrança/execução de cota, notificação extrajudicial, defesa) |
| K-004 | 2026-06-21 | low | 🔵 | Behavioral eval/bless harness for source-discipline (does it mark 🟢/🟡/🔴 correctly? does it refuse to fabricate jurisprudence?) per cobb/TESTING.md |
| K-005 | 2026-06-21 | med | 🔵 | Narrow the newly-granted `Bash` permission — Saul only needs it to resolve `$AGENT_WORKDIR`; consider a settings.json allow rule scoped to that (or `mkdir -p` for dossier dirs) instead of full shell |

### K-001 — Activate connected areas on demand
- **Status:** 🔵 proposed
- **Priority:** medium
- **Rationale:** user's immediate need is condomínio edilício (civil); explicitly wants other areas available later.
- **Proposed change:** when the user opts into Consumidor / Família-Sucessões / Trabalhista, deepen the "Áreas" section with that area's marco legal and dossier nuances (mirror the condominial depth).
- **Notes:** keep base civil+penal; don't bloat the prompt — consider a per-area skill if it grows.

### K-002 — Legal-source WebFetch allowlist
- **Status:** 🔵 proposed
- **Priority:** medium
- **Rationale:** Saul's source-discipline mandate means frequent fetches to official legal sites; prompts on each break the flow (same pattern as cobb's doc-domain allowlist).
- **Proposed change:** add `WebFetch(domain:planalto.gov.br)`, `in.gov.br`, `stf.jus.br`, `stj.jus.br`, `tjsp.jus.br`, `al.sp.gov.br`, `WebSearch` to `~/.claude/settings.json` permissions.allow. Personal/global file, not in repo.
- **Notes:** user-scope = session-wide; acceptable.

### K-003 — Minuta template library
- **Status:** 🔵 proposed
- **Priority:** low
- **Rationale:** the most common condominial pieces repeat; reusable skeletons save time and reduce drafting errors.
- **Proposed change:** small set under a skill or a `templates/` ref, each with the RASCUNHO tarja and [PREENCHER] markers.

### K-004 — Behavioral eval harness
- **Status:** 🔵 proposed
- **Priority:** low
- **Rationale:** the agent's value hinges on honest 🟢/🟡/🔴 labeling and never fabricating law/jurisprudence — exactly the kind of behavior a bless harness should pin.
- **Proposed change:** follow `claude/cobb/TESTING.md` eval/bless pattern with a few scripted scenarios.

### K-005 — Narrow the Bash grant
- **Status:** 🔵 proposed
- **Priority:** medium
- **Rationale:** `Bash` was added so Saul can resolve `$AGENT_WORKDIR`, but the grant is full shell — broader than needed for a doc-keeping legal agent.
- **Proposed change:** scope Bash via `~/.claude/settings.json` permissions (e.g. allow only the env-var read + `mkdir -p` of the dossier path), or revisit harness-side path expansion to drop Bash entirely.
- **Notes:** ties to K-002 (the WebFetch allowlist) — same settings.json.

## Parking lot / ideas
- Prazo/prescrição calculator helper (dossier could carry a computed deadlines block).
- Optional structured front-matter per dossiê for future graph ingestion (ties into the repo's FalkorDB theme).
