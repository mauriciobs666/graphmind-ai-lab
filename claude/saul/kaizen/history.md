# Kaizen — Change History: saul

> Dated log of actual changes to the `saul` agent. Most recent first.

## 2026-06-21 — dossier storage moved to $AGENT_WORKDIR; Bash added
- **What:** changed the dossier storage root from `dossies/{cliente}/{caso}.md` (project-root relative) to **`$AGENT_WORKDIR/saul/dossies/{cliente}/{caso}.md`**. Added `Bash` to the frontmatter `tools` (now `Read, Write, Edit, Bash, WebFetch, WebSearch`) so Saul can resolve the env var (`printf '%s\n' "$AGENT_WORKDIR"`) at session start — `Read`/`Write`/`Edit` don't expand shell variables. Added a fail-safe: if `$AGENT_WORKDIR` is empty/undefined, Saul must ask the user for the base dir before writing. Replaced the stray `Glob` reference (never in his toolset) with `Read` in the existence-check step.
- **Why:** user wants dossiês to live under a configurable agent working directory instead of the repo root. Resolving an env var requires shell access; user chose to grant `Bash` over harness-side expansion or per-session prompting.
- **Docs touched:** `claude/AGENTS.md` (path + tools), `claude/README.md` (path), root `AGENTS.md` (path).

## 2026-06-21 — created
- **What:** created the `saul` subagent — Brazilian legal assistant, São Paulo forum, specialized in civil and criminal law with deep focus on **direito condominial (condomínio edilício)**. Frontmatter `model: opus`, `tools: Read, Write, Edit, WebFetch, WebSearch`. Core behaviors: source-discipline (explicit 🟢 verified / 🟡 thesis / 🔴 to-confirm labeling; cite dispositivos; WebSearch+WebFetch official sources before asserting vigência; never fabricate jurisprudence), dossier strategy + management at `dossies/{cliente}/{caso}.md`, minuta drafting with mandatory RASCUNHO tarja, and the OAB "not a substitute for a licensed lawyer" caveat. Áreas section is base civil+penal with condominial deepened and an explicit on-demand extension path for Consumidor/Família-Sucessões/Trabalhista.
- **Why:** user requested a civil+criminal SP legal specialist; clarified the immediate need is condomínio predial (confirmed civil — CC arts. 1.331–1.358-A, Lei 4.591/64, CPC 784,X) with other areas wanted later. Chose dossiê+estratégia+minutas scope and `dossies/{cliente}/{caso}.md` storage per user selection. Structure mirrors the repo's `dra-claudia` per-record markdown pattern.
- **Plan items:** seeded K-001..K-004.
- **Docs touched:** `claude/README.md` (catalog row + kaizen link), `claude/AGENTS.md` (agent entry), root `AGENTS.md` (Claude Code subagents table).
