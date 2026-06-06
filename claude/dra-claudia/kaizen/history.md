# Kaizen — Change History: dra-claudia

> Dated log of actual changes to the `dra-claudia` agent. Most recent first.

## 2026-06-05 — Removeu o tom de gabarolice (tempo de experiência / volume de pacientes)
- **What:** Tirou "com décadas de experiência clínica... tendo atendido dezenas de milhares de pacientes" do `description` e "com mais de três décadas de prática clínica... Atendeu dezenas de milhares de pacientes em consultório próprio" da primeira linha do corpo. Mantidas as qualificações substantivas (formação em medicina convencional, especialização em homeopatia pela AMHB).
- **Why:** Feedback do usuário — o enquadramento de "décadas de experiência" soa convencido e não altera o comportamento. Aplicado em toda a coleção (também graph-dba e tdd-engineer).
- **Plan items:** —

## 2026-05-31 — Rebrand from `medicina-alternativa` to `dra-claudia`
- **What:** Renamed the agent folder and file (`medicina-alternativa/medicina-alternativa.md` → `dra-claudia/dra-claudia.md`), changed the `name:` frontmatter to `dra-claudia`, and gave the persona an explicit name — **Dra. Cláudia** — in the `description` and the opening line of the system prompt. Seeded kaizen files (none existed before). Updated `README.md` and `CLAUDE.md` catalogs.
- **Why:** User requested the rebrand. The clinical behavior (anamnese, prontuário management, red-flag triage) is unchanged; only the identity/branding moved.
- **Plan items:** —
