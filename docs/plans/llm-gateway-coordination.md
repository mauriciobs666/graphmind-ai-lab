# Common LLM gateway — Coordination

> **Status:** active · **Owner:** `teco` · **Tracks:** `GAL-F0046` · **Reviews:** `docs/reviews/llm-gateway.md`

Requirements (Ready for design): `docs/requirements/llm-gateway.md`; extension: `docs/requirements/llm-gateway-usage-stats.md`. Scope of this coordination: design of the first delivery (ccg), then build units drawn from the plan.

| Unit | Owner | Agent id | Status | Deliverable | Gate → verdict | Cost |
|---|---|---|---|---|---|---|
| U1 design | `architect` | `a5d2d6277492067da` | accepted (rev 6; analyst Pass 9 stopping condition met, security Pass 4 approve w/ suggestions; word count and both edits verified by teco) | `docs/plans/llm-gateway.md` | `analyst` `aa13ebf1ebab68708` + `security-expert` `a7b80d0ec280cf976` (FR-8) → rev 1 needs changes (both); rev 2 re-gate: analyst Pass 6, security Pass 2 | — |
| U0 req amend | `tico` | `ab935e9e2d6a3ab05` | delivered | `docs/requirements/llm-gateway.md` (FR-15: Ollama first-delivery, stakeholder 2026-11-23) | — → — | — |

2026-11-23: Ollama scope change relayed to U1 architect by SendMessage while in flight; U1's plan must cover it.
| U0b req amend FR-10/FR-3 | `tico` | `a4be0d204f90d4bd9` | delivered | `docs/requirements/llm-gateway.md` | — → — | — |

2026-11-23 stakeholder answers: HMAC kept, same-uid residual accepted (D-1); holder-side ceiling adopted (FR-10, reverses D-6); registered scopes only (FR-3, resolves D-7). Gates: analyst Pass 5 and `docs/reviews/llm-gateway-security.md` both "needs changes"; U1 revision relayed to architect by SendMessage.
Open: FR-8 / AC-5 wording ("no agent can reach", "direct call by anything but broker/service is refused") overclaims against the accepted same-uid HMAC residual; to be reworded by `tico` once the revised plan fixes what FR-8 truthfully guarantees.
| U0c req reword FR-8/AC-5 | `tico` | `abc21a85f6242a29b` | delivered | `docs/requirements/llm-gateway.md` | — → — | — |

Pending stakeholder: D-2 (Ollama gaps), D-3, D-4, D-5 in `docs/plans/llm-gateway.md`; `tico` words D-3/D-4 in the requirements only after answers.
| U0d req amend FR-10 global ceiling, D-3, D-4 | `tico` | `a7fd92134498e5cff` | delivered | `docs/requirements/llm-gateway.md` | — → — | — |

2026-11-24: rev 2 re-gates both "needs changes" (analyst Pass 6: 3 new majors; security Pass 2: N1–N5). Stakeholder: holder-wide rolling ceiling adopted; D-2(a–g), D-3, D-4, D-5 accepted. Rev 3 relayed to architect; analyst's stopping condition (4 items) is the done-condition.
| U0e req reconcile AC-14/AC-2/AC-6 | `tico` | `a99451ad323911b3e` | delivered | `docs/requirements/llm-gateway.md` | — → — | — |

2026-11-24: rev 4 delivered (code root /opt/llmgateway, release-by-outcome, boot/service actors). Open assumption to verify at build: a complete provider 4xx/5xx reply is not billed.

Build phase (plan accepted, committed `749633ae`). Plan dispatch table is the unit list; ccg adoption (D11) waits for the P1-38 phase exit. D7's installer and tier-2 need a human `sudo` run.

| Unit | Owner | Agent id | Status | Deliverable | Gate → verdict | Cost |
|---|---|---|---|---|---|---|
| D1 | `coder` | `a1002519559900dfd` | delivered (20 tests re-run green by teco; 21 mutants killed per report; committed) | `llm-gateway/` scaffold + types | `analyst` (batched D1–D6 diff gate) → — | — |
| D2 | `coder` | `adf703ee1a37de980` | delivered (177 tests re-run green; teco mutation: dropping O_NOFOLLOW on file → test fails; committed) | `ownercheck.py`, `config.py`, `failures.py` + tests | `analyst` (batched D1–D6) → — | — |
| D3 | `coder` | `a2909bc9c6200a30e` | delivered (86 D3 tests re-run green; teco mutation: dropping scope_required → test fails; 34 mutants killed per report; committed b12c7b46) | `records.py`, `policy.py` + tests | `analyst` (batched D1–D6) → — | — |
| D7 | `coder` | `aba8fe4bb4e2cc4a7` | delivered (867 tests full-suite re-run green; teco mutation: window_max_released >= → > fails; ~60 mutants killed per report; committed; tier-2 pending human sudo) | `framing.py`, `keyholder.py`, `ledger.py`, `bin/keyholder.py`, `deploy/install.sh` + tests | `analyst` + `security-expert` (own gate) → — | — |

Note: D3 ∥ D7 file-disjoint (D7 does not touch records/policy); `llm-gateway/AGENTS.md` "State" line stale — teco updates after D6.
| D4 | `coder` | `ab53d36fa731b7a6b` | delivered (87 tests re-run green; teco mutation: DirectTransport locality check disabled → test fails; 18 mutants killed per report; committed 50a3820e) | `engine.py`, `transport.py`, `providers/{__init__,base}.py` + tests | `analyst` (batched D1–D6) → — | — |
| D5 | `coder` | `abe1111f3fafcec78` | delivered (130 D5-area tests re-run green; teco mutation: Anthropic cache-token settlement removed → test fails; 36 mutants killed per report; committed 6b2e51c9, `test_providers_holder_agreement.py` held until D7 commits) | `providers/{openai_compat,lmstudio,ollama,anthropic}.py` + tests | `analyst` (batched D1–D6) → — | — |
| D6 | `coder` | `a31443104be371b47` | delivered (517 non-D7 tests re-run green; teco mutation: embedding filter removed → test fails; 18 mutants killed per report; committed) | `load.py`, `engine.py` (edit) + tests | `analyst` (batched D1–D6) → — | — |

Open for stakeholder (from D5): Anthropic adapter refuses `tools` (OpenAI-style tool-call translation not built) — FR-1 says chat completions with tool calls; ccg does not use tools, falkor-chat follow-on will. `min_p` unsupported on lmstudio/openai_compat and Ollama frequency/presence penalties unsupported (vendor docs do not list them; ccg does not send them). Unverified vendor facts listed in D5 report: LM Studio `stats`/`runtime` on /v1, `/v1/decisions` shape, Anthropic temperature family and thinking-token inclusion, Ollama `eval_count` thinking and `num_ctx` effect.
| Gate D1–D6 | `analyst` | `aa88268ad17048330` | in-flight | `docs/reviews/llm-gateway.md` Pass 10 | — → — | — |
| U-S5 | `architect` | `a6c231652670eed13` | accepted (analyst Pass 11-12 approve w/ suggestions, security Pass 6 approve w/ suggestions; committed) | `docs/plans/connected-code-graph-egress-spec5.md` | `analyst` + `security-expert` → — | — |
| Gate D7 + spec5 (security) | `security-expert` | `a7b80d0ec280cf976` | in-flight | `docs/reviews/llm-gateway-security.md` Pass 5, 6 | — → — | — |
| Gate D7 + spec5 (analyst) | `analyst` | `aa88268ad17048330` | queued | after Pass 10; `docs/reviews/llm-gateway.md` | — → — | — |

Known defect carried from spec5 review: holder matches `endpoint` to raw `base_url`; broker sends canonical `https://host:443` — fix in D7 follow-up (after security says where). `service-unit` writes `ExecStart=… /opt/llmgateway/bin/service.py`; D8 must ship that shim. D9 needs 5 files (spec5 §8), not 2.
| D7b holder fixes | `coder` | `a2a53606d76208971` | in-flight | security Pass 5 P5-1..P5-5 (endpoint canonicalisation, install.sh preview/read-as-user, code-root precheck, reserve/release cap, service-unit guard) | `security-expert` re-check → — | — |
| U-S5 rev | `architect` | `a6c231652670eed13` | in-flight | spec5 revised for security Pass 6 | `analyst` → — | — |

2026-11-24 security Pass 5 (D7 code) and Pass 6 (spec5): both approve with suggestions; P5-1 and P5-2 majors routed to D7b.
| D1–D6 gate | `analyst` | `aa88268ad17048330` | gated: approve with suggestions (Pass 10; F1–F3 majors) | `docs/reviews/llm-gateway.md` Pass 10 | → fixes in D456fix | — |
| D456fix | `coder` | `a790ddded54ac051a` | in-flight | Pass 10 F1–F7 (run_id validation, map_settings tests, docs, overrides, is_hosted unify) | `analyst` re-check → — | — |
| spec5 gate (analyst) | `analyst` | `aa88268ad17048330` | gated: approve with suggestions (Pass 11; F1–F3 majors → spec5 revision `a6c231652670eed13`) | `docs/reviews/llm-gateway.md` Pass 11 | — → — | — |
| plan absorb (shapes, grammar note) | `architect` | `a5d2d6277492067da` | in-flight | `docs/plans/llm-gateway.md` | — → — | — |
| FR-18 reword | `tico` | `a5d4e249e30706097` | delivered (ccg checked: json_schema only) | `docs/requirements/llm-gateway.md` | — → — | — |
