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
| D456fix | `coder` | `a790ddded54ac051a` | delivered (968 tests green at the time; 17 mutants killed per report; committed 7456b430) | Pass 10 F1–F7 (run_id validation, map_settings tests, docs, overrides, is_hosted unify) | `analyst` re-check → — | — |
| spec5 gate (analyst) | `analyst` | `aa88268ad17048330` | gated: approve with suggestions (Pass 11; F1–F3 majors → spec5 revision `a6c231652670eed13`) | `docs/reviews/llm-gateway.md` Pass 11 | — → — | — |
| plan absorb (shapes, grammar note) | `architect` | `a5d2d6277492067da` | in-flight | `docs/plans/llm-gateway.md` | — → — | — |
| FR-18 reword | `tico` | `a5d4e249e30706097` | delivered (ccg checked: json_schema only) | `docs/requirements/llm-gateway.md` | — → — | — |
| D7b | `coder` | `a2a53606d76208971` | delivered (972 tests re-run green; ~21 mutants killed per report; keyed-local-origin refuses start; committed f9aaf0d1) | security Pass 5 fixes + wire vectors | `security-expert` Pass 7 + `analyst` Pass 13 in-flight | — |
| D8 | `coder` | `a815b5fa2b9c9815c` | delivered (1095 tests re-run green, ruff clean; teco mutation: DirectTransport provider-table hosted check removed → test fails; 16 mutants killed per report; P7-1 closed; committed) | `service.py`, `client.py`, `transport.py` (HolderTransport), `bin/service.py` + tests | `analyst` → — | — |
| D9a | `coder` | `a3b35093dc5e27523` | delivered (cpg broker tests 1133 re-run green; teco mutation: run-id length regex widened → test fails; 27 mutants per report; `gateway` extra removed by teco: dependency-confusion risk, spec5 §8 text to amend; committed) | `cpg/ccg/broker/{holderclient,policy,server}.py`, `cpg/pyproject.toml` + tests (ccg impact: unavailable, anchor-unresolved) | `analyst` + `security-expert` → — | — |
D7b security re-check: Pass 7 approve with suggestions; P5-1/2/3/5 closed, P5-4 accepted residual. Open minors: `calls.log` size-based rotation (backlog); P7-1 (DirectTransport/`load.py` hosted flag) routed to D8 (relayed to `a815b5fa2b9c9815c`). Tier-2 on D7 code may be attempted by the human (bootstrap, `holder-account`, `owner-files`, `caller`, `check`, AC-5 probe); end-to-end hosted waits for D8 + D9.
| D7 gate (analyst) | `analyst` | `aa88268ad17048330` | gated: approve with suggestions (Pass 13; H1 major run_bounds match unpinned) | `docs/reviews/llm-gateway.md` Pass 13 | → D7c |
| D7c | `coder` | `a2ecc7b066645a497` | delivered (1095 tests re-run green, ruff clean; 22 mutants killed per report; committed) | H1 run_bounds tests, H2 refusal branch, H3 body-key narrowing, lint, `calls.log` rotation | `analyst` re-check → — | — |
| plan absorb H1/H4 | `architect` | `a5d2d6277492067da` | delivered (5,035 words; committed) | `docs/plans/llm-gateway.md` §3.4 | — → — | — |
D9a notes for D9b/D10/D13: default-port hosted test cannot listen on 443 so asserts signed `upstream-error` with `bytesIn>0`, not a 499 (differs from spec5 item 12 wording); D9b must add the real-subprocess variant of item 12, credential/fake-key scan over argv and /proc, items 8/9 start-up tests, and the static hosted-entry check will break integration fixtures that rely on loopback hosted entries (need an injected seam); spec5 §8 'gateway extra' must become 'editable install from the checkout, no extra'.
| D10 | `coder` | `aaa181f71095e2653` | delivered (gateway tests re-run green; 19 mutants killed per report; committed; uses private `BrokerClient._call` — close_reason not exposed by `BrokerClient.request`: follow-up before D11; one unexplained 15-min hang in a combined run, not reproduced) | `cpg/ccg/gateway/{__init__,brokertransport,manifestmarkers}.py` + tests | `analyst` → — | — |
| D9a gate (analyst) | `analyst` | `aa88268ad17048330` | in-flight | `docs/reviews/llm-gateway.md` Pass 14 | — → — | — |
| D9a gate (security) | `security-expert` | `a7b80d0ec280cf976` | in-flight | `docs/reviews/llm-gateway-security.md` Pass 8 | — → — | — |
| spec5 amend (D9a lessons) | `architect` | `a6c231652670eed13` | delivered (4,212 words; committed) | `docs/plans/connected-code-graph-egress-spec5.md` | — → — | — |
| D9b | `coder` | `a03ff974185d6bebb` | delivered (1507 broker+manifest+gateway tests re-run green; teco mutation: holder-uid≠own-uid check removed → test fails; 29 mutants killed, 1 equivalent per report; committed 7bb28f9c; `HolderClient` ctor uid check NOT added — main-only enforcement, accepted) | `cpg/ccg/broker/{ownerprot,server}.py`, `cpg/ccg/manifest.py` + tests (ccg impact: unavailable) | `analyst` + `security-expert` → — | — |
| D8 gates | `analyst` `aa88268ad17048330` (Pass 15), `security-expert` `a7b80d0ec280cf976` (Pass 9) | — | queued behind D9a gates | `docs/reviews/…` | — → — | — |
| D8b | `coder` | `a3a998384646de428` | delivered (1113 llm-gateway tests + 59 holderclient tests re-run green, ruff clean; ~30 mutants killed per report; committed 291316f5; client sha re-pinned 0d1424d2…) | security Pass 8/9 minors: client SO_PEERCRED + deadline + sha re-pin, service unknown envelope keys + frame cap, typed HolderTimeout, holderclient non-ASCII mac + deadline test | `security-expert` re-check → — | — |
Security gates: D9a Pass 8 and D8 Pass 9 both approve with suggestions (see reviews). P8-3 (holder_uid ≠ broker uid) relayed to D9b `a03ff974185d6bebb`; P9-1 (service same-uid residual stated in plan §1) pending plan edit.
D9a/D8 analyst gates: Pass 14 (D9a) and Pass 15 (D8) approve with suggestions; K1 (service.main trust plumbing untested, major), J1–J3, K2–K4 routed to D8b `a3a998384646de428` (relayed); plan absorption (10 protocol choices, P9-1, scope grammar, close_reason follow-up) routed to architect `a5d2d6277492067da`; AGENTS.md/README refresh owed at D8 close.
| D9b gates | `analyst` `aa88268ad17048330` (Pass 16), `security-expert` `a7b80d0ec280cf976` (Pass 10) | — | in-flight | `docs/reviews/…` | — → — | — |

Blocked: D11/D12 (ccg switchover) need "ccg P1 phase exit closed" — `docs/plans/connected-code-graph-coordination.md` last records P1-38 phase exit STARTED (2026-11-23), not closed; closing it is ccg's coordination, not this one. D13 (suites) needs D12; D14 (tier 2) needs the human's sudo run; D15 (docs) partly doable now (AGENTS.md/README refresh owed).
| D9b gates | analyst Pass 16 (approve w/ suggestions; L1 major: production `trusted_uids` default unpinned) + security Pass 10 (approve w/ suggestions) | — | gated | reviews | → D9c |
| D9c | `coder` | `a766eb7f93cf1ab35` | delivered (ccg 1514 + llm-gateway 1113 + units 94 re-run green; 11 mutants killed per report; committed 317e9578, golden b52059c6 (3 units, teco applied), docs refreshed) | L1–L3 tests, ccg golden units list regen (red `test_units` caused by new `llm-gateway/` component), AGENTS.md/README refresh | `analyst` spot-check → — | — |
| spec5 match D9b | `architect` | `a6c231652670eed13` | delivered (4,288 words; committed) | `docs/plans/connected-code-graph-egress-spec5.md` | — → — | — |

2026-11-24 stakeholder decisions (AskUserQuestion): (1) ccg P1-38 exit: PAUSE — D11/D12/D13 held until the ccg coordination records P1 closed; (2) tier-2: prepare runbook now; (3) Anthropic `tools`: DEFER to the falkor-chat follow-on (backlog item owed: `llm-gateway/docs/BACKLOG.md`, with `Depends on`/`Priority`/`Touches`).
| D14 prep | `coder` | `a6ee3133137f013e4` | accepted | `llm-gateway/tests/deploy/*`, `llm-gateway/deploy/RUNBOOK.md` (human runs with sudo) | `analyst` + `security-expert` → approve with suggestions x3 rounds (analyst Pass 17-19, security Pass 11-13); committed 580b12db, 979c944d, 2d7b92bf, 3c632f01 | 365k+ tokens |
Owed backlog items (to `llm-gateway/docs/BACKLOG.md`): Anthropic tool-call translation (follow-on); `keyholder log` reading rotated `calls.log.1`; `calls.log`/ledger pruning of `reservations`; embeddings (FR-14 follow-on); per-copy sha pin for client consumers; two client-contract tests (call_id/raw dropped, GatewayError→refusal class); `BrokerClient.Response.close_reason` exposure (D10 uses private `_call`); `HolderClient` ctor uid check (optional defence in depth); one unexplained 15-min test hang in a combined cpg run (D13 to watch).

## Paused 2026-11-24 — restart note (fresh session)

No delegate is in flight. Everything built is committed (D14 prep final at `3c632f01`). Cold-start order:
1. Human runs `llm-gateway/deploy/RUNBOOK.md` (sudo); first commit/stash the in-flight `cpg/` work, take `SHA` ≥ `3c632f01`; docker-group decision open (blocks AC-5/D14 and step 13c). Paste-back (step 16) routes to `analyst` + `security-expert`.
2. D11/D12/D13 stay held until `docs/plans/connected-code-graph-coordination.md` records the ccg P1-38 phase exit closed; never close it from here.
3. Owed: D15 docs (plan §7); `llm-gateway/docs/BACKLOG.md` (items listed above, `wi mint-legacy`); feature close (live spec, `-impl` check, `REVIEW.md`, `HISTORY.md`); milestone freeze list.
