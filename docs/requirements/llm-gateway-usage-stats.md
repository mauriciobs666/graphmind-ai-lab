# LLM gateway usage and statistics — Feature Requirements

> **Status:** Interviewing · **Owner:** `tico` · **Tracks:** — · **Extends:** `docs/requirements/llm-gateway.md` · **Last updated:** 2026-11-23 · **Item:** `GAL-F0047`

Split from `docs/requirements/llm-gateway.md` (FR-21 to FR-25, AC-15, AC-16) at the analyst's finding M6. The call record these reports read is defined there (FR-5; one record per attempt); this document adds no record fields. Phase: after the gateway's first delivery; the exact phase is an open question.

## Intent
The gateway owner can understand cost and performance of every model call across components, and compare models and providers, without writing queries by hand.

## User stories
- As the repo owner, I want to see tokens, money, speed and reliability by provider, model, component and scope over a period, so that I can judge what a model costs and how well it works.
- As the repo owner, I want to compare two models or providers side by side, so that I can choose between a local and a hosted model on evidence.
- As the repo owner, I want to be told when spend, error rate or latency crosses a threshold, so that I notice a problem without watching.

## Functional requirements
- **FR-1** — Reports answer, filtered by provider, model, component, scope, call site and period: token volume (input, output, total); cost in money; latency and throughput (time to answer, tokens per second, error and timeout rates, each stated per call or per attempt); fallback and reliability (how often and why a fallback fired, failure rates of hosted versus local calls).
- **FR-2** — Reports are available as command-line output and through an interface a program can query, returning the same figures; its format is a design matter.
- **FR-3** — Two models or providers can be compared side by side on cost and performance. The comparison states its basis: the same recorded inputs replayed (only possible for a scope that opted in to text recording, gateway FR-5), or each model's own traffic.
- **FR-4** — A threshold on spend, error rate or latency raises an alert. An alert only notifies; it never stops or reroutes a call. (A spend threshold is a cross-run aggregate, so this notify-only rule is what separates it from the excluded cross-run budget.)
- **FR-5** — Cost in money uses a price table the owner maintains in the gateway's configuration, with separate input and output prices per model. A local model's cost is zero while its tokens and speed are still reported. A hosted call with no price entry is reported as unpriced, not zero.

## Out of scope
- A cross-run spending budget that stops calls (excluded in the gateway document).
- A dashboard or UI beyond command-line output and the query interface.
- Changes to the call record's fields (owned by the gateway document).

## Acceptance criteria
- **AC-1** — Given a hosted model with a price entry and calls in a period, the cost report equals input tokens times input price plus output tokens times output price; a local model shows zero cost with tokens and speed; an unpriced hosted model is flagged unpriced. (FR-1, FR-5)
- **AC-2** — Each report filter (provider, model, component, scope, call site, period) narrows the figures to the matching records only. (FR-1)
- **AC-3** — The same report is obtainable from the command line and from the query interface with identical figures. (FR-2)
- **AC-4** — A comparison names its basis and returns cost and performance for both models; a replay comparison on a scope without text recording is refused with that reason. (FR-3)
- **AC-5** — A threshold crossing produces one alert at the stated delivery point, and the next call still proceeds unchanged. (FR-4)

## Open questions
- Where do call records live, for how long, and who can read them?
- How are alerts delivered, and who or what evaluates a threshold when the gateway is used inside a short-lived process with no daemon?
- Which comparison basis is wanted: replayed inputs, or each model's own traffic?
- Where is an alert threshold configured, and who owns it?
- Does a report price a call at the price in effect when it was made, or at the current price?
- In which phase does this ship relative to the gateway's first delivery and follow-on?

## Decision log
- 2026-11-22 — Which measures, how delivered? → Token volume, cost in money, latency/throughput, fallback/reliability; command-line reports, a queryable interface, model comparison and notify-only threshold alerts (FR-1 to FR-4). Taken in the gateway document's session.
- 2026-11-22 — Prices and local cost? → Owner-maintained price table; local model cost is zero (FR-5).
- 2026-11-23 — Split the statistics tooling out of the gateway document? → Yes, its own document and phase (this document).
