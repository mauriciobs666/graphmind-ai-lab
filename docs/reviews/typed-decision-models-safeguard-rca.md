# Typed decision models: pilot bat_0001 safeguard halt (RCA)

> **Status:** active · **Owner:** `analyst` · **Tracks:** — · **Last updated:** 2026-11-11

CPG: not applicable — RCA of a live campaign run's artifacts (ledger, transcripts, plan text); no code-level change under review.

Scope: `~/dm-campaigns/pilot` (guard-judge point, pilot phase), batch `bat_0001`, session `bf1f741d-9145-41e1-af23-a55ed4f1700b`. Read only: ledger, run result JSON, req/ans parts, hook log, the session's own transcripts under `~/.claude/projects/-home-mauricio-dm-campaigns-pilot-exchange/`, plan `docs/plans/typed-decision-models6.md` (F7 at line 34, §3.4 row, §3.5), `decision-models/points/guard-judge/`. No driver, resolve, or model call was run; nothing modified. No generated text is reproduced here.

## 1. Symptom & impact

`bat_0001` (25 guard-judge seeds, attempt 1) ended `outcome: halted`, `safeguardStop`, `voidClass: environmental`; details: a refusal stop / `refusedMessageId` entry, and `message.model` `claude-opus-4-8` is not the pin `claude-opus-5-5`. Batches 2-4 were queued but never issued (ledger has one `issued` event). Impact: the batch is void and uncounted, as designed; the campaign waits on a stakeholder `resolve`. Cost of the void run: $0.52 total, $0.49 of it on Opus 4.8.

## 2. Reproduction & evidence

Not reproduced (a live call was out of bounds). Traced from the transcripts, which carry timestamps:

| Time (UTC) | Entry (subagent transcript) |
|---|---|
| 23:35:51 | spawn message to `dm-generator` (Opus 5.5): "Read {request} offset 1 limit 1 ..." |
| 23:35:54 | Opus 5.5 assistant, 137 output tokens: one `Read` of request line 1 (the header: schema, role, point, `answerPath`, `bodySha256`, `pages`) |
| 23:35:54 | tool result: the header line only |
| 23:35:56 | `user` entry, `isMeta: true`, with `refusedMessageId`; text begins "Your response above was stopped by a safety classifier" |
| 23:35:59-23:36:00 | **Opus 4.8** thinks, then `Read`s offset 2, limit 26 (instructions and seeds) |
| 23:37:48-23:38:45 | Opus 4.8: ~14.2k tokens (about 8k thinking), then one `Write` of the answer |
| 23:38:46 | `done` |

Result JSON `modelUsage`: Sonnet 5.5 (driver) 228 output tokens; Opus 5.5 137 output tokens (only the first Read); Opus 4.8 14,386 output tokens. `safety_stops: 2`. The driver was never flagged. The session transcript holds one `refusedMessageId` entry; I found no source for the second counted stop in the files read.

## 3. Causal chain

1. The run's refused message was the Opus 5.5 reply **after the header-only tool result** (23:35:54 to 23:35:56). Before it, Opus 5.5 had seen only the agent's system prompt, the spawn text and line 1 of the request.
2. Opus 5.5 never read the instructions (line 2) or any seed. Those were first read by Opus 4.8 at 23:36:00. So the flag cannot have been caused by the guard-judge request's instructions, its seed conditions or domains, or by anything the generator wrote. The refused message's own content is not persisted, so what it contained is unobserved.
3. The harness continued on Opus 4.8 without blocking: consistent with the user's "Switch automatically" setting that plan §3.5 anticipated (the "message.model ≠ pin" row). Both audit rows fired (the details list shows both), and the batch halted before counting.
4. The text the model saw contains no adversarial or injection content in any case: the instruction text (5,265 chars) has zero hits for injection, adversarial, attack, exploit, hack, password or secret; `generator.md`, `rubric.md` and the agent definition have zero hits for those words. The seed conditions are 8 ordinary checkpoint conditions (budget stated, option chosen, date given, and so on) across 11 plain business domains (bicycle shop, library help, vet clinic, bakery, and similar). The Opus 4.8 answer I skimmed for shape (bike-shop, IT-help-desk items) is benign chat.
5. Precedent for a flag on non-adversarial work: the probe report (`docs/test-reports/typed-decision-models-harness-probe-report.md` lines 33-51) records an Opus 5.5 flag on a driver turn that was an input-validation error on an `Agent` call, and the dialog text itself says the safeguards "can sometimes flag non-cybersecurity work" and are being tuned. Plan §3.5 says a subagent-side flag under `-p` was **not probed**; this is the first observation of one.

Hypotheses ruled out: (a) the guard-judge generator is asked to write prompt-injection or adversarial examples (the template and seeds ask for neutral yes/no checkpoint conversations; it is also temporally impossible, since the flag preceded reading them); (b) generated output tripped it (nothing had been generated; the refused message came 2 seconds after the header read); (c) a harness or guard fault (hook log shows 4 allowed `PreToolUse` calls and no deny; `permission_denials: []`). Not ruled out, and unobservable from these files: some property of the dm-generator system prompt plus spawn text (an agent prompt that says write only to a path, a hash-bearing file header) that the classifier reacts to, or plain classifier noise. I did not find evidence for either.

## 4. Root cause

**Inferred, moderate confidence for "not content-driven":** the Opus 5.5 safeguard flagged a turn that had seen only the generator's system prompt, the spawn message and the request header. The trigger is therefore unrelated to the guard-judge content class. The deeper cause (why the classifier fired) lies outside the codebase's control and is not observable here.
- **Determinism across seeds: unknown, and the data cannot say.** One batch, one flag, before any seed was visible. This run gives no evidence of a seed-specific content class. The same prefix (system prompt, spawn text, header) is identical for every generator batch, so a prefix-driven flag, if it is one, would recur on batches 2-4; a classifier-noise flag would not. Only a further live call separates these, and none was made.
- **Contributing factor:** the Opus 5.5 pin for the generator, set (per the F7 precedent) knowing the safeguards can flag non-cybersecurity work; plus "Switch automatically" in the user's `/config`, which turned a would-be dialog into a silent model switch that the audit caught by the pin row.
- **Design is behaving as specified:** the `safeguardStop` row halted before any count, and nothing from the run was counted.

## 5. Options and who decides

| Option | Evidence-based assessment | Decider |
|---|---|---|
| Prompt or rubric reframing | Not supported: nothing in the visible text is adversarial, and the flag preceded the instructions. A reframing would be a guess at the classifier. If tried, note the ledger `expectations` event records `promptSha256`/`settingsSha256`; I did not check whether a prompt edit mid-campaign is accepted or needs a restart. | Prompt edit by owner, but only after the architect confirms what an edit does to the recorded digests |
| Per-seed content class | Not supported: no seed content was visible at the flag. | n/a |
| Smaller batch size | Does not address a flag that fired on the header-only prefix; batch size changes the cost of a void, not its likelihood. | Plan/config change: architect |
| **Retry** (`resolve --decision`) | Already a stakeholder-chosen action in §3.5 (rule 2026-11-01). It would test whether the flag recurs on a second attempt, at the same ~$0.5 per void run. | Stakeholder (no automatic retry allowed by the rule) |
| **Skip the batch** (`--skip-batch`) | Existing action, counts into `dropped.safeguardSkipped`; it hides a possibly systematic flag for 4 of 4 batches. | Stakeholder |
| **Move generation to another model** (`--repin-model generator <id>`) | Existing action with plan4 §3.3 `generation-restart`; it also changes the generator model for the dataset. | Stakeholder (plan rule), architect for dataset-validity impact |
| Accept as design limit | Defensible only if retries recur; one sample is not enough. | Stakeholder, with architect recording it in plan §3.5 |

Prompt edits are the owner's call. Every other option is already a named stakeholder action in plan §3.5 or a plan change, so none of them is mine or `coder`'s to take.

## 6. Suggested next steps and prevention

1. Stakeholder picks retry first (cheapest evidence): a second flag on an identical prefix would favour a prefix-driven cause and make a repin or an architect ruling the next step; a clean pass favours noise.
2. Architect: plan §3.5 says a subagent-side flag was not probed; amend it with this observation (flag fired on the first reply after the header read; "Switch automatically" continued the subagent on Opus 4.8 under `-p`, no dialog). Check `safety_stops: 2` against the one `refusedMessageId` entry; I could not account for the second.
3. Evidence worth capturing on any retry (read-only): the flagged message's timing relative to the header read, and whether the spawn is identical across batches. Per the report convention, `teco` appends the last entries and stderr to the probe report.
4. Guardrail suggestion, for the architect to weigh: the audit's pin row could record, in the halt details, which message index carried the switch, so a halt reports the turn where the model changed.

## Open questions

- Is a second retry before a repin acceptable cost-wise (about $0.5 per void run)?
- Does the stakeholder's `/config` "Switch automatically" stay on? With it off, the pilot would block on a dialog under `-p` rather than continue, which plan §3.5 has no row for.
