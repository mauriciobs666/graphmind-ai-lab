# Backlog — decision-models

> **Status:** active · **Owner:** `teco` · **Tracks:** —

> Forward-looking only: what is proposed but unbuilt. A delivered item is recorded in
> [`HISTORY.md`](./HISTORY.md) and does not stay here (root `AGENTS.md`).

## Open items

### Hand-off prompt parity for join-in and message type

The guard's hand-off prompt is checked against the served one (`tests/test_handoff_parity.py`,
`_JUDGE_SYSTEM_PROMPT` in `falkor-chat/server/falkorchat/app.py`). The join-in judge and the message
type classifier have no runtime prompt yet, so `fit/join-in-judge/handoff.md` and
`fit/message-type-classifier/handoff.md` are unchecked against anything served
(`docs/plans/typed-decision-models4.md` §3.11, R-24).

- **Owner:** whichever unit adds the join-in or message-type runtime prompt to falkor-chat.
- **Done when:** a guarded test, same shape as the guard's (`ast` parse of the served prompt
  constant; equals the pack prompt after `strip()`; skips only when the sibling checkout is absent;
  fails when the file exists and the name is not a module-level assignment), covers each prompt
  that unit adds.
