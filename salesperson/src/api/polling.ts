// C8 — "both hooks read one shared exported constant, never two literals."
// `useShopState()` and the messages-poll hook both import this single value,
// so retuning R10's budget is a one-line, deliberate change rather than a
// per-view drift that silently reopens it (docs/plans/salesperson-ui.md
// §5.3 C8).
export const POLL_INTERVAL_MS = 2_000;
