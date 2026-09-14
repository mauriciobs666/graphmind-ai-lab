// §5.2 *The dead-turn signal* / §5.3 C6a — `turn.lastTurn === 'failed'`
// (`api/hooks.ts`'s `useShopState().deadTurnNotice`) is the visible half of a
// turn that died on the worker: the participant's message is in the
// transcript, no reply ever came, and nothing else distinguishes this from a
// completed turn. It is **not** a fourth `state` and **must not gate the
// composer** — `ChatView.tsx` drives `disabled` from `turn.state` alone, this
// component never touches it. `role="status"` (not `"alert"`): recoverable
// by the participant's own next action, not a blocking failure.
export function DeadTurnNotice({ visible }: { visible: boolean }) {
  if (!visible) return null;
  return (
    <p
      role="status"
      className="mx-3 mt-2 rounded-md bg-amber-100 px-3 py-2 text-sm text-amber-900 dark:bg-amber-950/40 dark:text-amber-200"
    >
      The reply never arrived — send again.
    </p>
  );
}
