// The Chat view (docs/plans/salesperson-ui.md §5.1's S13 row): transcript,
// composer, the thinking/queued indicator, and the dead-turn notice. Replaces
// `routes.tsx`'s inline `ChatScreen` placeholder — mounted by that file's
// `ParticipantRoute` once a participant session exists and no language step
// is pending, itself inside `LayoutShell`'s pathless layout route
// (`layout/Shell.tsx`), so `useSession()`/TanStack Query's hooks are all
// genuinely reachable here (§4.11). This file owns no hook of its own beyond
// composing `api/hooks.ts`'s three — every §5.3 rule it renders (C6a, C4,
// C9, C11, C14, C6b) is already classified by `usePostMessage()`'s `action`/
// `reconciliation`; `./composerNotice.ts` only picks the copy.
//
// **Welcome turn — deliberately not implemented here.** §5.2/§5.3 make the
// join response's `welcome` line explicitly *not* session state ("the
// greeting belongs to the join moment and is not something the client
// re-reads"), and `routes.tsx`'s `JoinScreen`/`useJoin()` (both outside this
// step's edit rights — the S13 row's own words: "edits no shared entry file
// except one narrow, additive swap in `routes.tsx`") discard `data.welcome`
// after `setParticipant()` rather than handing it anywhere this view can
// reach. There is currently no path for that line to arrive here without an
// edit to `api/hooks.ts` or `session/SessionContext.tsx` — both S12a's,
// undelivered to this step. Flagged in this unit's report rather than
// guessed at.
import { type FormEvent, useState } from 'react';
import { usePostMessage, useShopState, useMessages } from '../api/hooks';
import { Composer } from '../components/message/Composer';
import { composerNoticeFor } from '../components/message/composerNotice';
import { DeadTurnNotice } from '../components/message/DeadTurnNotice';
import { Transcript } from '../components/message/Transcript';
import { TurnIndicator } from '../components/message/TurnIndicator';
import { withOptimisticRow } from '../components/message/optimistic';

export function ChatView() {
  const shopState = useShopState();
  const messages = useMessages();
  const postMessage = usePostMessage();
  const [text, setText] = useState('');

  const turnState = shopState.data?.turn.state ?? 'idle';
  // C6a — the composer gates on `turn.state` alone; the dead-turn notice
  // below (`lastTurn`) is read independently and never disables it.
  const composerDisabled = turnState !== 'idle' || postMessage.isPending;

  function handleSubmit(event: FormEvent) {
    event.preventDefault();
    const trimmed = text.trim();
    if (!trimmed || composerDisabled) return;
    postMessage.mutate(trimmed, {
      // C6a — cleared only on success; a 409/503/504 leaves the composer
      // text exactly as typed, so sending again is a single tap.
      onSuccess: () => setText(''),
    });
  }

  // Optimistic send — `postMessage`'s own successful response already *is*
  // the real, server-assigned row (`api/endpoints.ts`'s `postMessage`
  // returns `Promise<MessageRow>`); `withOptimisticRow` stops adding it the
  // moment the polled transcript catches up (matched by `msgId`).
  const rows = withOptimisticRow(
    messages.data ?? [],
    postMessage.isSuccess ? postMessage.data : null,
  );
  const pendingText = postMessage.isPending ? (postMessage.variables ?? null) : null;
  const notice = composerNoticeFor(postMessage.action, postMessage.reconciliation);

  return (
    // `layout/Shell.tsx`'s `<Outlet/>` wrapper is `flex-1` on a `display:
    // block` element (not itself a flex/grid container), so a plain
    // `h-full` here does not resolve to a definite size in every browser —
    // verified live (a Playwright probe against the running dev server
    // measured this root collapsing to its content height, ~177px, instead
    // of the wrapper's actual 787px). `Header`'s height is the one fixed,
    // known quantity in that chain (`layout/Header.tsx`'s `h-14` = 3.5rem,
    // plus its own `env(safe-area-inset-top)`, which `dvh` already accounts
    // for at the viewport level) — sized against the viewport directly
    // rather than against the ancestor's collapsed box.
    <div className="flex h-[calc(100dvh-3.5rem)] min-h-0 flex-col">
      <Transcript rows={rows} pendingText={pendingText} />
      <TurnIndicator turn={shopState.data?.turn} />
      <DeadTurnNotice visible={shopState.deadTurnNotice} />
      <Composer
        value={text}
        onChange={setText}
        onSubmit={handleSubmit}
        disabled={composerDisabled}
        notice={notice}
      />
    </div>
  );
}
