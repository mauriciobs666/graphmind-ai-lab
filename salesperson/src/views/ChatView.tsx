// The Chat view (docs/plans/salesperson-ui.md §5.1's S13 row): transcript,
// composer, the thinking/queued indicator, the dead-turn notice, and the
// welcome turn. Replaces `routes.tsx`'s inline `ChatScreen` placeholder —
// mounted by that file's `ParticipantRoute` once a participant session
// exists and no language step is pending, itself inside `LayoutShell`'s
// pathless layout route (`layout/Shell.tsx`), so `useSession()`/TanStack
// Query's hooks are all genuinely reachable here (§4.11). This file owns no
// hook of its own beyond composing `api/hooks.ts`'s three — every §5.3 rule
// it renders (C6a, C4, C9, C11, C14, C6b) is already classified by
// `usePostMessage()`'s `action`/`reconciliation`; `./composerNotice.ts` only
// picks the copy.
//
// **Welcome turn (§4.12, v1.37).** `session/SessionContext.tsx`'s
// `welcomeMessage` carries the join response's one-shot `welcome` line, set
// by `api/hooks.ts`'s `useJoin()`. This is the only clearing site: `greeting`
// captures whatever `welcomeMessage` holds on this component's *first*
// render (a `useState` initializer, evaluated once), and the mount effect
// immediately clears the context field — so a second `ChatView` mount within
// the same join reads `welcomeMessage === null` and captures nothing,
// exactly the "shows once per join" contract. Capturing into local state
// first — rather than clearing inline during render — is what keeps this
// component's return value pure this render (clearing the context during
// render would be a same-render side effect on a value another consumer
// could read).
import { type FormEvent, useEffect, useState } from 'react';
import { useTranslation } from 'react-i18next';
import { usePostMessage, useShopState, useMessages } from '../api/hooks';
import { Composer } from '../components/message/Composer';
import { composerNoticeFor } from '../components/message/composerNotice';
import { DeadTurnNotice } from '../components/message/DeadTurnNotice';
import { Transcript } from '../components/message/Transcript';
import { TurnIndicator } from '../components/message/TurnIndicator';
import { withOptimisticRow } from '../components/message/optimistic';
import { withWelcomeRow } from '../components/message/welcome';
import { useSession } from '../session/SessionContext';

export function ChatView() {
  const { t } = useTranslation();
  const shopState = useShopState();
  const messages = useMessages();
  const postMessage = usePostMessage();
  const { welcomeMessage, setWelcomeMessage } = useSession();
  const [greeting] = useState(welcomeMessage);
  const [text, setText] = useState('');

  useEffect(() => {
    // §4.12 — the only clearing site. Unconditional: if a fresh join left a
    // line to capture, `greeting` above already has it; either way the
    // context field must not survive past this mount.
    setWelcomeMessage(null);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

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
  const rows = withWelcomeRow(
    withOptimisticRow(messages.data ?? [], postMessage.isSuccess ? postMessage.data : null),
    greeting,
  );
  const pendingText = postMessage.isPending ? (postMessage.variables ?? null) : null;
  const notice = composerNoticeFor(postMessage.action, postMessage.reconciliation, t);

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
