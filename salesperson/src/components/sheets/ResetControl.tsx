// AC-5's participant half — the participant's own reset control, living in
// the profile sheet's *chrome* (this subtree, `components/sheets/`), not
// S14's profile card (`views/Profile*`). Behind a confirm step.
//
// v1.36/§4.11 — now a genuine router/session/query descendant (`LayoutShell`
// is the router's own pathless layout route, `layout/Shell.tsx`), so this
// calls `api/hooks.ts`'s `useResetMine()` directly instead of hand-rolling a
// `resetMine`/`resolveErrorAction` dispatch. That hook already implements
// the full §5.3 C1-C14 dispatch (`resolveErrorAction` via
// `useErrorEffects`), C7's navigate-to-language-step, C3's clear-and-redirect
// on a 401, and C4/C9's `queryClient.invalidateQueries` on a `state`
// reread — this component only switches on `.action.kind` for branch-
// specific copy; it performs no dispatch of its own.
import { useEffect, useState } from 'react';
import type { ErrorAction } from '../../api/dispatch';
import { useResetMine } from '../../api/hooks';
import { useSession } from '../../session/SessionContext';

type Phase = 'idle' | 'confirming';

function errorMessageFor(action: ErrorAction | null): string | null {
  if (!action) return null;
  switch (action.kind) {
    case 'reread':
      // C4/F8 — a 504 leaves the reset's outcome ambiguous (it may have
      // committed server-side); say so rather than implying it failed.
      return "We couldn't confirm whether that worked — check your cart/order before trying again.";
    case 'nothingChangedRetry':
      // C9 — 503: the reset itself never ran.
      return 'Something went wrong and nothing was reset. Please try again.';
    case 'unscopedAlarm':
      // C6b — a distinct alarm, not busy and not success. No implied retry.
      return 'Your session is no longer scoped to this store. Reload the page to continue.';
    case 'unhandled':
      // C13 — loud, never a silent generic fallback.
      return `Unexpected response from the server (status ${action.status}). Please try again.`;
    case 'clearParticipant':
      // C3 — the credential is already cleared and the client is navigating
      // away; nothing to show inline.
      return null;
    default:
      return null;
  }
}

export function ResetControl({ onReset }: { onReset?: () => void }) {
  const [phase, setPhase] = useState<Phase>('idle');
  const { participant } = useSession();
  const resetMine = useResetMine();

  useEffect(() => {
    // C3 — a 401 during this call clears the participant credential and
    // navigates to join (useResetMine's own useErrorEffects dispatch); close
    // the sheet the same way the success path does, since staying open on a
    // now-cleared session makes no sense.
    if (resetMine.action?.kind === 'clearParticipant') {
      onReset?.();
    }
  }, [resetMine.action, onReset]);

  if (!participant) {
    return (
      <p className="text-sm text-slate-500 dark:text-slate-400">
        Join the store to manage your session.
      </p>
    );
  }

  function handleConfirm() {
    resetMine.mutate(undefined, {
      onSuccess: () => onReset?.(),
    });
  }

  // A retry in flight clears the previous attempt's error immediately,
  // rather than leaving it rendered until the new response lands —
  // `useResetMine()` itself only clears `action` in its own `onSuccess`, not
  // when a new `mutate()` call starts (analyst Pass 3 minor, S12b v1.36).
  const error = resetMine.isPending ? null : errorMessageFor(resetMine.action);

  return (
    <div className="flex flex-col gap-2">
      <h3 className="text-sm font-semibold text-slate-900 dark:text-slate-50">
        Session controls
      </h3>

      {phase === 'idle' && (
        <button
          type="button"
          onClick={() => setPhase('confirming')}
          className="self-start rounded-md border border-red-200 px-3 py-1.5 text-sm font-medium text-red-600 transition hover:bg-red-50 dark:border-red-900 dark:text-red-400 dark:hover:bg-red-950/40"
        >
          Reset my session
        </button>
      )}

      {phase === 'confirming' && (
        <div className="rounded-md border border-red-200 bg-red-50 p-3 text-sm text-red-900 dark:border-red-900 dark:bg-red-950/30 dark:text-red-100">
          <p>This clears your cart, order and chat. This cannot be undone.</p>
          {error && (
            <p role="alert" className="mt-2 text-xs font-medium text-red-700 dark:text-red-300">
              {error}
            </p>
          )}
          <div className="mt-2 flex gap-2">
            <button
              type="button"
              onClick={() => setPhase('idle')}
              disabled={resetMine.isPending}
              className="rounded-md border border-slate-300 px-3 py-1.5 text-xs font-medium text-slate-700 disabled:cursor-not-allowed disabled:opacity-50 dark:border-slate-600 dark:text-slate-200"
            >
              Cancel
            </button>
            <button
              type="button"
              onClick={handleConfirm}
              disabled={resetMine.isPending}
              className="rounded-md bg-red-600 px-3 py-1.5 text-xs font-medium text-white disabled:cursor-not-allowed disabled:opacity-50"
            >
              {resetMine.isPending ? 'Resetting…' : 'Yes, reset'}
            </button>
          </div>
        </div>
      )}
    </div>
  );
}
