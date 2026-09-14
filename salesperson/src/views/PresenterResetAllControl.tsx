// AC-5's reset-everyone control (docs/plans/salesperson-ui.md §5.1's S12d
// row) — behind a confirm step, same shape as the participant's own
// `components/sheets/ResetControl.tsx`, and rendering `reset-all`'s
// `incomplete: true`/`unresolved` body (§5.2) as a named list of
// participants whose state is still live rather than letting it read as a
// clean sweep.
//
// `usePresenterResetAll()` (S12a, `api/hooks.ts`) already runs the mutation
// through §5.3's C1-C14 dispatch — including C4's re-read of
// `GET /shop/api/presenter/participants` on a 504 and C12's `retry: 0` /
// `networkMode: 'always'` pins — this component only switches on
// `.action.kind` for copy and renders the mutation's own `data` for the
// success/incomplete cases.
import { useState } from 'react';
import type { TFunction } from 'i18next';
import { useTranslation } from 'react-i18next';
import type { ErrorAction } from '../api/dispatch';
import { usePresenterResetAll } from '../api/hooks';

type Phase = 'idle' | 'confirming';

function errorMessageFor(action: ErrorAction | null, t: TFunction): string | null {
  if (!action) return null;
  switch (action.kind) {
    case 'reread':
      // C4 — a 504 leaves the sweep's outcome ambiguous; the re-read this
      // hook already fires is the report, so point at the roster above.
      return t('presenter.resetAll.error.reread');
    case 'nothingChangedRetry':
      // C9 — 503: the sweep never ran. User-initiated, so the retry control
      // is this same confirm button — no separate control needed.
      return t('presenter.resetAll.error.nothingChanged');
    case 'unhandled':
      // C13 — loud, never a silent generic fallback.
      return t('presenter.resetAll.error.unhandled', { status: action.status });
    case 'clearPresenter':
      // C2 — the credential is already cleared and the client is navigating
      // back to key entry; nothing to show inline.
      return null;
    default:
      return null;
  }
}

export function PresenterResetAllControl() {
  const { t } = useTranslation();
  const [phase, setPhase] = useState<Phase>('idle');
  const resetAll = usePresenterResetAll();

  function handleConfirm() {
    resetAll.mutate(undefined, { onSuccess: () => setPhase('idle') });
  }

  // A retry in flight clears the previous attempt's error/result immediately
  // rather than leaving it rendered until the new response lands (mirrors
  // `ResetControl.tsx`'s own analyst-flagged fix).
  const error = resetAll.isPending ? null : errorMessageFor(resetAll.action, t);
  const result = resetAll.isPending ? null : resetAll.data;
  const unresolved = result?.unresolved ?? [];
  const showIncomplete = result?.incomplete === true;

  return (
    <div className="flex flex-col gap-2 border-t border-slate-200 pt-4 dark:border-slate-700">
      <h2 className="text-sm font-semibold text-slate-900 dark:text-slate-50">
        {t('presenter.resetAll.heading')}
      </h2>

      {phase === 'idle' && (
        <button
          type="button"
          onClick={() => setPhase('confirming')}
          className="self-start rounded-md border border-red-200 px-3 py-1.5 text-sm font-medium text-red-600 transition hover:bg-red-50 dark:border-red-900 dark:text-red-400 dark:hover:bg-red-950/40"
        >
          {t('presenter.resetAll.cta')}
        </button>
      )}

      {phase === 'confirming' && (
        <div className="rounded-md border border-red-200 bg-red-50 p-3 text-sm text-red-900 dark:border-red-900 dark:bg-red-950/30 dark:text-red-100">
          <p>{t('presenter.resetAll.confirmBody')}</p>
          {error && (
            <p role="alert" className="mt-2 text-xs font-medium text-red-700 dark:text-red-300">
              {error}
            </p>
          )}
          <div className="mt-2 flex gap-2">
            <button
              type="button"
              onClick={() => setPhase('idle')}
              disabled={resetAll.isPending}
              className="rounded-md border border-slate-300 px-3 py-1.5 text-xs font-medium text-slate-700 disabled:cursor-not-allowed disabled:opacity-50 dark:border-slate-600 dark:text-slate-200"
            >
              {t('presenter.resetAll.cancel')}
            </button>
            <button
              type="button"
              onClick={handleConfirm}
              disabled={resetAll.isPending}
              className="rounded-md bg-red-600 px-3 py-1.5 text-xs font-medium text-white disabled:cursor-not-allowed disabled:opacity-50"
            >
              {resetAll.isPending ? t('presenter.resetAll.confirming') : t('presenter.resetAll.confirm')}
            </button>
          </div>
        </div>
      )}

      {!resetAll.isPending && result && !error && (
        showIncomplete ? (
          <div
            role="alert"
            className="rounded-md border border-amber-300 bg-amber-50 p-3 text-sm text-amber-900 dark:border-amber-800 dark:bg-amber-950/30 dark:text-amber-100"
          >
            <p className="font-medium">
              {t('presenter.resetAll.incomplete.heading', { count: unresolved.length })}
            </p>
            <ul className="mt-1 list-disc pl-5">
              {unresolved.map((id) => (
                <li key={id}>{id}</li>
              ))}
            </ul>
          </div>
        ) : (
          <p
            role="status"
            className="rounded-md border border-emerald-200 bg-emerald-50 p-3 text-sm text-emerald-900 dark:border-emerald-900 dark:bg-emerald-950/30 dark:text-emerald-100"
          >
            {t('presenter.resetAll.success', { count: result.clearedParticipants })}
          </p>
        )
      )}
    </div>
  );
}
