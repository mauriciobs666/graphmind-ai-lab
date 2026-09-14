// AC-5's presenter-side entry point (docs/plans/salesperson-ui.md §5.1's
// S12d row): the presenter key form. Moved out of `routes.tsx`'s own
// placeholder (§5.0's `routes.tsx` row licenses exactly one narrow,
// additive swap there) — this file is this step's real implementation.
//
// `usePresenterLogin()` (S12a, `api/hooks.ts`) already runs every submit
// through §5.3's C1-C14 dispatch; this component only switches on
// `.action.kind` for copy, the same division of labour
// `components/sheets/ResetControl.tsx` uses for `useResetMine()`.
import { type FormEvent, useState } from 'react';
import { useTranslation } from 'react-i18next';
import { usePresenterLogin } from '../api/hooks';

export function PresenterKeyScreen() {
  const { t } = useTranslation();
  const login = usePresenterLogin();
  const [key, setKey] = useState('');

  function handleSubmit(event: FormEvent) {
    event.preventDefault();
    login.mutate(key);
  }

  // A retry in flight clears the previous attempt's error immediately,
  // rather than leaving it rendered until the new response lands (mirrors
  // `PresenterResetAllControl.tsx`'s own analyst-flagged fix, itself
  // mirroring `ResetControl.tsx`'s).
  const action = login.isPending ? null : login.action;
  // C2's second half — the key just typed is wrong; no credential to clear,
  // report it in place.
  const keyRejected = action?.kind === 'presenterKeyRejected';
  // C11 — a blank `key` is a 422 the server answers on `field: "key"`
  // (§5.3 C11's own note: "a blank key is a human pressing Enter on an empty
  // box, the most ordinary mistake in the presenter flow"). The `required`
  // input below keeps this the backstop, not the mechanism.
  const fieldError = action?.kind === 'fieldError' && action.field === 'key' ? action : null;
  // C13 — anything else on this route (e.g. a hypothetical bare 401, which
  // `resolveErrorAction` deliberately leaves unhandled for
  // `presenterSession`) is loud rather than swallowed.
  const unhandled = action?.kind === 'unhandled' ? action : null;

  return (
    <main className="mx-auto flex min-h-svh max-w-sm flex-col justify-center gap-4 px-6">
      <h1 className="text-xl font-semibold text-slate-900 dark:text-slate-50">
        {t('presenter.keyScreen.heading')}
      </h1>
      <form onSubmit={handleSubmit} className="flex flex-col gap-3">
        <label className="flex flex-col gap-1 text-sm text-slate-700 dark:text-slate-200">
          {t('presenter.keyScreen.keyLabel')}
          <input
            type="password"
            className="rounded-md border border-slate-300 px-3 py-2 text-base outline-none focus:border-slate-500 dark:border-slate-600 dark:bg-slate-900"
            value={key}
            onChange={(event) => setKey(event.target.value)}
            required
          />
        </label>
        {keyRejected && (
          <p role="alert" className="text-xs text-red-600">
            {t('presenter.keyScreen.error.rejected')}
          </p>
        )}
        {fieldError && (
          <p role="alert" className="text-xs text-red-600">
            {t('presenter.keyScreen.error.fieldError')}
          </p>
        )}
        {unhandled && (
          <p role="alert" className="text-xs text-red-600">
            {t('presenter.keyScreen.error.unhandled', { status: unhandled.status })}
          </p>
        )}
        <button
          type="submit"
          disabled={login.isPending}
          className="rounded-md bg-slate-900 px-4 py-2 text-sm font-medium text-white disabled:opacity-50 dark:bg-slate-100 dark:text-slate-900"
        >
          {login.isPending ? t('presenter.keyScreen.submitting') : t('presenter.keyScreen.submit')}
        </button>
      </form>
    </main>
  );
}
