// AC-5's presenter view (docs/plans/salesperson-ui.md §5.1's S12d row):
// the roster table over `GET /shop/api/presenter/participants`, one row per
// participant, carrying only `displayName`/`language` — §5.2's four-key
// projection (`participantId, displayName, language, joinedAt`) also carries
// `participantId` (used as the row key, never rendered) and `joinedAt`
// (not rendered — this view has no use for it); it never carries cart/order
// activity, because the API response itself does not (see S10) — so there
// is no activity data this component could render even by mistake, short of
// reaching past its own typed row shape. Mounted into S12b's layout shell
// via `routes.tsx`'s one narrow swap (§5.0).
import type { ReactNode } from 'react';
import { useTranslation } from 'react-i18next';
import { usePresenterParticipants } from '../api/hooks';
import { PresenterResetAllControl } from './PresenterResetAllControl';

export function PresenterRoster() {
  const { t } = useTranslation();
  const roster = usePresenterParticipants();

  // C2 — a 401/403 here already clears the presenter credential and
  // navigates back to key entry (`useErrorEffects`, run inside
  // `usePresenterParticipants` itself); render nothing for that one instant
  // rather than an alarming "couldn't load" flash right before this view
  // unmounts (mirrors `views/CartPanel.tsx`'s own C3 instant).
  if (roster.action?.kind === 'clearPresenter') return null;

  // C13 — a genuinely unmapped response (never `roster.isError`'s C9
  // staleness copy, which is only for the documented `graph_unavailable`/
  // `graph_read_timeout` shapes) is loud, naming the route and status —
  // ahead of the generic error/staleness fallback below, mirroring
  // `PresenterKeyScreen.tsx`/`PresenterResetAllControl.tsx` in this same
  // subtree.
  const unhandled = roster.action?.kind === 'unhandled' ? roster.action : null;

  let body: ReactNode;
  if (!roster.data) {
    if (unhandled) {
      body = (
        <p role="alert" className="text-sm text-red-600 dark:text-red-400">
          {t('presenter.roster.error.unhandled', { status: unhandled.status })}
        </p>
      );
    } else if (roster.isError) {
      body = (
        <p role="alert" className="text-sm text-red-600 dark:text-red-400">
          {t('presenter.roster.loadError')}
        </p>
      );
    } else {
      body = (
        <p className="text-sm text-slate-500 dark:text-slate-400">{t('presenter.roster.loading')}</p>
      );
    }
  } else if (roster.data.length === 0) {
    body = <p className="text-sm text-slate-500 dark:text-slate-400">{t('presenter.roster.empty')}</p>;
  } else {
    body = (
      <>
        {unhandled ? (
          <p role="alert" className="text-xs font-medium text-red-600 dark:text-red-400">
            {t('presenter.roster.error.unhandled', { status: unhandled.status })}
          </p>
        ) : (
          roster.isError && (
            <p className="text-xs text-amber-600 dark:text-amber-400">
              {t('presenter.roster.staleNotice')}
            </p>
          )
        )}
        <table className="w-full text-left text-sm">
          <thead>
            <tr className="border-b border-slate-200 dark:border-slate-700">
              <th scope="col" className="py-2 pr-4 font-medium text-slate-500 dark:text-slate-400">
                {t('presenter.roster.table.nameHeader')}
              </th>
              <th scope="col" className="py-2 font-medium text-slate-500 dark:text-slate-400">
                {t('presenter.roster.table.languageHeader')}
              </th>
            </tr>
          </thead>
          <tbody className="divide-y divide-slate-100 dark:divide-slate-800">
            {roster.data.map((row) => (
              <tr key={row.participantId}>
                <td className="py-2 pr-4 text-slate-900 dark:text-slate-50">{row.displayName}</td>
                <td className="py-2 text-slate-600 dark:text-slate-300">{row.language}</td>
              </tr>
            ))}
          </tbody>
        </table>
      </>
    );
  }

  return (
    <main className="mx-auto flex max-w-2xl flex-col gap-6 px-4 py-6">
      <h1 className="text-xl font-semibold text-slate-900 dark:text-slate-50">
        {t('presenter.roster.heading')}
      </h1>
      {body}
      <PresenterResetAllControl />
    </main>
  );
}
