// FR-10/AC-8 — profile card, em-dash placeholders for unset fields
// (docs/plans/salesperson-ui.md §5.1's S14 row, §2.4's parity table: the old
// app's sidebar `Nome: …` / `Endereço: …`). Reads `GET /shop/api/state`'s
// `profile` block via `useShopState()` (S12a). `views/{Cart,Order,Profile,
// Catalog}*` is this step's own owned subtree; this file only replaces
// S12b's seed placeholder content, per §5.0.
import { useTranslation } from 'react-i18next';
import { useShopState } from '../api/hooks';

const EM_DASH = '—';

export function ProfilePanel() {
  const { t } = useTranslation();
  const state = useShopState();

  // A 401 here is `useShopState()`'s own dispatch navigating the participant
  // back to join (C3) — render nothing for that one instant rather than an
  // alarming "couldn't load" flash right before the sheet unmounts.
  if (state.action?.kind === 'clearParticipant') return null;

  if (!state.data) {
    if (state.isError) {
      return (
        <p role="alert" className="text-sm text-red-600 dark:text-red-400">
          {t('profile.loadError')}
        </p>
      );
    }
    return (
      <p className="text-sm text-slate-500 dark:text-slate-400">{t('profile.loading')}</p>
    );
  }

  const { name, deliveryAddress } = state.data.profile;

  return (
    <div className="flex flex-col gap-3">
      {state.isError && (
        <p className="text-xs text-amber-600 dark:text-amber-400">
          {t('profile.staleNotice')}
        </p>
      )}
      <dl className="grid grid-cols-[auto_1fr] items-baseline gap-x-4 gap-y-2.5 text-sm">
        <dt className="font-medium text-slate-500 dark:text-slate-400">{t('profile.nameLabel')}</dt>
        <dd className="text-slate-900 dark:text-slate-50">{name || EM_DASH}</dd>
        <dt className="font-medium text-slate-500 dark:text-slate-400">{t('profile.addressLabel')}</dt>
        <dd className="text-slate-900 dark:text-slate-50">{deliveryAddress || EM_DASH}</dd>
      </dl>
    </div>
  );
}
