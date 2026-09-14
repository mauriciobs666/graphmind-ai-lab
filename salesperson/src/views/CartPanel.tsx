// FR-8/AC-6 — cart lines + running total + empty state (docs/plans/
// salesperson-ui.md §5.1's S14 row, §2.4's parity table). Reads
// `GET /shop/api/state`'s `cart` block via `useShopState()` (S12a) — no
// separate cart endpoint exists. `views/{Cart,Order,Profile,Catalog}*` is
// this step's own owned subtree; this file only replaces S12b's seed
// placeholder content, per §5.0.
import { useTranslation } from 'react-i18next';
import { useShopState } from '../api/hooks';
import { formatCurrency } from '../i18n/format';
import { useLocale } from '../i18n/useLocale';

export function CartPanel() {
  const { t } = useTranslation();
  const state = useShopState();
  const { locale } = useLocale();

  // A 401 here is `useShopState()`'s own dispatch navigating the participant
  // back to join (C3) — render nothing for that one instant rather than an
  // alarming "couldn't load" flash right before the sheet unmounts.
  if (state.action?.kind === 'clearParticipant') return null;

  if (!state.data) {
    if (state.isError) {
      return (
        <p role="alert" className="text-sm text-red-600 dark:text-red-400">
          {t('cart.loadError')}
        </p>
      );
    }
    return (
      <p className="text-sm text-slate-500 dark:text-slate-400">{t('cart.loading')}</p>
    );
  }

  const { cart } = state.data;

  if (cart.items.length === 0) {
    return <p className="text-sm text-slate-500 dark:text-slate-400">{t('cart.empty')}</p>;
  }

  return (
    <div className="flex flex-col gap-3">
      {state.isError && (
        <p className="text-xs text-amber-600 dark:text-amber-400">
          {t('cart.staleNotice')}
        </p>
      )}
      <ul className="flex flex-col divide-y divide-slate-100 dark:divide-slate-800">
        {cart.items.map((item) => (
          <li
            key={item.productId}
            className="flex items-center justify-between gap-3 py-2.5 text-sm"
          >
            <span className="text-slate-700 dark:text-slate-200">
              <span className="font-medium text-slate-900 dark:text-slate-50">
                {item.quantity}×
              </span>{' '}
              {item.name}
            </span>
            <span className="shrink-0 tabular-nums text-slate-600 dark:text-slate-300">
              {formatCurrency(item.unitPrice * item.quantity, locale)}
            </span>
          </li>
        ))}
      </ul>
      <div className="flex items-center justify-between border-t border-slate-200 pt-3 text-sm font-semibold text-slate-900 dark:border-slate-700 dark:text-slate-50">
        <span>{t('cart.total')}</span>
        <span className="tabular-nums">{formatCurrency(cart.total, locale)}</span>
      </div>
    </div>
  );
}
