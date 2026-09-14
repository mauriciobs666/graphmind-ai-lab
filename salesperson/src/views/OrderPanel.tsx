// FR-9/AC-7 — order card: status chip, `cancel` as an ordinary customer
// action, `fulfill`/`deliver` inside a visually distinct "demo controls"
// affordance explicitly labelled as a warehouse simulation (docs/plans/
// salesperson-ui.md §5.1's S14 row, §4.6). Reads `GET /shop/api/state`'s
// `order` block via `useShopState()` and drives transitions through
// `useAdvanceOrder()` (both S12a). `views/{Cart,Order,Profile,Catalog}*` is
// this step's own owned subtree; this file only replaces S12b's seed
// placeholder content, per §5.0.
//
// §4.6, confirmed (OQ-4): the participant is the *only* actor who can drive
// `fulfill`/`deliver` — there is no presenter-side variant. A customer
// tapping "Fulfil" then "Deliver" on their own purchase reads as a broken
// product to a business audience, so those two transitions are boxed apart
// from `cancel` and named as a simulation, never presented as an ordinary
// storefront action.
import type { TFunction } from 'i18next';
import { useTranslation } from 'react-i18next';
import { useAdvanceOrder, useShopState } from '../api/hooks';
import type { ErrorAction } from '../api/dispatch';
import { formatCurrency } from '../i18n/format';
import { useLocale } from '../i18n/useLocale';

// `OrderBlock.lines` is deliberately typed `unknown[]` in `api/endpoints.ts`
// (S12a's own call — no client-side commitment to the line shape until a
// consumer needs it). This is that consumer: a narrow, local type guard
// against the shape `falkor-chat/server/falkorchat/repository.py`'s
// `_CURRENT_ORDER_CYPHER` actually returns (`productId`, `name`, `unitPrice`,
// `quantity`, `lineTotal`) — anything that doesn't match is dropped from the
// itemised list rather than crashing the panel; the total still renders from
// `order.total` regardless.
interface OrderLine {
  productId: string;
  name: string;
  quantity: number;
  unitPrice: number;
  lineTotal: number;
}

function isOrderLine(value: unknown): value is OrderLine {
  if (typeof value !== 'object' || value === null) return false;
  const v = value as Record<string, unknown>;
  return (
    typeof v.productId === 'string' &&
    typeof v.name === 'string' &&
    typeof v.quantity === 'number' &&
    typeof v.unitPrice === 'number' &&
    typeof v.lineTotal === 'number'
  );
}

const STATUS_STYLES: Record<string, string> = {
  placed:
    'border-blue-200 bg-blue-50 text-blue-700 dark:border-blue-900 dark:bg-blue-950/40 dark:text-blue-300',
  fulfilled:
    'border-purple-200 bg-purple-50 text-purple-700 dark:border-purple-900 dark:bg-purple-950/40 dark:text-purple-300',
  delivered:
    'border-emerald-200 bg-emerald-50 text-emerald-700 dark:border-emerald-900 dark:bg-emerald-950/40 dark:text-emerald-300',
  cancelled:
    'border-slate-200 bg-slate-100 text-slate-500 dark:border-slate-700 dark:bg-slate-800 dark:text-slate-400',
};
const DEFAULT_STATUS_STYLE =
  'border-slate-200 bg-slate-100 text-slate-600 dark:border-slate-700 dark:bg-slate-800 dark:text-slate-300';

function statusLabel(status: string): string {
  return status.length === 0 ? status : status.charAt(0).toUpperCase() + status.slice(1);
}

function errorMessageFor(action: ErrorAction | null, t: TFunction): string | null {
  if (!action) return null;
  switch (action.kind) {
    case 'staleOrderRefresh':
      // C10 — a 404/409 here is an ordinary stale-button outcome: the order
      // above has already been refreshed from the re-read.
      return t('order.error.staleOrderRefresh');
    case 'reread':
      // C4 — a 504 leaves the outcome ambiguous; the order above reflects
      // the latest state the client could confirm.
      return t('order.error.reread');
    case 'unhandled':
      return t('order.error.unhandled', { status: action.status });
    case 'clearParticipant':
      // C3 — credential already cleared, client is navigating away.
      return null;
    default:
      return null;
  }
}

function WarehouseIcon({ className }: { className?: string }) {
  return (
    <svg
      viewBox="0 0 24 24"
      fill="none"
      stroke="currentColor"
      strokeWidth={1.75}
      strokeLinecap="round"
      strokeLinejoin="round"
      aria-hidden="true"
      className={className}
    >
      <path d="M3 10.5 12 4l9 6.5" />
      <path d="M4.5 9.5V20h15V9.5" />
      <path d="M9.5 20v-5.5h5V20" />
    </svg>
  );
}

export function OrderPanel() {
  const { t } = useTranslation();
  const state = useShopState();
  const advance = useAdvanceOrder();
  const { locale } = useLocale();

  // A 401 here is `useShopState()`'s own dispatch navigating the participant
  // back to join (C3) — render nothing for that one instant rather than an
  // alarming "couldn't load" flash right before the sheet unmounts.
  if (state.action?.kind === 'clearParticipant') return null;

  if (!state.data) {
    if (state.isError) {
      return (
        <p role="alert" className="text-sm text-red-600 dark:text-red-400">
          {t('order.loadError')}
        </p>
      );
    }
    return (
      <p className="text-sm text-slate-500 dark:text-slate-400">{t('order.loading')}</p>
    );
  }

  const { order } = state.data;

  if (!order) {
    return (
      <p className="text-sm text-slate-500 dark:text-slate-400">
        {t('order.empty')}
      </p>
    );
  }

  const lines = order.lines.filter(isOrderLine);
  const advanceError = advance.isPending ? null : errorMessageFor(advance.action, t);
  const canCancel = order.status === 'placed';
  const canFulfill = order.status === 'placed';
  const canDeliver = order.status === 'fulfilled';

  return (
    <div className="flex flex-col gap-4">
      {state.isError && (
        <p className="text-xs text-amber-600 dark:text-amber-400">
          {t('order.staleNotice')}
        </p>
      )}

      <div className="flex items-center justify-between gap-2">
        <span className="text-xs font-medium text-slate-500 dark:text-slate-400">
          {t('order.idLabel', { orderId: order.orderId })}
        </span>
        <span
          className={`rounded-full border px-2.5 py-0.5 text-xs font-semibold ${
            STATUS_STYLES[order.status] ?? DEFAULT_STATUS_STYLE
          }`}
        >
          {t(`order.status.${order.status}`, statusLabel(order.status))}
        </span>
      </div>

      {lines.length > 0 && (
        <ul className="flex flex-col divide-y divide-slate-100 dark:divide-slate-800">
          {lines.map((line) => (
            <li
              key={line.productId}
              className="flex items-center justify-between gap-3 py-2 text-sm"
            >
              <span className="text-slate-700 dark:text-slate-200">
                <span className="font-medium text-slate-900 dark:text-slate-50">
                  {line.quantity}×
                </span>{' '}
                {line.name}
              </span>
              <span className="shrink-0 tabular-nums text-slate-600 dark:text-slate-300">
                {formatCurrency(line.lineTotal, locale)}
              </span>
            </li>
          ))}
        </ul>
      )}

      <div className="flex items-center justify-between border-t border-slate-200 pt-3 text-sm font-semibold text-slate-900 dark:border-slate-700 dark:text-slate-50">
        <span>{t('order.total')}</span>
        <span className="tabular-nums">{formatCurrency(order.total, locale)}</span>
      </div>

      {advanceError && (
        <p role="alert" className="text-xs font-medium text-red-600 dark:text-red-400">
          {advanceError}
        </p>
      )}

      {canCancel && (
        <button
          type="button"
          onClick={() => advance.mutate('cancel')}
          disabled={advance.isPending}
          className="self-start rounded-md border border-red-200 px-3 py-1.5 text-sm font-medium text-red-600 transition hover:bg-red-50 disabled:cursor-not-allowed disabled:opacity-50 dark:border-red-900 dark:text-red-400 dark:hover:bg-red-950/40"
        >
          {t('order.cancel')}
        </button>
      )}

      {/* §4.6's "demo controls" affordance — visually boxed apart from the
       * ordinary customer action above and explicitly labelled as a
       * warehouse simulation, never presented as something a real customer
       * would do to their own order. */}
      <div
        data-testid="order-demo-controls"
        className="rounded-lg border-2 border-dashed border-purple-300 bg-purple-50/60 p-3 dark:border-purple-800 dark:bg-purple-950/20"
      >
        <p className="flex items-center gap-1.5 text-xs font-semibold uppercase tracking-wide text-purple-700 dark:text-purple-300">
          <WarehouseIcon className="h-3.5 w-3.5 shrink-0" />
          {t('order.demoControls.label')}
        </p>
        <p className="mt-1 text-xs text-purple-700/80 dark:text-purple-300/80">
          {t('order.demoControls.description')}
        </p>
        <div className="mt-2 flex flex-wrap gap-2">
          <button
            type="button"
            onClick={() => advance.mutate('fulfill')}
            disabled={!canFulfill || advance.isPending}
            className="rounded-md border border-purple-300 bg-white px-3 py-1.5 text-xs font-medium text-purple-700 transition hover:bg-purple-100 disabled:cursor-not-allowed disabled:opacity-40 dark:border-purple-700 dark:bg-slate-900 dark:text-purple-300 dark:hover:bg-purple-950/40"
          >
            {t('order.demoControls.fulfill')}
          </button>
          <button
            type="button"
            onClick={() => advance.mutate('deliver')}
            disabled={!canDeliver || advance.isPending}
            className="rounded-md border border-purple-300 bg-white px-3 py-1.5 text-xs font-medium text-purple-700 transition hover:bg-purple-100 disabled:cursor-not-allowed disabled:opacity-40 dark:border-purple-700 dark:bg-slate-900 dark:text-purple-300 dark:hover:bg-purple-950/40"
          >
            {t('order.demoControls.deliver')}
          </button>
        </div>
      </div>
    </div>
  );
}
