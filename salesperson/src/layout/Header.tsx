// Sticky header chrome (docs/plans/salesperson-ui.md §5.1's S12b row):
// brand mark + the four icon buttons that open each bottom sheet. Pure
// presentation over `useSheetState()` (this subtree's own context, safe —
// see `./Shell.tsx`'s top comment for why nothing here reaches for
// session/query/router state instead).
import type { ReactElement } from 'react';
import { useTranslation } from 'react-i18next';
import { CartIcon, CatalogIcon, type IconProps, OrderIcon, ProfileIcon } from './icons';
import { type SheetKey, useSheetState } from './SheetContext';

const ICON_BUTTONS: ReadonlyArray<{
  key: SheetKey;
  labelKey: string;
  Icon: (props: IconProps) => ReactElement;
}> = [
  { key: 'catalog', labelKey: 'layout.header.icon.catalog', Icon: CatalogIcon },
  { key: 'cart', labelKey: 'layout.header.icon.cart', Icon: CartIcon },
  { key: 'order', labelKey: 'layout.header.icon.order', Icon: OrderIcon },
  { key: 'profile', labelKey: 'layout.header.icon.profile', Icon: ProfileIcon },
];

export function Header() {
  const { t } = useTranslation();
  const { openSheet, open } = useSheetState();

  return (
    <header className="sticky top-0 z-30 border-b border-slate-200/80 bg-white/90 pt-[env(safe-area-inset-top)] backdrop-blur supports-[backdrop-filter]:bg-white/70 dark:border-slate-800/80 dark:bg-slate-950/90 dark:supports-[backdrop-filter]:bg-slate-950/70">
      <div className="mx-auto flex h-14 max-w-lg items-center justify-between gap-2 px-3">
        <span className="truncate text-sm font-semibold tracking-tight text-slate-900 dark:text-slate-50">
          {t('layout.header.brand')}
        </span>
        <nav aria-label={t('layout.header.nav')} className="flex items-center gap-1">
          {ICON_BUTTONS.map(({ key, labelKey, Icon }) => (
            <button
              key={key}
              type="button"
              aria-label={t(labelKey)}
              aria-pressed={openSheet === key}
              onClick={() => open(key)}
              className="flex h-11 w-11 items-center justify-center rounded-full text-slate-600 transition hover:bg-slate-100 active:scale-95 aria-pressed:bg-slate-900 aria-pressed:text-white dark:text-slate-300 dark:hover:bg-slate-800 dark:aria-pressed:bg-slate-100 dark:aria-pressed:text-slate-900"
            >
              <Icon className="h-5 w-5" />
            </button>
          ))}
        </nav>
      </div>
    </header>
  );
}
