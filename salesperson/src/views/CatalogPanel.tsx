// FR-11/AC-11 — image-or-text-only product grid (docs/plans/
// salesperson-ui.md §5.1's S14 row, §4.7). Reads `GET /shop/api/catalog`
// via `useCatalog()` (S12a): each `CatalogEntry.imageUrl` is
// `/shop/products/<productId>.<ext>` or `null` — the manifest builder
// (`falkor-chat` §4.7) has already resolved presence/absence server-side, so
// this file's only job is to render exactly one of the two card shapes per
// product, never an `<img>` with an empty/placeholder `src` for the `null`
// case. `views/{Cart,Order,Profile,Catalog}*` is this step's own owned
// subtree; this file only replaces S12b's seed placeholder content, per
// §5.0.
import { useTranslation } from 'react-i18next';
import { useCatalog } from '../api/hooks';
import { formatCurrency } from '../i18n/format';
import { useLocale } from '../i18n/useLocale';

export function CatalogPanel() {
  const { t } = useTranslation();
  const catalog = useCatalog();
  const { locale } = useLocale();

  if (catalog.action?.kind === 'clearParticipant') return null;

  if (!catalog.data) {
    if (catalog.isError) {
      return (
        <p role="alert" className="text-sm text-red-600 dark:text-red-400">
          {t('catalog.loadError')}
        </p>
      );
    }
    return (
      <p className="text-sm text-slate-500 dark:text-slate-400">{t('catalog.loading')}</p>
    );
  }

  if (catalog.data.length === 0) {
    return (
      <p className="text-sm text-slate-500 dark:text-slate-400">{t('catalog.empty')}</p>
    );
  }

  return (
    <ul className="grid grid-cols-2 gap-3">
      {catalog.data.map((product) => (
        <li
          key={product.productId}
          className="flex flex-col overflow-hidden rounded-lg border border-slate-200 bg-white shadow-sm dark:border-slate-800 dark:bg-slate-900"
        >
          {product.imageUrl ? (
            <img
              src={product.imageUrl}
              alt={product.name}
              loading="lazy"
              className="h-24 w-full object-cover"
            />
          ) : (
            <div
              aria-hidden="true"
              className="flex h-24 w-full items-center justify-center bg-slate-100 px-2 text-center text-[11px] font-medium uppercase tracking-wide text-slate-400 dark:bg-slate-800 dark:text-slate-500"
            >
              {t('catalog.noPhoto')}
            </div>
          )}
          <div className="flex flex-1 flex-col gap-0.5 p-2.5">
            <span className="text-[11px] font-medium uppercase tracking-wide text-slate-400 dark:text-slate-500">
              {product.category}
            </span>
            <span className="text-sm font-semibold text-slate-900 dark:text-slate-50">
              {product.name}
            </span>
            <span className="mt-auto pt-1 text-sm tabular-nums text-slate-600 dark:text-slate-300">
              {formatCurrency(product.price, locale)}
            </span>
          </div>
        </li>
      ))}
    </ul>
  );
}
