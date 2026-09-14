// Locale-aware currency/date formatting helpers (docs/plans/salesperson-ui.md
// §5.1's S12c row). No caller exists yet in this tree — S14's cart/order
// panels (prices) and profile/presenter views (timestamps, `joinedAt`) are
// the first intended consumers — so these are exercised by their own unit
// tests (`./format.test.ts`) rather than by a rendered screen.
//
// Both wrap the platform `Intl` APIs directly rather than adding a
// dependency: `Intl.NumberFormat`/`Intl.DateTimeFormat` already take a BCP
// 47 locale tag, which is exactly what `useLocale()` returns.

/** Formats a numeric amount as currency for the given locale. Defaults to
 * USD because nothing in this catalog carries a currency of its own
 * (§2.1/§2.3's `Product` shape has no currency field) — a caller with a
 * different currency passes it explicitly. */
export function formatCurrency(amount: number, locale: string, currency = 'USD'): string {
  return new Intl.NumberFormat(locale, { style: 'currency', currency }).format(amount);
}

/** Formats a timestamp (epoch ms, matching this API's wire format — e.g.
 * `joinedAt`, `createdAt`, §5.2) or a `Date` for the given locale. */
export function formatDate(
  value: number | Date,
  locale: string,
  options: Intl.DateTimeFormatOptions = { dateStyle: 'medium', timeStyle: 'short' },
): string {
  const date = value instanceof Date ? value : new Date(value);
  return new Intl.DateTimeFormat(locale, options).format(date);
}
