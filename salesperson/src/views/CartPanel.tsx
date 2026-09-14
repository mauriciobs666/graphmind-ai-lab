// S12b's no-op seed placeholder for S14's cart panel
// (docs/plans/salesperson-ui.md §5.0's v1.34 seed row) — wired into
// `components/sheets/CartSheet.tsx`'s body. `src/views/{Cart,Order,Profile,
// Catalog}*` is S14's owned subtree; S14 replaces this file's *content*
// only (FR-8: cart lines, running total, empty state) and never touches
// `layout/**`/`components/sheets/**` to reach it — this file is the mount
// point. Deliberately no hooks, no data: mirrors `layout/Shell.tsx`'s and
// `i18n/Provider.tsx`'s own pass-through style.
export function CartPanel() {
  return (
    <p className="text-sm text-slate-500 dark:text-slate-400">Your cart will appear here.</p>
  );
}
