// S12b's no-op seed placeholder for S14's order panel
// (docs/plans/salesperson-ui.md §5.0's v1.34 seed row) — wired into
// `components/sheets/OrderSheet.tsx`'s body. `src/views/{Cart,Order,Profile,
// Catalog}*` is S14's owned subtree; S14 replaces this file's *content*
// only (FR-9: status chip, cancel, the "demo controls" fulfil/deliver
// affordance, §4.6) and never touches `layout/**`/`components/sheets/**` to
// reach it — this file is the mount point. Deliberately no hooks, no data:
// mirrors `layout/Shell.tsx`'s and `i18n/Provider.tsx`'s own pass-through
// style.
export function OrderPanel() {
  return (
    <p className="text-sm text-slate-500 dark:text-slate-400">
      Your order status will appear here once you have one.
    </p>
  );
}
