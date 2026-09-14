// S12b's no-op seed placeholder for S14's profile panel
// (docs/plans/salesperson-ui.md §5.0's v1.34 seed row) — wired into
// `components/sheets/ProfileSheet.tsx`'s body, above this step's own
// `ResetControl` (AC-5's participant reset lives in the sheet's *chrome*,
// `components/sheets/`, never in this file). `src/views/{Cart,Order,
// Profile,Catalog}*` is S14's owned subtree; S14 replaces this file's
// *content* only (FR-10: name/delivery-address card with em-dash
// placeholders) and never touches `layout/**`/`components/sheets/**` to
// reach it. Deliberately no hooks, no data: mirrors `layout/Shell.tsx`'s and
// `i18n/Provider.tsx`'s own pass-through style.
export function ProfilePanel() {
  return (
    <p className="text-sm text-slate-500 dark:text-slate-400">Your profile will appear here.</p>
  );
}
