// The mobile layout shell (docs/plans/salesperson-ui.md §5.1's S12b row):
// sticky header with cart/order/profile/catalog icon buttons, bottom-sheet
// overlays, safe-area insets, no horizontal scroll at 360px (verified by
// tests/e2e/mobile-shell.spec.ts at 360x740/390x844).
//
// v1.36/§4.11 — `LayoutShell` is the router's own top-level **pathless
// layout route** (`./routes.tsx`), not a wrapper `App.tsx` renders around
// `<RouterProvider>`. `App.tsx`'s composition is now:
//
//   <I18nProvider><QueryClientProvider><SessionProvider>
//     <RouterProvider router={router} />
//   </SessionProvider></QueryClientProvider></I18nProvider>
//
// and `routes.tsx` nests the three real routes as this component's
// `children` via `{ element: <LayoutShell />, children: [...] }`. Because
// the router itself renders `LayoutShell` as part of its matched-route tree
// (react-router's standard "layout + sidebar" pattern), everything this
// component renders — not just `<Outlet/>`'s content — sits genuinely
// inside `SessionContext`/TanStack Query/the router's own context. That is
// what lets `components/sheets/ResetControl.tsx` call `api/hooks.ts`'s
// `useResetMine()` directly instead of the `sessionBridge`/`injectBridge`
// splice this file used to carry (deleted in the same revision — see
// `falkor-chat/docs/reviews/salesperson-ui-s12b.md`'s Major and this plan's
// §4.11 for the full trace and the rejected bare-provider-swap alternative).
import { Outlet } from 'react-router-dom';
import { CartSheet } from '../components/sheets/CartSheet';
import { CatalogSheet } from '../components/sheets/CatalogSheet';
import { OrderSheet } from '../components/sheets/OrderSheet';
import { ProfileSheet } from '../components/sheets/ProfileSheet';
import { Header } from './Header';
import { SheetStateProvider } from './SheetContext';

export function LayoutShell() {
  return (
    <SheetStateProvider>
      <div className="flex min-h-svh flex-col bg-slate-50 text-slate-900 dark:bg-slate-950 dark:text-slate-50">
        <Header />
        <div className="flex-1">
          <Outlet />
        </div>
        <CatalogSheet />
        <CartSheet />
        <OrderSheet />
        <ProfileSheet />
      </div>
    </SheetStateProvider>
  );
}
