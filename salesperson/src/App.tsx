// The SPA's root component (S5 scaffolded → S12a owns thereafter; S12b earns
// one narrow, additive edit right here in v1.36 — docs/plans/salesperson-ui.md
// §4.11/§5.0's shared-file map). Composes the app-wide providers plus
// `I18nProvider`'s (`./i18n/Provider.tsx`, S12c's owned subtree) mount slot.
//
// `LayoutShell` (`./layout/Shell.tsx`, S12b's owned subtree) is deliberately
// NOT rendered here — as of v1.36 it is the router's own pathless layout
// route (`./routes.tsx`), not a wrapper around `<RouterProvider>`. That is
// what lets `LayoutShell` (and everything it renders — the header, the four
// bottom sheets) reach `useSession()`/`useQuery()`/`useMutation()` (via the
// two providers below) *and* `useNavigate()`/`useLocation()` (only reachable
// from inside the router's own matched-route tree) — a bare provider-order
// swap keeping `LayoutShell` as a `children`-wrapper cannot do the latter,
// because `RouterProvider`'s own props type takes no `children` slot. Full
// trace in §4.11.
import { QueryClient, QueryClientProvider } from '@tanstack/react-query';
import { RouterProvider } from 'react-router-dom';
import { I18nProvider } from './i18n/Provider';
import { router } from './routes';
import { SessionProvider } from './session/SessionContext';

const queryClient = new QueryClient();

function App() {
  return (
    <I18nProvider>
      <QueryClientProvider client={queryClient}>
        <SessionProvider>
          <RouterProvider router={router} />
        </SessionProvider>
      </QueryClientProvider>
    </I18nProvider>
  );
}

export default App;
