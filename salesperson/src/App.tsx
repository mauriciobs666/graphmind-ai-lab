// The SPA's root component (S5 scaffolded → S12a owns thereafter; no later
// step edits this file — docs/plans/salesperson-ui.md §5.0's shared-file
// map). Composes the app-wide providers and the two **mount slots** S12b
// and S12c consume without ever touching this file themselves:
//
//   - `LayoutShell` (`./layout/Shell.tsx`, S12b's owned subtree) — the
//     mobile chrome (sticky header, bottom sheets, safe-area insets).
//   - `I18nProvider` (`./i18n/Provider.tsx`, S12c's owned subtree) —
//     `react-i18next` wiring and locale state.
//
// Both are pass-through today (S12a's placeholders); S12b/S12c replace the
// *content* of their own file, never this one.
import { QueryClient, QueryClientProvider } from '@tanstack/react-query';
import { RouterProvider } from 'react-router-dom';
import { I18nProvider } from './i18n/Provider';
import { LayoutShell } from './layout/Shell';
import { router } from './routes';
import { SessionProvider } from './session/SessionContext';

const queryClient = new QueryClient();

function App() {
  return (
    <I18nProvider>
      <LayoutShell>
        <QueryClientProvider client={queryClient}>
          <SessionProvider>
            <RouterProvider router={router} />
          </SessionProvider>
        </QueryClientProvider>
      </LayoutShell>
    </I18nProvider>
  );
}

export default App;
