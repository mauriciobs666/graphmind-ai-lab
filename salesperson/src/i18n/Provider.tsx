// S12a's placeholder for App.tsx's i18n-provider slot
// (docs/plans/salesperson-ui.md §5.1's S12a row: "App.tsx exposes an
// i18n-provider slot ... with a no-op default, so that ... S12c can mount
// into [it] without editing [App.tsx]").
//
// `src/i18n/**` is S12c's owned subtree (§5.0's shared-file map) — S12c
// replaces this file's content with the real `react-i18next` provider
// wiring. Until then it is a pass-through: no translation, no locale state.
import type { ReactNode } from 'react';

export function I18nProvider({ children }: { children: ReactNode }) {
  return <>{children}</>;
}
