// S12c's replacement for S12a's pass-through placeholder — App.tsx's
// i18n-provider slot (docs/plans/salesperson-ui.md §5.1's S12a row) is
// mounted into here, not edited: this file's *content* changes, its path
// and export name do not, so App.tsx needs no change.
import type { ReactNode } from 'react';
import { I18nextProvider } from 'react-i18next';
import i18n from './config';

export function I18nProvider({ children }: { children: ReactNode }) {
  return <I18nextProvider i18n={i18n}>{children}</I18nextProvider>;
}
