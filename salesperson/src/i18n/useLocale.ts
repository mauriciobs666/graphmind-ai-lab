// One of S12c's two named exports (docs/plans/salesperson-ui.md §5.1's
// S12c row: "t(), useLocale()"). `t()` itself needs no wrapper here — every
// consumer reaches it the ordinary `react-i18next` way, `useTranslation()` —
// this hook is only for reading/changing which locale is active.
import { useCallback } from 'react';
import { useTranslation } from 'react-i18next';
import { DEFAULT_LOCALE } from './config';

export interface UseLocaleResult {
  /** The active `i18next` language tag, e.g. `"pt-BR"`. */
  locale: string;
  /** Switches the UI chrome's locale immediately — does not touch the
   * server-side per-participant `language` (§4.5), which is a separate
   * value carried in `run_ctx` and set once at join time. */
  setLocale: (locale: string) => void;
}

export function useLocale(): UseLocaleResult {
  const { i18n } = useTranslation();

  const setLocale = useCallback(
    (locale: string) => {
      void i18n.changeLanguage(locale);
    },
    [i18n],
  );

  return {
    locale: i18n.resolvedLanguage ?? i18n.language ?? DEFAULT_LOCALE,
    setLocale,
  };
}
