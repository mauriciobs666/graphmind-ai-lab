// The one `i18next` instance for this SPA, wired to `react-i18next`. S12c's
// owned subtree (docs/plans/salesperson-ui.md §5.0/§5.1's S12c row).
//
// The three bundles below are the shipped locale set (§4.5, OQ-2): English
// (default), Brazilian Portuguese, Spanish — the same three
// `config.STOREFRONT_LOCALES` ships server-side
// (`falkor-chat/server/falkorchat/config.py`), which is what
// `GET /shop/api/health`'s `locales` field seeds the join-screen chooser
// from (`./LanguageChooser.tsx`). A deployment that widens
// `FALKORCHAT_STOREFRONT_LOCALES` offers a locale this bundle set has no
// translation for; `i18next`'s `fallbackLng` renders English chrome for it
// rather than throwing, mirroring the server's own `welcome`-line fallback
// (§5.2).
import i18n from 'i18next';
import { initReactI18next } from 'react-i18next';
import en from '../locales/en.json';
import es from '../locales/es.json';
import ptBR from '../locales/pt-BR.json';

export const SUPPORTED_LOCALES = ['en', 'pt-BR', 'es'] as const;
export type SupportedLocale = (typeof SUPPORTED_LOCALES)[number];
export const DEFAULT_LOCALE: SupportedLocale = 'en';

export const localeResources: Record<SupportedLocale, unknown> = {
  en,
  'pt-BR': ptBR,
  es,
};

// Guarded so importing this module twice (e.g. once from a test, once from
// the app) never double-registers the instance.
if (!i18n.isInitialized) {
  void i18n.use(initReactI18next).init({
    resources: {
      en: { translation: en },
      'pt-BR': { translation: ptBR },
      es: { translation: es },
    },
    lng: DEFAULT_LOCALE,
    fallbackLng: DEFAULT_LOCALE,
    interpolation: { escapeValue: false },
    returnNull: false,
  });
}

export default i18n;
