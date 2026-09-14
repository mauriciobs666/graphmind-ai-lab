// The join-screen language chooser (docs/plans/salesperson-ui.md §5.1's
// S12c row). Lives in S12c's own subtree rather than inline in
// `../routes.tsx` on purpose: `routes.tsx` isn't S12c's file, and factoring
// the field out here keeps that file's edit to a single-line swap (import
// this component, drop it in where the inline `<label>`/`<select>` used to
// be) instead of a hand-rolled translation of surrounding JoinScreen markup.
import type { ChangeEvent } from 'react';
import { useTranslation } from 'react-i18next';
import { useLocale } from './useLocale';

// Each language's own name for itself — shown the same way regardless of
// which locale is currently active, so a Portuguese speaker can find
// "Português (Brasil)" in the list even while the chrome around it still
// reads in English. Deliberately not part of the translation bundles: this
// is identity data about a locale, not chrome text `t()` looks up for the
// *current* locale.
const NATIVE_NAMES: Record<string, string> = {
  en: 'English',
  'pt-BR': 'Português (Brasil)',
  es: 'Español',
};

export interface LanguageChooserProps {
  /** The value the join form will post to `POST /shop/api/session`'s
   * `language` field (§5.2) — the caller (`../routes.tsx`) stays the source
   * of truth for what is actually sent; this component only presents the
   * choice and live-previews the chrome in it. */
  value: string;
  onChange: (locale: string) => void;
  /** Seeded from `GET /shop/api/health`'s `locales` field, never hard-coded
   * here — the caller already does this (§5.2; nothing consumed this field
   * before S12c). A deployment-widened locale with no entry in
   * `NATIVE_NAMES` still renders, using its own raw code as the label. */
  locales: string[];
  id?: string;
}

export function LanguageChooser({ value, onChange, locales, id }: LanguageChooserProps) {
  const { t } = useTranslation();
  const { setLocale } = useLocale();

  function handleChange(event: ChangeEvent<HTMLSelectElement>) {
    const next = event.target.value;
    onChange(next);
    // Live preview: the chrome switches as soon as a language is picked,
    // before the join request is ever sent (AC-9/§4.5's per-participant
    // `language` is a separate, server-side value set at join).
    setLocale(next);
  }

  return (
    <label
      htmlFor={id}
      className="flex flex-col gap-1 text-sm text-slate-700 dark:text-slate-200"
    >
      {t('join.languageLabel')}
      <select
        id={id}
        className="rounded-md border border-slate-300 px-3 py-2 text-base outline-none focus:border-slate-500 dark:border-slate-600 dark:bg-slate-900"
        value={value}
        onChange={handleChange}
      >
        {locales.map((locale) => (
          <option key={locale} value={locale}>
            {NATIVE_NAMES[locale] ?? locale}
          </option>
        ))}
      </select>
      <span className="text-xs font-normal text-slate-500 dark:text-slate-400">
        {t('join.languageHint')}
      </span>
    </label>
  );
}
