// Key-coverage guard for the three shipped locale bundles (§4.5/OQ-2:
// English, Brazilian Portuguese, Spanish). "All three bundles complete (no
// missing-key fallbacks)" is this row's own done-condition
// (docs/plans/salesperson-ui.md §5.1's S12c row) — this test is what makes
// that decidable rather than a reading exercise: it fails whenever any
// bundle is missing a key another one carries, so a translator adding a key
// to only one file goes red here instead of surfacing as a silent
// `i18next` fallback in the running UI.
import { describe, expect, it } from 'vitest';
import en from '../locales/en.json';
import es from '../locales/es.json';
import ptBR from '../locales/pt-BR.json';

type JsonObject = Record<string, unknown>;

function isPlainObject(value: unknown): value is JsonObject {
  return typeof value === 'object' && value !== null && !Array.isArray(value);
}

/** Flattens a nested translation bundle into dotted leaf-key paths, e.g.
 * `{ join: { languageLabel: "…" } }` -> `["join.languageLabel"]`. */
function flattenKeys(bundle: JsonObject, prefix = ''): string[] {
  return Object.entries(bundle).flatMap(([key, value]) => {
    const path = prefix ? `${prefix}.${key}` : key;
    return isPlainObject(value) ? flattenKeys(value, path) : [path];
  });
}

const BUNDLES: Record<string, JsonObject> = {
  en,
  'pt-BR': ptBR,
  es,
};

describe('locale bundle key coverage', () => {
  it('carries every key in every bundle — no missing-key fallbacks', () => {
    const keysByLocale = Object.fromEntries(
      Object.entries(BUNDLES).map(([locale, bundle]) => [locale, new Set(flattenKeys(bundle))]),
    );

    const allKeys = new Set<string>();
    for (const keys of Object.values(keysByLocale)) {
      for (const key of keys) allKeys.add(key);
    }

    const missingByLocale: Record<string, string[]> = {};
    for (const [locale, keys] of Object.entries(keysByLocale)) {
      const missing = [...allKeys].filter((key) => !keys.has(key));
      if (missing.length > 0) missingByLocale[locale] = missing;
    }

    expect(missingByLocale).toEqual({});
  });

  it('has at least one key, so the coverage check above cannot pass vacuously', () => {
    for (const bundle of Object.values(BUNDLES)) {
      expect(flattenKeys(bundle).length).toBeGreaterThan(0);
    }
  });
});
