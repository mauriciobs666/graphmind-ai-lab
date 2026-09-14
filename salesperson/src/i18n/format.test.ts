import { describe, expect, it } from 'vitest';
import { formatCurrency, formatDate } from './format';

describe('formatCurrency', () => {
  it('uses locale-appropriate decimal separators', () => {
    // en-US: period decimal. pt-BR/es use a comma instead (and, per this
    // runtime's CLDR data, group only from 10,000 up, so a 4-digit amount
    // is the sharper probe for the separator swap than grouping would be).
    expect(formatCurrency(1234.5, 'en')).toMatch(/1,?234\.50/);
    expect(formatCurrency(1234.5, 'pt-BR')).toMatch(/1\.?234,50/);
    expect(formatCurrency(1234.5, 'es')).toMatch(/1\.?234,50/);
  });

  it('honours an explicit currency', () => {
    expect(formatCurrency(10, 'en', 'EUR')).toContain('€');
  });
});

describe('formatDate', () => {
  it('renders the same instant differently across locales', () => {
    const instant = new Date(Date.UTC(2026, 0, 15, 10, 30));
    const en = formatDate(instant, 'en', { dateStyle: 'medium' });
    const ptBR = formatDate(instant, 'pt-BR', { dateStyle: 'medium' });
    const es = formatDate(instant, 'es', { dateStyle: 'medium' });

    expect(en).not.toBe(ptBR);
    expect(en).not.toBe(es);
  });

  it('accepts an epoch-ms number, matching this API surface\'s wire format', () => {
    const ms = Date.UTC(2026, 0, 15, 10, 30);
    expect(formatDate(ms, 'en')).toBe(formatDate(new Date(ms), 'en'));
  });
});
