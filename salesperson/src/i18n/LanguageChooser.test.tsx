// Covers two of S12c's done-conditions at once (docs/plans/salesperson-ui.md
// §5.1's S12c row): "chosen locale reaches the join request" (the `onChange`
// contract below) and "UI chrome switches" (the second test — the
// chooser's own label/hint actually re-render, not merely that a bundle
// loaded).
import { render, screen } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import { afterEach, describe, expect, it, vi } from 'vitest';
import i18n from './config';
import { LanguageChooser } from './LanguageChooser';

// `i18next`'s instance is a module-level singleton (`./config`); reset it
// after every test so one test's `changeLanguage` can't leak into the next.
afterEach(async () => {
  await i18n.changeLanguage('en');
});

describe('LanguageChooser', () => {
  it('renders the seeded locale set and reports the picked value to the caller', async () => {
    const user = userEvent.setup();
    const onChange = vi.fn();
    render(<LanguageChooser value="en" onChange={onChange} locales={['en', 'pt-BR', 'es']} />);

    const select = screen.getByRole('combobox');
    expect(select).toHaveValue('en');
    // Native per-locale names, not translated by the current chrome locale.
    expect(screen.getByRole('option', { name: 'English' })).toBeInTheDocument();
    expect(screen.getByRole('option', { name: 'Português (Brasil)' })).toBeInTheDocument();
    expect(screen.getByRole('option', { name: 'Español' })).toBeInTheDocument();

    await user.selectOptions(select, 'pt-BR');

    // The caller (the join screen) decides what actually reaches
    // `POST /shop/api/session`'s `language` field — this component only
    // reports the pick.
    expect(onChange).toHaveBeenCalledTimes(1);
    expect(onChange).toHaveBeenCalledWith('pt-BR');
  });

  it('offers a deployment-widened locale absent from NATIVE_NAMES by its raw code', () => {
    render(<LanguageChooser value="en" onChange={vi.fn()} locales={['en', 'de']} />);
    expect(screen.getByRole('option', { name: 'de' })).toBeInTheDocument();
  });

  it('switches its own visible copy immediately when a language is picked — not merely once the bundle loads', async () => {
    const user = userEvent.setup();
    render(<LanguageChooser value="en" onChange={vi.fn()} locales={['en', 'pt-BR', 'es']} />);

    expect(screen.getByText('Language')).toBeInTheDocument();
    expect(
      screen.getByText("Choose the language you'd like to chat in."),
    ).toBeInTheDocument();

    await user.selectOptions(screen.getByRole('combobox'), 'pt-BR');

    expect(await screen.findByText('Idioma')).toBeInTheDocument();
    expect(
      await screen.findByText('Escolha o idioma em que você quer conversar.'),
    ).toBeInTheDocument();

    await user.selectOptions(screen.getByRole('combobox'), 'es');

    expect(await screen.findByText('Idioma')).toBeInTheDocument();
    expect(
      await screen.findByText('Elige el idioma en el que quieres chatear.'),
    ).toBeInTheDocument();
  });
});
