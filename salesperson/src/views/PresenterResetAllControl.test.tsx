// AC-5's reset-everyone control (docs/plans/salesperson-ui.md §5.1's S12d
// row): behind a confirm step, and rendering `reset-all`'s
// `incomplete: true`/`unresolved` body (§5.2) as a named list rather than
// letting it read as a clean sweep — the row's own acceptance criterion,
// asserted on rendered text.
import { QueryClient, QueryClientProvider } from '@tanstack/react-query';
import { render, screen, waitFor } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import { MemoryRouter } from 'react-router-dom';
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';
import i18n from '../i18n/config';
import { SessionProvider } from '../session/SessionContext';
import { savePresenterSession } from '../session/storage';
import { PresenterResetAllControl } from './PresenterResetAllControl';

function jsonResponse(status: number, body: unknown): Response {
  return {
    ok: status >= 200 && status < 300,
    status,
    json: async () => body,
    text: async () => JSON.stringify(body),
  } as Response;
}

function renderControl() {
  const client = new QueryClient({ defaultOptions: { queries: { retry: false } } });
  return render(
    <QueryClientProvider client={client}>
      <MemoryRouter>
        <SessionProvider>
          <PresenterResetAllControl />
        </SessionProvider>
      </MemoryRouter>
    </QueryClientProvider>,
  );
}

beforeEach(() => {
  window.localStorage.clear();
  savePresenterSession({ token: 'ptok-1' });
});

afterEach(async () => {
  vi.unstubAllGlobals();
  await i18n.changeLanguage('en');
});

describe('PresenterResetAllControl', () => {
  it('requires a confirm step, and Cancel backs out without calling the API', async () => {
    const fetchMock = vi.fn(async () => jsonResponse(200, { clearedParticipants: 0 }));
    vi.stubGlobal('fetch', fetchMock);
    renderControl();
    const user = userEvent.setup();

    expect(screen.queryByText(/cannot be undone/i)).not.toBeInTheDocument();
    await user.click(screen.getByRole('button', { name: 'Reset everyone' }));
    expect(screen.getByText(/cannot be undone/i)).toBeInTheDocument();

    await user.click(screen.getByRole('button', { name: 'Cancel' }));
    expect(screen.queryByText(/cannot be undone/i)).not.toBeInTheDocument();
    expect(fetchMock).not.toHaveBeenCalled();
  });

  it('confirming calls POST /shop/api/presenter/reset-all and shows a clean-sweep result', async () => {
    let callCount = 0;
    const fetchMock = vi.fn(async (input: RequestInfo | URL, init?: RequestInit) => {
      callCount += 1;
      expect(String(input)).toMatch(/\/shop\/api\/presenter\/reset-all$/);
      expect((init?.method ?? 'GET').toUpperCase()).toBe('POST');
      return jsonResponse(200, { clearedParticipants: 3 });
    });
    vi.stubGlobal('fetch', fetchMock);
    renderControl();
    const user = userEvent.setup();

    await user.click(screen.getByRole('button', { name: 'Reset everyone' }));
    await user.click(screen.getByRole('button', { name: 'Yes, reset everyone' }));

    await waitFor(() => expect(callCount).toBe(1));
    expect(await screen.findByRole('status')).toHaveTextContent(
      'Reset complete — 3 participants cleared.',
    );
    // A clean sweep must never render the incomplete list.
    expect(screen.queryByRole('list')).not.toBeInTheDocument();
  });

  it('renders the singular success copy for a 1-participant clean sweep (i18next count:1 -> success_one)', async () => {
    vi.stubGlobal('fetch', vi.fn(async () => jsonResponse(200, { clearedParticipants: 1 })));
    renderControl();
    const user = userEvent.setup();

    await user.click(screen.getByRole('button', { name: 'Reset everyone' }));
    await user.click(screen.getByRole('button', { name: 'Yes, reset everyone' }));

    // Exact text: the singular form ("1 participant", no trailing "s") —
    // `success_one` resolving is the thing under test, not merely that some
    // count-bearing string appeared.
    expect(await screen.findByRole('status')).toHaveTextContent(
      'Reset complete — 1 participant cleared.',
    );
  });

  it('renders an incomplete reset-all response as a named list of still-live participants — must not read as clean', async () => {
    vi.stubGlobal(
      'fetch',
      vi.fn(async () =>
        jsonResponse(200, {
          clearedParticipants: 1,
          incomplete: true,
          unresolved: ['p-unresolved-1', 'p-unresolved-2'],
        }),
      ),
    );
    renderControl();
    const user = userEvent.setup();

    await user.click(screen.getByRole('button', { name: 'Reset everyone' }));
    await user.click(screen.getByRole('button', { name: 'Yes, reset everyone' }));

    const banner = await screen.findByRole('alert');
    // Exact text: the plural form ("2 participants are still live") —
    // `incomplete.heading_other` resolving, not merely a generic match.
    expect(banner).toHaveTextContent('Reset incomplete — 2 participants are still live:');
    expect(banner).toHaveTextContent('p-unresolved-1');
    expect(banner).toHaveTextContent('p-unresolved-2');
    // Must not additionally render as a clean sweep.
    expect(screen.queryByRole('status')).not.toBeInTheDocument();
    expect(screen.queryByText(/reset complete/i)).not.toBeInTheDocument();
  });

  it('renders the singular incomplete-count copy for one still-live participant (i18next count:1 -> incomplete.heading_one, en)', async () => {
    vi.stubGlobal(
      'fetch',
      vi.fn(async () =>
        jsonResponse(200, { clearedParticipants: 0, incomplete: true, unresolved: ['p-solo'] }),
      ),
    );
    renderControl();
    const user = userEvent.setup();

    await user.click(screen.getByRole('button', { name: 'Reset everyone' }));
    await user.click(screen.getByRole('button', { name: 'Yes, reset everyone' }));

    const banner = await screen.findByRole('alert');
    // Exact text: the singular form ("1 participant is", no trailing "s"
    // and singular "is" rather than "are").
    expect(banner).toHaveTextContent('Reset incomplete — 1 participant is still live:');
    expect(banner).toHaveTextContent('p-solo');
  });

  it('renders the singular incomplete-count copy for one still-live participant in es (incomplete.heading_one)', async () => {
    vi.stubGlobal(
      'fetch',
      vi.fn(async () =>
        jsonResponse(200, { clearedParticipants: 0, incomplete: true, unresolved: ['p-solo'] }),
      ),
    );
    renderControl();
    const user = userEvent.setup();

    await i18n.changeLanguage('es');

    await user.click(screen.getByRole('button', { name: 'Restablecer a todos' }));
    await user.click(screen.getByRole('button', { name: 'Sí, restablecer a todos' }));

    const banner = await screen.findByRole('alert');
    expect(banner).toHaveTextContent(
      'Restablecimiento incompleto — 1 participante sigue activo:',
    );
    expect(banner).toHaveTextContent('p-solo');
  });

  it('shows a nothing-changed message on a 503, staying on the confirm step (C9)', async () => {
    vi.stubGlobal('fetch', vi.fn(async () => jsonResponse(503, { error: 'quiesce_failed' })));
    renderControl();
    const user = userEvent.setup();

    await user.click(screen.getByRole('button', { name: 'Reset everyone' }));
    await user.click(screen.getByRole('button', { name: 'Yes, reset everyone' }));

    await waitFor(() =>
      expect(screen.getByRole('alert')).toHaveTextContent(/nothing was reset/i),
    );
    expect(screen.getByRole('button', { name: 'Cancel' })).toBeInTheDocument();
  });

  it('reports the sweep as unconfirmed rather than failed on a 504 (C4)', async () => {
    vi.stubGlobal('fetch', vi.fn(async () => jsonResponse(504, null)));
    renderControl();
    const user = userEvent.setup();

    await user.click(screen.getByRole('button', { name: 'Reset everyone' }));
    await user.click(screen.getByRole('button', { name: 'Yes, reset everyone' }));

    await waitFor(() =>
      expect(screen.getByRole('alert')).toHaveTextContent(/couldn.t confirm whether that worked/i),
    );
    expect(screen.getByRole('alert')).not.toHaveTextContent(/nothing was reset/i);
  });

  it('shows a visible unexpected-response message for a status resolveErrorAction leaves unhandled (C13)', async () => {
    vi.stubGlobal('fetch', vi.fn(async () => jsonResponse(500, { error: 'boom' })));
    renderControl();
    const user = userEvent.setup();

    await user.click(screen.getByRole('button', { name: 'Reset everyone' }));
    await user.click(screen.getByRole('button', { name: 'Yes, reset everyone' }));

    await waitFor(() =>
      expect(screen.getByRole('alert')).toHaveTextContent(/unexpected response/i),
    );
    expect(screen.getByRole('alert')).toHaveTextContent('500');
  });

  it('routes its own chrome through t() — switches to pt-BR, English literals gone, including the incomplete-list copy', async () => {
    vi.stubGlobal(
      'fetch',
      vi.fn(async () =>
        jsonResponse(200, { clearedParticipants: 1, incomplete: true, unresolved: ['p-1'] }),
      ),
    );
    renderControl();
    const user = userEvent.setup();

    await i18n.changeLanguage('pt-BR');

    expect(screen.getByText('Controles de redefinição')).toBeInTheDocument();
    await user.click(screen.getByRole('button', { name: 'Redefinir todos' }));
    expect(
      screen.getByText('Isso limpa a sessão de todos os participantes de toda a demonstração. Isso não pode ser desfeito.'),
    ).toBeInTheDocument();

    await user.click(screen.getByRole('button', { name: 'Sim, redefinir todos' }));

    const banner = await screen.findByRole('alert');
    expect(banner).toHaveTextContent('Redefinição incompleta — 1 participante ainda ativo:');
    expect(banner).toHaveTextContent('p-1');
    expect(screen.queryByText(/still live/i)).not.toBeInTheDocument();
  });
});
