// AC-5's presenter key entry (docs/plans/salesperson-ui.md §5.1's S12d
// row): submits `POST /shop/api/presenter/session`, stores the returned
// token on success (§5.3's presenter credential), and renders §5.3's C2
// (bad key), C11 (blank key) and C13 (unhandled) outcomes distinctly.
import { QueryClient, QueryClientProvider } from '@tanstack/react-query';
import { render, screen, waitFor } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import { MemoryRouter } from 'react-router-dom';
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';
import i18n from '../i18n/config';
import { SessionProvider } from '../session/SessionContext';
import { SESSION_STORAGE_KEYS } from '../session/storage';
import { PresenterKeyScreen } from './PresenterKeyScreen';

function jsonResponse(status: number, body: unknown): Response {
  return {
    ok: status >= 200 && status < 300,
    status,
    json: async () => body,
    text: async () => JSON.stringify(body),
  } as Response;
}

function renderScreen() {
  const client = new QueryClient({ defaultOptions: { queries: { retry: false } } });
  return render(
    <QueryClientProvider client={client}>
      <MemoryRouter>
        <SessionProvider>
          <PresenterKeyScreen />
        </SessionProvider>
      </MemoryRouter>
    </QueryClientProvider>,
  );
}

beforeEach(() => {
  window.localStorage.clear();
});

afterEach(async () => {
  vi.unstubAllGlobals();
  await i18n.changeLanguage('en');
});

describe('PresenterKeyScreen', () => {
  it('submits the typed key to POST /shop/api/presenter/session and stores the token on success', async () => {
    const fetchMock = vi.fn(async (input: RequestInfo | URL, init?: RequestInit) => {
      expect(String(input)).toMatch(/\/shop\/api\/presenter\/session$/);
      expect(JSON.parse(String(init?.body))).toEqual({ key: 'sesame' });
      return jsonResponse(200, { token: 'ptok-1' });
    });
    vi.stubGlobal('fetch', fetchMock);
    renderScreen();
    const user = userEvent.setup();

    await user.type(screen.getByLabelText('Presenter key'), 'sesame');
    await user.click(screen.getByRole('button', { name: 'Enter' }));

    await waitFor(() =>
      expect(window.localStorage.getItem(SESSION_STORAGE_KEYS.presenter)).toBe(
        JSON.stringify({ token: 'ptok-1' }),
      ),
    );
  });

  it('reports a rejected key without clearing anything (C2)', async () => {
    vi.stubGlobal(
      'fetch',
      vi.fn(async () => jsonResponse(403, { error: 'bad_key' })),
    );
    renderScreen();
    const user = userEvent.setup();

    await user.type(screen.getByLabelText('Presenter key'), 'wrong');
    await user.click(screen.getByRole('button', { name: 'Enter' }));

    await waitFor(() =>
      expect(screen.getByRole('alert')).toHaveTextContent('That key was not accepted.'),
    );
    expect(window.localStorage.getItem(SESSION_STORAGE_KEYS.presenter)).toBeNull();
  });

  it("clears the previous attempt's stale rejection the instant a retry starts, before the new response resolves", async () => {
    let resolveSecond: (response: Response) => void = () => {};
    let attempt = 0;
    const fetchMock = vi.fn(async () => {
      attempt += 1;
      if (attempt === 1) return jsonResponse(403, { error: 'bad_key' });
      return new Promise<Response>((resolve) => {
        resolveSecond = resolve;
      });
    });
    vi.stubGlobal('fetch', fetchMock);
    renderScreen();
    const user = userEvent.setup();

    await user.type(screen.getByLabelText('Presenter key'), 'wrong');
    await user.click(screen.getByRole('button', { name: 'Enter' }));
    await waitFor(() =>
      expect(screen.getByRole('alert')).toHaveTextContent('That key was not accepted.'),
    );

    // Retry, without the second (deliberately never-resolving) response
    // landing yet: the first attempt's rejection must be gone the instant
    // the retry starts (`isPending` flips synchronously on `mutate()`), not
    // still rendered — the button already reads "Checking…".
    await user.click(screen.getByRole('button', { name: 'Enter' }));
    expect(screen.getByRole('button', { name: 'Checking…' })).toBeInTheDocument();
    expect(screen.queryByRole('alert')).not.toBeInTheDocument();

    // Let the slow response settle so the mutation doesn't leak into the
    // next test.
    resolveSecond(jsonResponse(403, { error: 'bad_key' }));
    await waitFor(() => expect(screen.getByRole('alert')).toBeInTheDocument());
  });

  it('reports a blank key as a field error, not the generic rejection (C11)', async () => {
    vi.stubGlobal(
      'fetch',
      vi.fn(async () => jsonResponse(422, { error: 'validation_failed', field: 'key' })),
    );
    renderScreen();
    const user = userEvent.setup();

    // `required` normally stops this client-side; the assertion is on the
    // server round trip's own rendering, not on bypassing the attribute.
    const input = screen.getByLabelText('Presenter key');
    input.removeAttribute('required');
    await user.click(screen.getByRole('button', { name: 'Enter' }));

    await waitFor(() =>
      expect(screen.getByRole('alert')).toHaveTextContent('Enter the presenter key.'),
    );
    expect(screen.queryByText('That key was not accepted.')).not.toBeInTheDocument();
  });

  it('shows a visible unexpected-response message for a status resolveErrorAction leaves unhandled (C13)', async () => {
    vi.stubGlobal(
      'fetch',
      vi.fn(async () => jsonResponse(500, { error: 'boom' })),
    );
    renderScreen();
    const user = userEvent.setup();

    await user.type(screen.getByLabelText('Presenter key'), 'sesame');
    await user.click(screen.getByRole('button', { name: 'Enter' }));

    await waitFor(() =>
      expect(screen.getByRole('alert')).toHaveTextContent(/unexpected response/i),
    );
    expect(screen.getByRole('alert')).toHaveTextContent('500');
  });

  it('routes its own chrome through t() — switches to pt-BR, English literals gone', async () => {
    vi.stubGlobal(
      'fetch',
      vi.fn(async () => jsonResponse(403, { error: 'bad_key' })),
    );
    renderScreen();
    const user = userEvent.setup();

    await i18n.changeLanguage('pt-BR');

    expect(screen.getByRole('heading', { name: 'Chave do apresentador' })).toBeInTheDocument();
    expect(screen.getByRole('button', { name: 'Entrar' })).toBeInTheDocument();
    expect(screen.queryByRole('heading', { name: 'Presenter key' })).not.toBeInTheDocument();
    expect(screen.queryByRole('button', { name: 'Enter' })).not.toBeInTheDocument();

    await user.type(screen.getByLabelText('Chave do apresentador'), 'errado');
    await user.click(screen.getByRole('button', { name: 'Entrar' }));

    await waitFor(() =>
      expect(screen.getByRole('alert')).toHaveTextContent('Essa chave não foi aceita.'),
    );
    expect(screen.queryByText(/was not accepted/i)).not.toBeInTheDocument();
  });
});
