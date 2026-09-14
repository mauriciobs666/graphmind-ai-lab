// Integration coverage — a real `QueryClientProvider > MemoryRouter >
// SessionProvider` stack (mirroring `api/hooks.test.tsx`'s own `Providers`),
// with a mocked `fetch`, driving `ChatView` the way the browser actually
// would: type, submit, poll. `dispatch.test.ts`/`composerNotice.test.ts`/
// `TurnIndicator.test.tsx` already prove each rule in isolation; these tests
// prove the wiring — the S13 row's own done-conditions
// (docs/plans/salesperson-ui.md §5.1).
import { type ReactElement, useState } from 'react';
import { render, screen, waitFor } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import { QueryClient, QueryClientProvider } from '@tanstack/react-query';
import { MemoryRouter } from 'react-router-dom';
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';
import type { MessageRow, StateResponse } from '../api/endpoints';
import { useJoin } from '../api/hooks';
import { SessionProvider, useSession } from '../session/SessionContext';
import { saveParticipantSession } from '../session/storage';
import { ChatView } from './ChatView';

function jsonResponse(status: number, body: unknown) {
  return {
    ok: status >= 200 && status < 300,
    status,
    json: async () => body,
    text: async () => JSON.stringify(body),
  } as Response;
}

function makeClient() {
  return new QueryClient({
    defaultOptions: { queries: { retry: false }, mutations: { retry: false } },
  });
}

function baseState(overrides: Partial<StateResponse['turn']> = {}): StateResponse {
  return {
    profile: { name: 'Ada', deliveryAddress: null },
    cart: { items: [], total: 0 },
    order: null,
    turn: { state: 'idle', queuePosition: 0, lastTurn: null, ...overrides },
  };
}

/** A manually-released gate — used below to hold the C4 reread's
 * `getMessages`/`getState` round trip open long enough to observe the render
 * that happens *between* `action` being set (synchronous) and
 * `reconciliation` being set (after that `await`) — see
 * `falkor-chat/docs/reviews/salesperson-ui-s13.md`'s Major finding. */
function createDeferred<T>() {
  let resolve!: (value: T) => void;
  const promise = new Promise<T>((res) => {
    resolve = res;
  });
  return { promise, resolve };
}

function renderChatView(client: QueryClient) {
  return render(
    <QueryClientProvider client={client}>
      <MemoryRouter initialEntries={['/shop/join']}>
        <SessionProvider>
          <ChatView />
        </SessionProvider>
      </MemoryRouter>
    </QueryClientProvider>,
  );
}

// §4.12's welcome-turn tests need a real `useJoin()` call in the loop — the
// context field is only ever set from inside its `onSuccess`, never
// injectable from outside `SessionProvider` the way a persisted
// `ParticipantSession` is via `saveParticipantSession`. This harness
// reproduces just enough of `routes.tsx`'s `ParticipantRoute` (join screen
// vs. chat, switched purely on `participant`) to exercise that real path,
// without touching `routes.tsx` itself — out of this fix's scope.
function JoinThenChat() {
  const { participant } = useSession();
  const join = useJoin();
  if (!participant) {
    return (
      <button type="button" onClick={() => join.mutate({ displayName: 'Ada', language: 'en' })}>
        Join
      </button>
    );
  }
  return <ChatView />;
}

// Toggles `ChatView` out of and back into the tree while `SessionProvider`
// (and therefore the session context, including `welcomeMessage`) stays
// mounted throughout — proving "does not reappear on a second `ChatView`
// mount within the same join" against a genuine unmount/remount, not a
// same-instance re-render.
function JoinThenToggleChat() {
  const { participant } = useSession();
  const join = useJoin();
  const [showChat, setShowChat] = useState(true);
  if (!participant) {
    return (
      <button type="button" onClick={() => join.mutate({ displayName: 'Ada', language: 'en' })}>
        Join
      </button>
    );
  }
  return (
    <>
      <button type="button" onClick={() => setShowChat((s) => !s)}>
        Toggle chat
      </button>
      {showChat ? <ChatView /> : <div>elsewhere</div>}
    </>
  );
}

function renderWithHarness(client: QueryClient, Harness: () => ReactElement) {
  return render(
    <QueryClientProvider client={client}>
      <MemoryRouter initialEntries={['/shop/join']}>
        <SessionProvider>
          <Harness />
        </SessionProvider>
      </MemoryRouter>
    </QueryClientProvider>,
  );
}

function mockJoinAndPollFetch(welcome: string) {
  return vi.fn(async (input: RequestInfo | URL, init?: RequestInit) => {
    const url = String(input);
    const method = (init?.method ?? 'GET').toUpperCase();
    if (method === 'POST' && url.includes('/shop/api/session')) {
      return jsonResponse(200, {
        participantId: 'p-1', token: 'tok', displayName: 'Ada', language: 'en', welcome,
      });
    }
    if (url.includes('/shop/api/messages')) return jsonResponse(200, []);
    if (url.includes('/shop/api/state')) return jsonResponse(200, baseState());
    throw new Error(`unexpected fetch: ${method} ${url}`);
  });
}

beforeEach(() => {
  window.localStorage.clear();
});

afterEach(() => {
  vi.unstubAllGlobals();
});

describe('ChatView — scripted 5-turn conversation', () => {
  it('renders every sent line and its reply, in order', async () => {
    saveParticipantSession({ participantId: 'p-1', token: 'tok', displayName: 'Ada', language: 'en' });
    let serverMessages: MessageRow[] = [];
    let seq = 0;

    const fetchMock = vi.fn(async (input: RequestInfo | URL, init?: RequestInit) => {
      const url = String(input);
      const method = (init?.method ?? 'GET').toUpperCase();
      if (method === 'POST' && url.includes('/shop/api/messages')) {
        seq += 1;
        const body = JSON.parse(init!.body as string) as { text: string };
        const userRow: MessageRow = {
          msgId: `u${seq}`, threadId: 't', authorId: 'p-1', text: body.text,
          role: 'user', createdAt: seq * 10, mentions: [],
        };
        const agentRow: MessageRow = {
          msgId: `a${seq}`, threadId: 't', authorId: 'agent', text: `Reply #${seq}`,
          role: 'assistant', createdAt: seq * 10 + 1, mentions: [],
        };
        serverMessages = [...serverMessages, userRow, agentRow];
        return jsonResponse(200, userRow);
      }
      if (url.includes('/shop/api/messages')) return jsonResponse(200, serverMessages);
      if (url.includes('/shop/api/state')) return jsonResponse(200, baseState());
      throw new Error(`unexpected fetch: ${method} ${url}`);
    });
    vi.stubGlobal('fetch', fetchMock);

    const user = userEvent.setup();
    renderChatView(makeClient());

    for (let i = 1; i <= 5; i++) {
      const input = screen.getByLabelText('Message');
      await user.type(input, `hello #${i}`);
      await user.click(screen.getByRole('button', { name: 'Send' }));
      expect(await screen.findByText(`Reply #${i}`)).toBeInTheDocument();
      expect(screen.getByText(`hello #${i}`)).toBeInTheDocument();
      await waitFor(() => expect(input).toHaveValue(''));
    }

    const rows = screen.getAllByTestId('message-row');
    expect(rows).toHaveLength(10);
    expect(rows.map((row) => row.textContent)).toEqual([
      expect.stringContaining('hello #1'), expect.stringContaining('Reply #1'),
      expect.stringContaining('hello #2'), expect.stringContaining('Reply #2'),
      expect.stringContaining('hello #3'), expect.stringContaining('Reply #3'),
      expect.stringContaining('hello #4'), expect.stringContaining('Reply #4'),
      expect.stringContaining('hello #5'), expect.stringContaining('Reply #5'),
    ]);
  });
});

describe('ChatView — agent-emitted markup', () => {
  it('renders as literal text, never as parsed HTML', async () => {
    saveParticipantSession({ participantId: 'p-1', token: 'tok', displayName: 'Ada', language: 'en' });
    const markup = '<img src=x onerror="window.__pwned = true">';
    const rows: MessageRow[] = [
      { msgId: 'a1', threadId: 't', authorId: 'agent', text: markup, role: 'assistant', createdAt: 1, mentions: [] },
    ];
    const fetchMock = vi.fn(async (input: RequestInfo | URL) => {
      const url = String(input);
      if (url.includes('/shop/api/messages')) return jsonResponse(200, rows);
      if (url.includes('/shop/api/state')) return jsonResponse(200, baseState());
      throw new Error(`unexpected fetch: ${url}`);
    });
    vi.stubGlobal('fetch', fetchMock);

    renderChatView(makeClient());

    expect(await screen.findByText(markup)).toBeInTheDocument();
    expect(document.querySelector('img')).toBeNull();
    expect((globalThis as { __pwned?: boolean }).__pwned).toBeUndefined();
  });
});

describe('ChatView — queued turn position', () => {
  it('queuePosition: 0 renders as first in line', async () => {
    saveParticipantSession({ participantId: 'p-1', token: 'tok', displayName: 'Ada', language: 'en' });
    const fetchMock = vi.fn(async (input: RequestInfo | URL) => {
      const url = String(input);
      if (url.includes('/shop/api/messages')) return jsonResponse(200, []);
      if (url.includes('/shop/api/state')) {
        return jsonResponse(200, baseState({ state: 'queued', queuePosition: 0 }));
      }
      throw new Error(`unexpected fetch: ${url}`);
    });
    vi.stubGlobal('fetch', fetchMock);

    renderChatView(makeClient());

    expect(await screen.findByText(/first in line/i)).toBeInTheDocument();
    expect(screen.getByRole('status')).toHaveTextContent(/first in line/i);
  });
});

describe('ChatView — the dead-turn notice (§5.2/§5.3 C6a)', () => {
  it('renders with send still enabled, and clears once the next post is accepted', async () => {
    saveParticipantSession({ participantId: 'p-1', token: 'tok', displayName: 'Ada', language: 'en' });
    let lastTurn: 'failed' | null = 'failed';

    const fetchMock = vi.fn(async (input: RequestInfo | URL, init?: RequestInit) => {
      const url = String(input);
      const method = (init?.method ?? 'GET').toUpperCase();
      if (method === 'POST' && url.includes('/shop/api/messages')) {
        lastTurn = null; // §5.2 — cleared when the next turn is accepted
        const body = JSON.parse(init!.body as string) as { text: string };
        return jsonResponse(200, {
          msgId: 'm1', threadId: 't', authorId: 'p-1', text: body.text,
          role: 'user', createdAt: 1, mentions: [],
        });
      }
      if (url.includes('/shop/api/messages')) return jsonResponse(200, []);
      if (url.includes('/shop/api/state')) return jsonResponse(200, baseState({ lastTurn }));
      throw new Error(`unexpected fetch: ${method} ${url}`);
    });
    vi.stubGlobal('fetch', fetchMock);

    const user = userEvent.setup();
    renderChatView(makeClient());

    expect(await screen.findByText('The reply never arrived — send again.')).toBeInTheDocument();
    // C6a — the notice must not gate the composer (the input field is never
    // disabled by `lastTurn`; the send *button* is separately, ordinarily,
    // disabled while the field is empty — that gate is content, not C6a's).
    expect(screen.getByLabelText('Message')).not.toBeDisabled();

    await user.type(screen.getByLabelText('Message'), 'send again');
    expect(screen.getByRole('button', { name: 'Send' })).not.toBeDisabled();
    await user.click(screen.getByRole('button', { name: 'Send' }));

    await waitFor(() =>
      expect(screen.queryByText('The reply never arrived — send again.')).not.toBeInTheDocument(),
    );
  });
});

// `falkor-chat/docs/reviews/salesperson-ui-s13.md`'s Major finding, reproduced
// as a permanent test: `usePostMessage()`'s `onError` sets `action` (sync)
// and `reconciliation` (after the reread's own network round trip) as two
// separate state updates, so React renders at least once with
// `action.kind === 'reread'` and `reconciliation` still `null` — the
// "…checking…" copy is that render, not a hypothetical default.
describe('ChatView — C4 reread: the in-between "checking…" state before reconciliation resolves', () => {
  it('renders "checking…" while the post-504 reread is gated, and a settled notice once it resolves', async () => {
    saveParticipantSession({ participantId: 'p-1', token: 'tok', displayName: 'Ada', language: 'en' });
    const gate = createDeferred<void>();

    const fetchMock = vi.fn(async (input: RequestInfo | URL, init?: RequestInit) => {
      const url = String(input);
      const method = (init?.method ?? 'GET').toUpperCase();
      if (method === 'POST' && url.includes('/shop/api/messages')) {
        return jsonResponse(504, null); // C4 — triggers the reread branch
      }
      if (url.includes('/shop/api/messages') || url.includes('/shop/api/state')) {
        // Holds every GET this component issues (the initial polls and the
        // reread's own re-read alike) until the test releases the gate —
        // mirrors the review's reproduction.
        await gate.promise;
        return url.includes('/shop/api/state')
          ? jsonResponse(200, baseState())
          : jsonResponse(200, []);
      }
      throw new Error(`unexpected fetch: ${method} ${url}`);
    });
    vi.stubGlobal('fetch', fetchMock);

    const user = userEvent.setup();
    renderChatView(makeClient());

    await user.type(screen.getByLabelText('Message'), 'will it arrive?');
    await user.click(screen.getByRole('button', { name: 'Send' }));

    // `action` is already `reread`; `reconciliation` is still `null` because
    // the gate is held — this is the window the review's Major targets.
    expect(await screen.findByText(/checking/i)).toBeInTheDocument();

    gate.resolve();

    // The message never actually reached the server (the POST 504'd before
    // any write), so once the gated re-read resolves, reconciliation reads
    // `nothingCommitted` and the "checking…" copy is replaced, not merely
    // joined by another notice.
    await waitFor(() => expect(screen.queryByText(/checking/i)).not.toBeInTheDocument());
    expect(
      await screen.findByText('Your message was not sent. Please try again.'),
    ).toBeInTheDocument();
  });
});

// §4.12 (v1.37) — the welcome turn's own done-conditions, added to the S13
// row's set.
describe('ChatView — the welcome turn (§4.12, v1.37)', () => {
  it("a fresh join's welcome line renders once in the chat view before anything is sent", async () => {
    vi.stubGlobal('fetch', mockJoinAndPollFetch('Welcome to the store, Ada.'));
    const user = userEvent.setup();
    renderWithHarness(makeClient(), JoinThenChat);

    await user.click(screen.getByRole('button', { name: 'Join' }));

    expect(await screen.findByText('Welcome to the store, Ada.')).toBeInTheDocument();
    // Nothing sent yet — the composer is still empty.
    expect(screen.getByLabelText('Message')).toHaveValue('');
  });

  it('a persisted session that did not just join (no `useJoin()` call) never shows a welcome line — `welcomeMessage` starts `null`', async () => {
    // Mirrors a page reload: `ParticipantSession` survives in storage
    // (`saveParticipantSession`), but `SessionProvider` itself has just
    // mounted fresh and no `useJoin()` call has run in this session.
    saveParticipantSession({ participantId: 'p-1', token: 'tok', displayName: 'Ada', language: 'en' });
    const fetchMock = vi.fn(async (input: RequestInfo | URL) => {
      const url = String(input);
      if (url.includes('/shop/api/messages')) return jsonResponse(200, []);
      if (url.includes('/shop/api/state')) return jsonResponse(200, baseState());
      throw new Error(`unexpected fetch: ${url}`);
    });
    vi.stubGlobal('fetch', fetchMock);

    renderChatView(makeClient());

    // The transcript renders its ordinary empty state, not a welcome turn —
    // proving `welcomeMessage` came up `null` rather than stale/leaked.
    expect(await screen.findByText(/no messages yet/i)).toBeInTheDocument();
    expect(screen.queryByText(/welcome to the store/i)).not.toBeInTheDocument();
  });

  it('does not reappear on a second ChatView mount within the same join', async () => {
    vi.stubGlobal('fetch', mockJoinAndPollFetch('Welcome to the store, Ada.'));
    const user = userEvent.setup();
    renderWithHarness(makeClient(), JoinThenToggleChat);

    await user.click(screen.getByRole('button', { name: 'Join' }));
    expect(await screen.findByText('Welcome to the store, Ada.')).toBeInTheDocument();

    // Unmount ChatView, then remount it — `SessionProvider` (and therefore
    // the already-cleared `welcomeMessage`) never itself remounts.
    await user.click(screen.getByRole('button', { name: 'Toggle chat' }));
    expect(screen.getByText('elsewhere')).toBeInTheDocument();
    await user.click(screen.getByRole('button', { name: 'Toggle chat' }));

    // The second mount renders normally (the composer is back)…
    expect(await screen.findByLabelText('Message')).toBeInTheDocument();
    // …but the welcome line does not return.
    expect(screen.queryByText('Welcome to the store, Ada.')).not.toBeInTheDocument();
  });
});
