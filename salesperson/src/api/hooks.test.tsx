// Integration-level coverage for the pieces `dispatch.test.ts` cannot reach
// on its own: C3's "must not navigate away from /shop/presenter" nuance
// (needs a router in the loop), C7's actual session+navigation side effect,
// C8's shared polling interval, and C12's no-retry / networkMode behaviour
// (needs TanStack Query's real retry/pause machinery, which is exactly the
// layer §5.3 says a call-count assertion is the only way to reach).
import { act, render, waitFor } from '@testing-library/react';
import { QueryClient, QueryClientProvider, onlineManager } from '@tanstack/react-query';
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';
import { MemoryRouter, useLocation } from 'react-router-dom';
import { SessionProvider, useSession } from '../session/SessionContext';
import { saveParticipantSession } from '../session/storage';
import {
  reconcilePostMessageFailure,
  useAdvanceOrder,
  useCatalog,
  useErrorEffects,
  useMessages,
  usePostMessage,
  usePresenterParticipants,
  usePresenterResetAll,
  useResetMine,
  useShopState,
} from './hooks';
import { POLL_INTERVAL_MS } from './polling';

// C8's dedicated test below overrides this module-wide so it can assert
// against a value that is NOT today's real production constant (2_000) —
// the only way to tell "both hooks import the shared binding" apart from
// "both hooks happen to hard-code the same number today" (§5.3 C8; a
// reverted mutation of `useMessages`'s `refetchInterval` to a hard-coded
// `2_000` left the whole suite green before this mock existed, precisely
// because the real constant IS 2_000). `vi.mock` calls are hoisted above
// every import in this file, so this applies to every test here, including
// the "both refetch together" test further down — which stays correct
// because it never hard-codes the constant's value, only reads it.
vi.mock('./polling', () => ({ POLL_INTERVAL_MS: 5_000 }));

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

function sampleStateResponse() {
  return {
    profile: { name: 'Ada', deliveryAddress: null },
    cart: { items: [], total: 0 },
    order: null,
    turn: { state: 'idle', queuePosition: 0, lastTurn: null },
  };
}

/** The URL + method of every recorded `fetch` call, in order — used by the
 * C4 network-effect tests below to find the write and check what the client
 * actually requested next, rather than only asserting the pure
 * classification `resolveErrorAction` returns (dispatch.test.ts already does
 * that; these tests reach the network-effect layer it cannot). */
function fetchCallLog(fetchMock: ReturnType<typeof vi.fn>): string[] {
  return fetchMock.mock.calls.map((call: unknown[]) => {
    const [input, init] = call as [RequestInfo | URL, RequestInit?];
    const method = (init?.method ?? 'GET').toUpperCase();
    return `${method} ${String(input)}`;
  });
}

function Providers({
  children,
  client,
  initialPath = '/',
}: {
  children: React.ReactNode;
  client: QueryClient;
  initialPath?: string;
}) {
  return (
    <QueryClientProvider client={client}>
      <MemoryRouter initialEntries={[initialPath]}>
        <SessionProvider>{children}</SessionProvider>
      </MemoryRouter>
    </QueryClientProvider>
  );
}

function LocationProbe({ onChange }: { onChange: (pathname: string) => void }) {
  const location = useLocation();
  onChange(location.pathname);
  return null;
}

beforeEach(() => {
  window.localStorage.clear();
});

afterEach(() => {
  vi.unstubAllGlobals();
  vi.useRealTimers();
  onlineManager.setOnline(true);
});

describe('reconcilePostMessageFailure (C4\'s POST /messages reconciliation)', () => {
  const state = (turnState: 'idle' | 'queued' | 'thinking') => ({
    profile: { name: 'A', deliveryAddress: null },
    cart: { items: [], total: 0 },
    order: null,
    turn: { state: turnState, queuePosition: 0, lastTurn: null },
  });

  it('message present and turn idle -> turnLost', () => {
    const rows = [{ msgId: '1', threadId: 't', authorId: 'p', text: 'hi', role: 'user', createdAt: 1, mentions: [] }];
    expect(reconcilePostMessageFailure('hi', rows, state('idle'))).toBe('turnLost');
  });

  it('message present and turn still running -> turnRunning', () => {
    const rows = [{ msgId: '1', threadId: 't', authorId: 'p', text: 'hi', role: 'user', createdAt: 1, mentions: [] }];
    expect(reconcilePostMessageFailure('hi', rows, state('thinking'))).toBe('turnRunning');
  });

  it('message absent -> nothingCommitted', () => {
    expect(reconcilePostMessageFailure('hi', [], state('idle'))).toBe('nothingCommitted');
  });
});

// C4 (hook level) — dispatch.test.ts's C4 block proves `resolveErrorAction`
// classifies each writing route's 504 correctly as a pure function; it never
// drives an actual `fetch`, so it cannot see whether the hook that consumes
// the classification issues the right *network* re-read. These four tests
// mock a global `fetch`, fail the write with a 504, and assert on the next
// recorded call's URL — the same call-inspection pattern C8/C12 already use
// on this surface (`hooks.test.tsx` below). A reverted mutation of
// `usePresenterResetAll`'s `onError` check (`resolved.via.endpoint ===
// 'state'` instead of `'presenterParticipants'`) left the full suite green
// before these existed.
describe('C4 (hook level) — resetMine\'s 504 actually re-reads GET /shop/api/state', () => {
  it('the next fetch call after the failed write targets /shop/api/state', async () => {
    saveParticipantSession({ participantId: 'p-1', token: 'tok', displayName: 'Ada', language: 'en' });
    const fetchMock = vi.fn(async (input: RequestInfo | URL, init?: RequestInit) => {
      const url = String(input);
      const method = (init?.method ?? 'GET').toUpperCase();
      if (method === 'POST' && url.includes('/shop/api/reset')) {
        return jsonResponse(504, null);
      }
      if (url.includes('/shop/api/state')) {
        return jsonResponse(200, sampleStateResponse());
      }
      throw new Error(`unexpected fetch: ${method} ${url}`);
    });
    vi.stubGlobal('fetch', fetchMock);
    const client = makeClient();
    let mutateFn: ReturnType<typeof useResetMine>['mutate'] | null = null;

    function Capture() {
      // An active /state observer, so C4's invalidateQueries actually issues
      // a fetch rather than merely marking an unobserved query stale.
      useShopState();
      mutateFn = useResetMine().mutate;
      return null;
    }

    render(
      <Providers client={client}>
        <Capture />
      </Providers>,
    );

    await waitFor(() => expect(fetchMock.mock.calls.length).toBeGreaterThanOrEqual(1));

    await act(async () => {
      mutateFn!();
      await new Promise((resolve) => setTimeout(resolve, 0));
    });

    await waitFor(() => {
      const log = fetchCallLog(fetchMock);
      const writeIndex = log.findIndex((c) => c.includes('/shop/api/reset'));
      expect(writeIndex).toBeGreaterThanOrEqual(0);
      expect(log.length).toBeGreaterThan(writeIndex + 1);
    });

    const log = fetchCallLog(fetchMock);
    const writeIndex = log.findIndex((c) => c.includes('/shop/api/reset'));
    expect(log[writeIndex + 1]).toMatch(/^GET .*\/shop\/api\/state/);
  });
});

describe('C4 (hook level) — orderAdvance\'s 504 actually re-reads GET /shop/api/state', () => {
  it('the next fetch call after the failed write targets /shop/api/state', async () => {
    saveParticipantSession({ participantId: 'p-1', token: 'tok', displayName: 'Ada', language: 'en' });
    const fetchMock = vi.fn(async (input: RequestInfo | URL, init?: RequestInit) => {
      const url = String(input);
      const method = (init?.method ?? 'GET').toUpperCase();
      if (method === 'POST' && url.includes('/shop/api/order/advance')) {
        return jsonResponse(504, null);
      }
      if (url.includes('/shop/api/state')) {
        return jsonResponse(200, sampleStateResponse());
      }
      throw new Error(`unexpected fetch: ${method} ${url}`);
    });
    vi.stubGlobal('fetch', fetchMock);
    const client = makeClient();
    let mutateFn: ReturnType<typeof useAdvanceOrder>['mutate'] | null = null;

    function Capture() {
      useShopState();
      mutateFn = useAdvanceOrder().mutate;
      return null;
    }

    render(
      <Providers client={client}>
        <Capture />
      </Providers>,
    );

    await waitFor(() => expect(fetchMock.mock.calls.length).toBeGreaterThanOrEqual(1));

    await act(async () => {
      mutateFn!('fulfill');
      await new Promise((resolve) => setTimeout(resolve, 0));
    });

    await waitFor(() => {
      const log = fetchCallLog(fetchMock);
      const writeIndex = log.findIndex((c) => c.includes('/shop/api/order/advance'));
      expect(writeIndex).toBeGreaterThanOrEqual(0);
      expect(log.length).toBeGreaterThan(writeIndex + 1);
    });

    const log = fetchCallLog(fetchMock);
    const writeIndex = log.findIndex((c) => c.includes('/shop/api/order/advance'));
    expect(log[writeIndex + 1]).toMatch(/^GET .*\/shop\/api\/state/);
  });
});

describe('C4 (hook level) — presenterResetAll\'s 504 actually re-reads GET /shop/api/presenter/participants', () => {
  it('the next fetch call after the failed write targets /shop/api/presenter/participants', async () => {
    window.localStorage.setItem('salesperson.presenter', JSON.stringify({ token: 'ptok' }));
    const fetchMock = vi.fn(async (input: RequestInfo | URL, init?: RequestInit) => {
      const url = String(input);
      const method = (init?.method ?? 'GET').toUpperCase();
      if (method === 'POST' && url.includes('/shop/api/presenter/reset-all')) {
        return jsonResponse(504, null);
      }
      if (url.includes('/shop/api/presenter/participants')) {
        return jsonResponse(200, []);
      }
      throw new Error(`unexpected fetch: ${method} ${url}`);
    });
    vi.stubGlobal('fetch', fetchMock);
    const client = makeClient();
    let mutateFn: ReturnType<typeof usePresenterResetAll>['mutate'] | null = null;

    function Capture() {
      // An active roster observer, so C4's invalidateQueries actually issues
      // a fetch rather than merely marking an unobserved query stale.
      usePresenterParticipants();
      mutateFn = usePresenterResetAll().mutate;
      return null;
    }

    render(
      <Providers client={client} initialPath="/presenter">
        <Capture />
      </Providers>,
    );

    await waitFor(() => expect(fetchMock.mock.calls.length).toBeGreaterThanOrEqual(1));

    await act(async () => {
      mutateFn!();
      await new Promise((resolve) => setTimeout(resolve, 0));
    });

    await waitFor(() => {
      const log = fetchCallLog(fetchMock);
      const writeIndex = log.findIndex((c) => c.includes('/shop/api/presenter/reset-all'));
      expect(writeIndex).toBeGreaterThanOrEqual(0);
      expect(log.length).toBeGreaterThan(writeIndex + 1);
    });

    const log = fetchCallLog(fetchMock);
    const writeIndex = log.findIndex((c) => c.includes('/shop/api/presenter/reset-all'));
    expect(log[writeIndex + 1]).toMatch(/^GET .*\/shop\/api\/presenter\/participants/);
  });
});

describe('C4 (hook level) — messagesPost\'s 504 actually re-reads GET /shop/api/messages AND GET /shop/api/state', () => {
  it('the two fetch calls after the failed write target /messages then /state, in that order', async () => {
    saveParticipantSession({ participantId: 'p-1', token: 'tok', displayName: 'Ada', language: 'en' });
    const fetchMock = vi.fn(async (input: RequestInfo | URL, init?: RequestInit) => {
      const url = String(input);
      const method = (init?.method ?? 'GET').toUpperCase();
      if (method === 'POST' && url.includes('/shop/api/messages')) {
        return jsonResponse(504, null);
      }
      if (method === 'GET' && url.includes('/shop/api/messages')) {
        return jsonResponse(200, []);
      }
      if (url.includes('/shop/api/state')) {
        return jsonResponse(200, sampleStateResponse());
      }
      throw new Error(`unexpected fetch: ${method} ${url}`);
    });
    vi.stubGlobal('fetch', fetchMock);
    const client = makeClient();
    let mutateFn: ReturnType<typeof usePostMessage>['mutate'] | null = null;

    function Capture() {
      mutateFn = usePostMessage().mutate;
      return null;
    }

    render(
      <Providers client={client}>
        <Capture />
      </Providers>,
    );

    await act(async () => {
      mutateFn!('hello');
      await new Promise((resolve) => setTimeout(resolve, 0));
    });

    await waitFor(() => {
      const log = fetchCallLog(fetchMock);
      const writeIndex = log.findIndex((c) => c.startsWith('POST') && c.includes('/shop/api/messages'));
      expect(writeIndex).toBeGreaterThanOrEqual(0);
      expect(log.length).toBeGreaterThanOrEqual(writeIndex + 3);
    });

    const log = fetchCallLog(fetchMock);
    const writeIndex = log.findIndex((c) => c.startsWith('POST') && c.includes('/shop/api/messages'));
    expect(log[writeIndex + 1]).toMatch(/^GET .*\/shop\/api\/messages/);
    expect(log[writeIndex + 2]).toMatch(/^GET .*\/shop\/api\/state/);
  });
});

describe('C3 (hook level) — clearing the participant must not navigate away from /shop/presenter', () => {
  it('stays on /presenter when the clear fires while the presenter view is mounted', async () => {
    saveParticipantSession({ participantId: 'p-1', token: 'tok', displayName: 'Ada', language: 'en' });
    const client = makeClient();
    let currentPath = '';
    let dispatchFn: ReturnType<typeof useErrorEffects> | null = null;

    function Capture() {
      dispatchFn = useErrorEffects();
      return null;
    }

    render(
      <Providers client={client} initialPath="/presenter">
        <Capture />
        <LocationProbe onChange={(p) => (currentPath = p)} />
      </Providers>,
    );

    await act(async () => {
      dispatchFn!('state', { status: 401, body: { error: 'invalid_token' } } as never, 'poll');
    });

    expect(currentPath).toBe('/presenter');
  });

  it('does navigate to the join screen when the clear fires on the participant view', async () => {
    saveParticipantSession({ participantId: 'p-1', token: 'tok', displayName: 'Ada', language: 'en' });
    const client = makeClient();
    let currentPath = '';
    let dispatchFn: ReturnType<typeof useErrorEffects> | null = null;

    function Capture() {
      dispatchFn = useErrorEffects();
      return null;
    }

    render(
      <Providers client={client} initialPath="/somewhere-else">
        <Capture />
        <LocationProbe onChange={(p) => (currentPath = p)} />
      </Providers>,
    );

    await act(async () => {
      dispatchFn!('state', { status: 401, body: { error: 'invalid_token' } } as never, 'poll');
    });

    expect(currentPath).toBe('/');
  });
});

describe('C7 (hook level) — reset-mine keeps the credential and lands on the language step', () => {
  it('sets pendingLanguageStep and navigates to the participant path on success', async () => {
    saveParticipantSession({ participantId: 'p-1', token: 'tok', displayName: 'Ada', language: 'en' });
    vi.stubGlobal(
      'fetch',
      vi.fn(async () => jsonResponse(200, { threadId: 'th-1', language: 'pt-BR' })),
    );
    const client = makeClient();
    let sessionBox: ReturnType<typeof useSession> | null = null;
    let mutateFn: ReturnType<typeof useResetMine>['mutate'] | null = null;
    let currentPath = '';

    function Capture() {
      sessionBox = useSession();
      mutateFn = useResetMine().mutate;
      return null;
    }

    render(
      <Providers client={client} initialPath="/somewhere">
        <Capture />
        <LocationProbe onChange={(p) => (currentPath = p)} />
      </Providers>,
    );

    await act(async () => {
      mutateFn!();
      await new Promise((resolve) => setTimeout(resolve, 0));
    });

    await waitFor(() => {
      expect(sessionBox!.pendingLanguageStep).toBe('pt-BR');
    });
    // The credential itself survives reset-mine (§4.8) — only the language
    // step flag changes.
    expect(sessionBox!.participant?.participantId).toBe('p-1');
    expect(currentPath).toBe('/');
  });
});

describe('C8 — /state and /messages poll on one shared exported constant', () => {
  it('both refetch together when POLL_INTERVAL_MS elapses, and neither refetches before it', async () => {
    vi.useFakeTimers();
    saveParticipantSession({ participantId: 'p-1', token: 'tok', displayName: 'Ada', language: 'en' });
    const fetchMock = vi.fn(async (input: RequestInfo | URL) => {
      const url = String(input);
      if (url.includes('/messages')) return jsonResponse(200, []);
      return jsonResponse(200, {
        profile: { name: 'Ada', deliveryAddress: null },
        cart: { items: [], total: 0 },
        order: null,
        turn: { state: 'idle', queuePosition: 0, lastTurn: null },
      });
    });
    vi.stubGlobal('fetch', fetchMock);
    const client = makeClient();

    function Capture() {
      useShopState();
      useMessages();
      return null;
    }

    render(
      <Providers client={client}>
        <Capture />
      </Providers>,
    );

    await act(async () => {
      await vi.advanceTimersByTimeAsync(0);
    });
    const callsAfterMount = fetchMock.mock.calls.length;
    expect(callsAfterMount).toBeGreaterThanOrEqual(2); // one /state + one /messages

    // Well before the shared interval elapses: no additional calls.
    await act(async () => {
      await vi.advanceTimersByTimeAsync(POLL_INTERVAL_MS - 100);
    });
    expect(fetchMock.mock.calls.length).toBe(callsAfterMount);

    // At the shared interval: both fire again.
    await act(async () => {
      await vi.advanceTimersByTimeAsync(200);
    });
    expect(fetchMock.mock.calls.length).toBeGreaterThanOrEqual(callsAfterMount + 2);
  });
});

describe('C8 — the two poll hooks import one shared exported symbol, not two literals', () => {
  it('both /state and /messages refetch at the mocked interval, which is not their real production value', () => {
    // Sanity on the module-level `vi.mock('./polling', ...)` above: if this
    // ever reads 2_000 (today's real constant), the test below stops
    // proving anything, because a hard-coded `2_000` would then be
    // indistinguishable from the shared import again.
    expect(POLL_INTERVAL_MS).toBe(5_000);
  });

  it('both fire together at the mocked interval — proven with a value that differs from the real constant, so a hard-coded literal cannot coincidentally match it', async () => {
    vi.useFakeTimers();
    saveParticipantSession({ participantId: 'p-1', token: 'tok', displayName: 'Ada', language: 'en' });
    const fetchMock = vi.fn(async (input: RequestInfo | URL) => {
      const url = String(input);
      if (url.includes('/messages')) return jsonResponse(200, []);
      return jsonResponse(200, {
        profile: { name: 'Ada', deliveryAddress: null },
        cart: { items: [], total: 0 },
        order: null,
        turn: { state: 'idle', queuePosition: 0, lastTurn: null },
      });
    });
    vi.stubGlobal('fetch', fetchMock);
    const client = makeClient();

    function Capture() {
      useShopState();
      useMessages();
      return null;
    }

    render(
      <Providers client={client}>
        <Capture />
      </Providers>,
    );

    await act(async () => {
      await vi.advanceTimersByTimeAsync(0);
    });
    const callsAfterMount = fetchMock.mock.calls.length;
    expect(callsAfterMount).toBeGreaterThanOrEqual(2);

    // Just under the mocked interval (5_000 - 100 = 4_900ms): a hook hard-
    // coded to the real 2_000ms constant would already have ticked twice by
    // here (at 2_000ms and 4_000ms), so this is exactly where that
    // regression would surface as extra calls.
    await act(async () => {
      await vi.advanceTimersByTimeAsync(POLL_INTERVAL_MS - 100);
    });
    expect(fetchMock.mock.calls.length).toBe(callsAfterMount);

    // At the mocked interval: both fire together.
    await act(async () => {
      await vi.advanceTimersByTimeAsync(200);
    });
    expect(fetchMock.mock.calls.length).toBeGreaterThanOrEqual(callsAfterMount + 2);
  });
});

describe('C12 — no automatic retry on mutations; the browser layer is the only one that can prove it', () => {
  it('POST /messages is never retried after a 5xx (retry: 0 pinned explicitly)', async () => {
    vi.useFakeTimers();
    saveParticipantSession({ participantId: 'p-1', token: 'tok', displayName: 'Ada', language: 'en' });
    const fetchMock = vi.fn(async () => jsonResponse(500, null));
    vi.stubGlobal('fetch', fetchMock);
    const client = makeClient();
    let mutateFn: ReturnType<typeof usePostMessage>['mutate'] | null = null;

    function Capture() {
      mutateFn = usePostMessage().mutate;
      return null;
    }

    render(
      <Providers client={client}>
        <Capture />
      </Providers>,
    );

    await act(async () => {
      mutateFn!('hello');
      await vi.advanceTimersByTimeAsync(0);
    });
    expect(fetchMock).toHaveBeenCalledTimes(1);

    // Even after a generous window (well past the library's default 1s/2s/4s
    // backoff ladder), still exactly one call.
    await act(async () => {
      await vi.advanceTimersByTimeAsync(10_000);
    });
    expect(fetchMock).toHaveBeenCalledTimes(1);
  });

  it('the one-shot catalog fetch keeps a bounded 5xx-only retry (unlike every other route)', async () => {
    vi.useFakeTimers();
    saveParticipantSession({ participantId: 'p-1', token: 'tok', displayName: 'Ada', language: 'en' });
    const fetchMock = vi.fn(async () => jsonResponse(500, null));
    vi.stubGlobal('fetch', fetchMock);
    const client = makeClient();

    function Capture() {
      useCatalog();
      return null;
    }

    render(
      <Providers client={client}>
        <Capture />
      </Providers>,
    );

    await act(async () => {
      await vi.advanceTimersByTimeAsync(0);
    });
    await act(async () => {
      await vi.advanceTimersByTimeAsync(1_000);
    });
    await act(async () => {
      await vi.advanceTimersByTimeAsync(2_000);
    });
    await act(async () => {
      await vi.advanceTimersByTimeAsync(4_000);
    });

    // One initial attempt + at most 2 retries (failureCount < 2).
    expect(fetchMock.mock.calls.length).toBeLessThanOrEqual(3);
    expect(fetchMock.mock.calls.length).toBeGreaterThan(1);
  });

  it('the catalog fetch does not retry a 401 — only a 5xx qualifies', async () => {
    vi.useFakeTimers();
    saveParticipantSession({ participantId: 'p-1', token: 'tok', displayName: 'Ada', language: 'en' });
    const fetchMock = vi.fn(async () => jsonResponse(401, { error: 'invalid_token' }));
    vi.stubGlobal('fetch', fetchMock);
    const client = makeClient();

    function Capture() {
      useCatalog();
      return null;
    }

    render(
      <Providers client={client} initialPath="/">
        <Capture />
      </Providers>,
    );

    await act(async () => {
      await vi.advanceTimersByTimeAsync(0);
    });
    await act(async () => {
      await vi.advanceTimersByTimeAsync(5_000);
    });
    expect(fetchMock).toHaveBeenCalledTimes(1);
  });
});

describe('C12 — the reset mutations pin networkMode "always"; everything else stays default', () => {
  it('reset-mine fires immediately while the browser is offline', async () => {
    saveParticipantSession({ participantId: 'p-1', token: 'tok', displayName: 'Ada', language: 'en' });
    onlineManager.setOnline(false);
    const fetchMock = vi.fn(async () => jsonResponse(200, { threadId: 't', language: 'en' }));
    vi.stubGlobal('fetch', fetchMock);
    const client = makeClient();
    let mutateFn: ReturnType<typeof useResetMine>['mutate'] | null = null;

    function Capture() {
      mutateFn = useResetMine().mutate;
      return null;
    }

    render(
      <Providers client={client}>
        <Capture />
      </Providers>,
    );

    await act(async () => {
      mutateFn!();
      await new Promise((resolve) => setTimeout(resolve, 0));
    });

    expect(fetchMock).toHaveBeenCalledTimes(1);
  });

  it('reset-everyone fires immediately while the browser is offline', async () => {
    window.localStorage.setItem('salesperson.presenter', JSON.stringify({ token: 'ptok' }));
    onlineManager.setOnline(false);
    const fetchMock = vi.fn(async () => jsonResponse(200, { clearedParticipants: 3 }));
    vi.stubGlobal('fetch', fetchMock);
    const client = makeClient();
    let mutateFn: ReturnType<typeof usePresenterResetAll>['mutate'] | null = null;

    function Capture() {
      mutateFn = usePresenterResetAll().mutate;
      return null;
    }

    render(
      <Providers client={client} initialPath="/presenter">
        <Capture />
      </Providers>,
    );

    await act(async () => {
      mutateFn!();
      await new Promise((resolve) => setTimeout(resolve, 0));
    });

    expect(fetchMock).toHaveBeenCalledTimes(1);
  });

  it('by contrast, POST /messages (default networkMode) does NOT fire while offline', async () => {
    saveParticipantSession({ participantId: 'p-1', token: 'tok', displayName: 'Ada', language: 'en' });
    onlineManager.setOnline(false);
    const fetchMock = vi.fn(async () => jsonResponse(200, { msgId: '1' }));
    vi.stubGlobal('fetch', fetchMock);
    const client = makeClient();
    let mutateFn: ReturnType<typeof usePostMessage>['mutate'] | null = null;

    function Capture() {
      mutateFn = usePostMessage().mutate;
      return null;
    }

    render(
      <Providers client={client}>
        <Capture />
      </Providers>,
    );

    await act(async () => {
      mutateFn!('hi');
      await new Promise((resolve) => setTimeout(resolve, 0));
    });

    expect(fetchMock).not.toHaveBeenCalled();
  });
});
