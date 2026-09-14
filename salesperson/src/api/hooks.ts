// The React/TanStack Query wiring over `./dispatch` + `./endpoints`.
// `useShopState()` is one of S12a's named exports (§5.1's S12a row); the
// other data hooks here follow its shape. Every hook that can receive an
// `ApiError` runs it through `useErrorEffects` (below), which is the single
// place credential-clearing and navigation happen — the mechanism that makes
// C1 structural rather than a discipline every hook has to remember.
import { useEffect, useState } from 'react';
import {
  type UseMutationResult,
  type UseQueryResult,
  useMutation,
  useQuery,
  useQueryClient,
} from '@tanstack/react-query';
import { useLocation, useNavigate } from 'react-router-dom';
import { APP_PATHS } from '../routePaths';
import { useSession } from '../session/SessionContext';
import { participantAuthHeader, presenterAuthHeader } from '../session/storage';
import { ApiError } from './client';
import { type ErrorAction, type Route, type Trigger, resolveErrorAction } from './dispatch';
import {
  type AdvanceOrderResponse,
  type CatalogEntry,
  type HealthResponse,
  type JoinRequest,
  type JoinResponse,
  type MessageRow,
  type OrderTransition,
  type PresenterParticipantRow,
  type PresenterResetAllResponse,
  type PresenterSessionResponse,
  type ResetMineResponse,
  type StateResponse,
  advanceOrder,
  getCatalog,
  getHealth,
  getMessages,
  getState,
  joinSession,
  postMessage,
  presenterParticipants as fetchPresenterParticipants,
  presenterResetAll as fetchPresenterResetAll,
  presenterSession as fetchPresenterSession,
  resetMine as fetchResetMine,
} from './endpoints';
import { POLL_INTERVAL_MS } from './polling';

export const queryKeys = {
  state: (participantId?: string | null) => ['state', participantId] as const,
  messages: (participantId?: string | null) => ['messages', participantId] as const,
  catalog: (participantId?: string | null) => ['catalog', participantId] as const,
  presenterParticipants: (presenterToken?: string | null) =>
    ['presenterParticipants', presenterToken] as const,
};

/** The one place C1–C3's side effects happen: which credential a failed
 * request carried decides what is cleared and where the user lands — never
 * a single "on 401, go to X" branch. Every other rule (C4, C6a/b, C9–C14) is
 * a pure classification each hook renders for itself; this function performs
 * no side effect for those, by construction (the `switch` below only has two
 * cases). */
export function useErrorEffects() {
  const navigate = useNavigate();
  const location = useLocation();
  const { clearParticipant, clearPresenter } = useSession();

  return (route: Route, error: ApiError, trigger: Trigger): ErrorAction => {
    const action = resolveErrorAction(route, error, trigger);
    if (action.kind === 'clearParticipant') {
      clearParticipant();
      // C3 — must not navigate the presenter away from /shop/presenter.
      if (!location.pathname.startsWith(APP_PATHS.presenter)) {
        navigate(APP_PATHS.participant, { replace: true });
      }
    } else if (action.kind === 'clearPresenter') {
      clearPresenter();
      navigate(APP_PATHS.presenter, { replace: true });
    }
    return action;
  };
}

/** C4/C6a's message-post reconciliation, isolated as a pure function so it
 * is unit-testable without a mutation in the loop. `text` is the line that
 * was being sent when the `504` fired. */
export type PostMessageReconciliation = 'turnLost' | 'turnRunning' | 'nothingCommitted';

export function reconcilePostMessageFailure(
  text: string,
  freshMessages: readonly MessageRow[],
  freshState: StateResponse,
): PostMessageReconciliation {
  const present = freshMessages.some((row) => row.text === text);
  if (!present) return 'nothingCommitted';
  return freshState.turn.state === 'idle' ? 'turnLost' : 'turnRunning';
}

// ── health (no credential, no graph access — §5.3) ───────────────────────

/** Seeds the join screen's language chooser from the deployment's own
 * `locales` (§5.3 C11's note: this is what makes a UI-supplied `language`
 * `422` unreachable in practice rather than merely well-rendered — S12c
 * wires the actual chooser; this hook is the data it will read). */
export function useHealth(): UseQueryResult<HealthResponse, ApiError> {
  return useQuery<HealthResponse, ApiError>({
    queryKey: ['health'],
    queryFn: ({ signal }) => getHealth(signal),
    staleTime: Number.POSITIVE_INFINITY,
    retry: 0,
  });
}

// ── participant: state (poll, C8) ────────────────────────────────────────

export type UseShopStateResult = UseQueryResult<StateResponse, ApiError> & {
  action: ErrorAction | null;
  /** C6a's second half — `turn.lastTurn === 'failed'`, surfaced without
   * gating the composer. */
  deadTurnNotice: boolean;
};

export function useShopState(): UseShopStateResult {
  const { participant } = useSession();
  const dispatch = useErrorEffects();
  const authHeader = participant ? participantAuthHeader(participant) : null;
  const [action, setAction] = useState<ErrorAction | null>(null);

  const query = useQuery<StateResponse, ApiError>({
    queryKey: queryKeys.state(participant?.participantId),
    queryFn: ({ signal }) => getState(authHeader as string, signal),
    enabled: Boolean(authHeader),
    refetchInterval: POLL_INTERVAL_MS, // C8 — shared constant
    retry: 0, // C12 — refetchInterval already is the retry
  });

  useEffect(() => {
    setAction(query.error ? dispatch('state', query.error, 'poll') : null);
    // dispatch is re-created each render (it closes over navigate/location);
    // keying only on the error identity is what makes this fire once per
    // failure rather than once per render.
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [query.error]);

  return {
    ...query,
    action,
    deadTurnNotice: query.data?.turn.lastTurn === 'failed',
  };
}

// ── participant: messages (poll, C8) ─────────────────────────────────────

export type UseMessagesResult = UseQueryResult<MessageRow[], ApiError> & {
  action: ErrorAction | null;
};

export function useMessages(since = 0): UseMessagesResult {
  const { participant } = useSession();
  const dispatch = useErrorEffects();
  const authHeader = participant ? participantAuthHeader(participant) : null;
  const [action, setAction] = useState<ErrorAction | null>(null);

  const query = useQuery<MessageRow[], ApiError>({
    queryKey: queryKeys.messages(participant?.participantId),
    queryFn: ({ signal }) => getMessages(authHeader as string, { since, signal }),
    enabled: Boolean(authHeader),
    refetchInterval: POLL_INTERVAL_MS, // C8 — same shared constant as /state
    retry: 0,
  });

  useEffect(() => {
    setAction(query.error ? dispatch('messagesRead', query.error, 'poll') : null);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [query.error]);

  return { ...query, action };
}

// ── participant: post message ────────────────────────────────────────────

export type UsePostMessageResult = UseMutationResult<MessageRow, ApiError, string> & {
  action: ErrorAction | null;
  reconciliation: PostMessageReconciliation | null;
};

export function usePostMessage(): UsePostMessageResult {
  const { participant } = useSession();
  const dispatch = useErrorEffects();
  const queryClient = useQueryClient();
  const authHeader = participant ? participantAuthHeader(participant) : null;
  const [action, setAction] = useState<ErrorAction | null>(null);
  const [reconciliation, setReconciliation] = useState<PostMessageReconciliation | null>(null);

  const mutation = useMutation<MessageRow, ApiError, string>({
    mutationFn: (text) => postMessage(authHeader as string, text),
    retry: 0, // C12 — pinned explicitly rather than inherited
    onSuccess: async () => {
      setAction(null);
      setReconciliation(null);
      await Promise.all([
        queryClient.invalidateQueries({ queryKey: queryKeys.messages(participant?.participantId) }),
        queryClient.invalidateQueries({ queryKey: queryKeys.state(participant?.participantId) }),
      ]);
    },
    onError: async (error, text) => {
      const resolved = dispatch('messagesPost', error, 'user');
      setAction(resolved);
      if (resolved.kind === 'reread' && resolved.via.endpoint === 'messagesAndState') {
        // C4's POST /messages row: re-read BOTH and reconcile — never just
        // /messages, which would say nothing about the turn.
        const [freshMessages, freshState] = await Promise.all([
          getMessages(authHeader as string, { since: 0 }),
          getState(authHeader as string),
        ]);
        setReconciliation(reconcilePostMessageFailure(text, freshMessages, freshState));
        await Promise.all([
          queryClient.invalidateQueries({ queryKey: queryKeys.messages(participant?.participantId) }),
          queryClient.invalidateQueries({ queryKey: queryKeys.state(participant?.participantId) }),
        ]);
      }
    },
  });

  return { ...mutation, action, reconciliation };
}

// ── participant: catalog (one-shot, C8/C12) ──────────────────────────────

export type UseCatalogResult = UseQueryResult<CatalogEntry[], ApiError> & {
  action: ErrorAction | null;
};

export function useCatalog(): UseCatalogResult {
  const { participant } = useSession();
  const dispatch = useErrorEffects();
  const authHeader = participant ? participantAuthHeader(participant) : null;
  const [action, setAction] = useState<ErrorAction | null>(null);

  const query = useQuery<CatalogEntry[], ApiError>({
    queryKey: queryKeys.catalog(participant?.participantId),
    queryFn: ({ signal }) => getCatalog(authHeader as string, signal),
    enabled: Boolean(authHeader),
    // C12 — "the one-shot catalog fetch may keep a bounded retry with a
    // 5xx-only predicate". No `refetchInterval`: fetched once.
    retry: (failureCount, error) => {
      if (!(error instanceof ApiError)) return false;
      if (error.status < 500) return false;
      return failureCount < 2;
    },
  });

  useEffect(() => {
    setAction(query.error ? dispatch('catalog', query.error, 'user') : null);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [query.error]);

  return { ...query, action };
}

// ── participant: order advance ───────────────────────────────────────────

export type UseAdvanceOrderResult = UseMutationResult<
  AdvanceOrderResponse,
  ApiError,
  OrderTransition
> & {
  action: ErrorAction | null;
};

export function useAdvanceOrder(): UseAdvanceOrderResult {
  const { participant } = useSession();
  const dispatch = useErrorEffects();
  const queryClient = useQueryClient();
  const authHeader = participant ? participantAuthHeader(participant) : null;
  const [action, setAction] = useState<ErrorAction | null>(null);

  const mutation = useMutation<AdvanceOrderResponse, ApiError, OrderTransition>({
    mutationFn: (transition) => advanceOrder(authHeader as string, transition),
    retry: 0,
    onSuccess: async () => {
      setAction(null);
      await queryClient.invalidateQueries({ queryKey: queryKeys.state(participant?.participantId) });
    },
    onError: async (error) => {
      const resolved = dispatch('orderAdvance', error, 'user');
      setAction(resolved);
      // C10's stale-button outcomes and C4's 504 both resolve the same way
      // here: re-read /state and re-render the order.
      if (resolved.kind === 'reread' || resolved.kind === 'staleOrderRefresh') {
        await queryClient.invalidateQueries({ queryKey: queryKeys.state(participant?.participantId) });
      }
    },
  });

  return { ...mutation, action };
}

// ── participant: reset mine ──────────────────────────────────────────────

export type UseResetMineResult = UseMutationResult<ResetMineResponse, ApiError, void> & {
  action: ErrorAction | null;
};

export function useResetMine(): UseResetMineResult {
  const { participant, setPendingLanguageStep } = useSession();
  const dispatch = useErrorEffects();
  const navigate = useNavigate();
  const queryClient = useQueryClient();
  const authHeader = participant ? participantAuthHeader(participant) : null;
  const [action, setAction] = useState<ErrorAction | null>(null);

  const mutation = useMutation<ResetMineResponse, ApiError, void>({
    mutationFn: () => fetchResetMine(authHeader as string),
    retry: 0,
    networkMode: 'always', // C12 — fail fast rather than queue while offline
    onSuccess: (data) => {
      setAction(null);
      // C7 — the credential survives; land on the language step, never join.
      setPendingLanguageStep(data.language);
      navigate(APP_PATHS.participant, { replace: true });
    },
    onError: async (error) => {
      const resolved = dispatch('resetMine', error, 'user');
      setAction(resolved);
      if (resolved.kind === 'reread' && resolved.via.endpoint === 'state') {
        await queryClient.invalidateQueries({ queryKey: queryKeys.state(participant?.participantId) });
      }
    },
  });

  return { ...mutation, action };
}

// ── join ──────────────────────────────────────────────────────────────────

export type UseJoinResult = UseMutationResult<JoinResponse, ApiError, JoinRequest> & {
  action: ErrorAction | null;
};

export function useJoin(): UseJoinResult {
  const { setParticipant, setPendingLanguageStep } = useSession();
  const dispatch = useErrorEffects();
  const [action, setAction] = useState<ErrorAction | null>(null);

  const mutation = useMutation<JoinResponse, ApiError, JoinRequest>({
    mutationFn: (body) => joinSession(body),
    retry: 0,
    onSuccess: (data) => {
      setAction(null);
      setPendingLanguageStep(null);
      setParticipant({
        participantId: data.participantId,
        token: data.token,
        displayName: data.displayName,
        language: data.language,
      });
    },
    onError: (error) => {
      // C4's join row: the token that would carry a credential is what was
      // lost on a `504` — there is no re-read to perform here, only the
      // "join may not have completed" report, which `action.kind ===
      // 'reread'` (`via.endpoint === 'none'`) already carries for the view.
      setAction(dispatch('join', error, 'user'));
    },
  });

  return { ...mutation, action };
}

// ── presenter: key entry ─────────────────────────────────────────────────

export type UsePresenterLoginResult = UseMutationResult<
  PresenterSessionResponse,
  ApiError,
  string
> & {
  action: ErrorAction | null;
};

export function usePresenterLogin(): UsePresenterLoginResult {
  const { setPresenter } = useSession();
  const dispatch = useErrorEffects();
  const [action, setAction] = useState<ErrorAction | null>(null);

  const mutation = useMutation<PresenterSessionResponse, ApiError, string>({
    mutationFn: (key) => fetchPresenterSession(key),
    retry: 0,
    onSuccess: (data) => {
      setAction(null);
      setPresenter({ token: data.token });
    },
    onError: (error) => {
      // A bad key (C2's second half) resolves to `presenterKeyRejected`,
      // which `useErrorEffects` deliberately leaves un-acted-on: nothing to
      // clear, no navigation, report it in place.
      setAction(dispatch('presenterSession', error, 'user'));
    },
  });

  return { ...mutation, action };
}

// ── presenter: roster (poll, C8) ─────────────────────────────────────────

export type UsePresenterParticipantsResult = UseQueryResult<
  PresenterParticipantRow[],
  ApiError
> & {
  action: ErrorAction | null;
};

export function usePresenterParticipants(): UsePresenterParticipantsResult {
  const { presenter } = useSession();
  const dispatch = useErrorEffects();
  const authHeader = presenter ? presenterAuthHeader(presenter) : null;
  const [action, setAction] = useState<ErrorAction | null>(null);

  const query = useQuery<PresenterParticipantRow[], ApiError>({
    queryKey: queryKeys.presenterParticipants(presenter?.token),
    queryFn: ({ signal }) => fetchPresenterParticipants(authHeader as string, signal),
    enabled: Boolean(authHeader),
    refetchInterval: POLL_INTERVAL_MS,
    retry: 0,
  });

  useEffect(() => {
    setAction(query.error ? dispatch('presenterParticipants', query.error, 'poll') : null);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [query.error]);

  return { ...query, action };
}

// ── presenter: reset everyone ────────────────────────────────────────────

export type UsePresenterResetAllResult = UseMutationResult<
  PresenterResetAllResponse,
  ApiError,
  void
> & {
  action: ErrorAction | null;
};

export function usePresenterResetAll(): UsePresenterResetAllResult {
  const { presenter } = useSession();
  const dispatch = useErrorEffects();
  const queryClient = useQueryClient();
  const authHeader = presenter ? presenterAuthHeader(presenter) : null;
  const [action, setAction] = useState<ErrorAction | null>(null);

  const mutation = useMutation<PresenterResetAllResponse, ApiError, void>({
    mutationFn: () => fetchPresenterResetAll(authHeader as string),
    retry: 0,
    networkMode: 'always', // C12 — fail fast rather than queue while offline
    onSuccess: () => setAction(null),
    onError: async (error) => {
      const resolved = dispatch('presenterResetAll', error, 'user');
      setAction(resolved);
      if (resolved.kind === 'reread' && resolved.via.endpoint === 'presenterParticipants') {
        await queryClient.invalidateQueries({
          queryKey: queryKeys.presenterParticipants(presenter?.token),
        });
      }
    },
  });

  return { ...mutation, action };
}
