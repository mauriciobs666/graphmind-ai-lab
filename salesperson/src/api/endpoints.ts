// The eleven `/shop/api` routes (docs/plans/salesperson-ui.md §5.2), typed.
// One function per route, aggregated as `apiClient` — the third symbol S12a's
// row names alongside `useSession()`/`useShopState()`. No route here decides
// what a failure means; that is `./dispatch`'s job against the `ApiError`
// these throw.
import { type ApiFetchOptions, apiFetch } from './client';

const API_PREFIX = '/shop/api';

export interface HealthResponse {
  status: string;
  storefrontEnabled: boolean;
  locales: string[];
}

export function getHealth(signal?: AbortSignal): Promise<HealthResponse> {
  return apiFetch<HealthResponse>({
    method: 'GET',
    path: `${API_PREFIX}/health`,
    credential: 'none',
    signal,
  });
}

export interface JoinRequest {
  displayName: string;
  language: string;
}

export interface JoinResponse {
  participantId: string;
  token: string;
  displayName: string;
  language: string;
  /** One-shot join greeting — not session state (§5.2/§5.3: deliberately not
   * stored alongside the other four keys). */
  welcome: string;
}

export function joinSession(body: JoinRequest, signal?: AbortSignal): Promise<JoinResponse> {
  return apiFetch<JoinResponse>({
    method: 'POST',
    path: `${API_PREFIX}/session`,
    credential: 'none',
    body,
    signal,
  });
}

export type TurnPhase = 'idle' | 'queued' | 'thinking';

export interface TurnBlock {
  state: TurnPhase;
  queuePosition: number;
  lastTurn: 'failed' | null;
}

export interface CartItem {
  productId: string;
  name: string;
  quantity: number;
  /** DEF-2: named `price`, not `unitPrice` — matches the server's actual
   * `/shop/api/state` `cart.items[]` shape (`falkor-chat/server/falkorchat/
   * services.py`'s `_priced_cart_lines`/`get_cart`). `OrderBlock.lines`'
   * frozen order lines are a *separate* server shape that does use
   * `unitPrice` (`place_order`'s `order_lines`) — do not conflate the two. */
  price: number;
}

export interface OrderBlock {
  orderId: string;
  status: string;
  lines: unknown[];
  total: number;
}

export interface StateResponse {
  profile: { name: string; deliveryAddress: string | null };
  cart: { items: CartItem[]; total: number };
  order: OrderBlock | null;
  turn: TurnBlock;
}

function participantGet<T>(
  path: string,
  authHeader: string,
  opts: { query?: ApiFetchOptions['query']; signal?: AbortSignal } = {},
): Promise<T> {
  return apiFetch<T>({
    method: 'GET',
    path,
    credential: 'participant',
    authHeader,
    query: opts.query,
    signal: opts.signal,
  });
}

export function getState(authHeader: string, signal?: AbortSignal): Promise<StateResponse> {
  return participantGet<StateResponse>(`${API_PREFIX}/state`, authHeader, { signal });
}

export interface MessageRow {
  msgId: string;
  threadId: string;
  authorId: string;
  text: string;
  role: string;
  createdAt: number;
  mentions: string[];
}

export function getMessages(
  authHeader: string,
  opts: { since?: number; limit?: number; signal?: AbortSignal } = {},
): Promise<MessageRow[]> {
  return participantGet<MessageRow[]>(`${API_PREFIX}/messages`, authHeader, {
    query: { since: opts.since ?? 0, limit: opts.limit },
    signal: opts.signal,
  });
}

export function postMessage(
  authHeader: string,
  text: string,
  signal?: AbortSignal,
): Promise<MessageRow> {
  return apiFetch<MessageRow>({
    method: 'POST',
    path: `${API_PREFIX}/messages`,
    credential: 'participant',
    authHeader,
    body: { text },
    signal,
  });
}

export interface CatalogEntry {
  productId: string;
  name: string;
  category: string;
  price: number;
  imageUrl: string | null;
}

export function getCatalog(authHeader: string, signal?: AbortSignal): Promise<CatalogEntry[]> {
  return participantGet<CatalogEntry[]>(`${API_PREFIX}/catalog`, authHeader, { signal });
}

export type OrderTransition = 'fulfill' | 'deliver' | 'cancel';

export interface AdvanceOrderResponse {
  orderId: string;
  status: string;
}

export function advanceOrder(
  authHeader: string,
  transition: OrderTransition,
  signal?: AbortSignal,
): Promise<AdvanceOrderResponse> {
  return apiFetch<AdvanceOrderResponse>({
    method: 'POST',
    path: `${API_PREFIX}/order/advance`,
    credential: 'participant',
    authHeader,
    body: { transition },
    signal,
  });
}

export interface ResetMineResponse {
  threadId: string;
  language: string;
}

export function resetMine(authHeader: string, signal?: AbortSignal): Promise<ResetMineResponse> {
  return apiFetch<ResetMineResponse>({
    method: 'POST',
    path: `${API_PREFIX}/reset`,
    credential: 'participant',
    authHeader,
    signal,
  });
}

export interface PresenterSessionResponse {
  token: string;
}

export function presenterSession(
  key: string,
  signal?: AbortSignal,
): Promise<PresenterSessionResponse> {
  return apiFetch<PresenterSessionResponse>({
    method: 'POST',
    path: `${API_PREFIX}/presenter/session`,
    credential: 'none',
    body: { key },
    signal,
  });
}

export interface PresenterParticipantRow {
  participantId: string;
  displayName: string;
  language: string;
  joinedAt: number;
}

export function presenterParticipants(
  authHeader: string,
  signal?: AbortSignal,
): Promise<PresenterParticipantRow[]> {
  return apiFetch<PresenterParticipantRow[]>({
    method: 'GET',
    path: `${API_PREFIX}/presenter/participants`,
    credential: 'presenter',
    authHeader,
    signal,
  });
}

// `incomplete`/`unresolved` are a co-varying pair, omitted together and
// present together, never mixed (§5.2's absent-vs-null rule) — typed here so
// S12d can render the partial-sweep case without re-deriving the shape.
export interface PresenterResetAllResponse {
  clearedParticipants: number;
  incomplete?: true;
  unresolved?: string[];
}

export function presenterResetAll(
  authHeader: string,
  signal?: AbortSignal,
): Promise<PresenterResetAllResponse> {
  return apiFetch<PresenterResetAllResponse>({
    method: 'POST',
    path: `${API_PREFIX}/presenter/reset-all`,
    credential: 'presenter',
    authHeader,
    signal,
  });
}

export const apiClient = {
  getHealth,
  joinSession,
  getState,
  getMessages,
  postMessage,
  getCatalog,
  advanceOrder,
  resetMine,
  presenterSession,
  presenterParticipants,
  presenterResetAll,
};
