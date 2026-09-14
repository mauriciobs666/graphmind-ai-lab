// The transport layer under `apiClient` (docs/plans/salesperson-ui.md §5.1's
// S12a row). Nothing here knows about credentials, sessions or navigation —
// it only turns a request into a typed response or a typed `ApiError`.
// §5.3's rules (C1–C14) are dispatched one layer up, in `./dispatch`, against
// the `ApiError` this module throws — kept apart deliberately, so the rules
// can be unit-tested against a plain object with no `fetch` in sight.

/** Which of the two credentials (or neither) a request carried. Not the
 * credential's *value* — `dispatch.ts` only ever needs to know which one. */
export type CredentialKind = 'participant' | 'presenter' | 'none';

/** The storefront's stable error envelope (§5.2/§5.3 C11): `{error, detail?,
 * field?}`, plus whatever extra keys a given route's row adds (`state` on a
 * reset's `504`, `participants` on reset-all's). A response with no JSON body
 * at all — a bare proxy `504` with an HTML body — parses to `null`, which is
 * exactly the shape C4 must still recognise by status alone (§5.3: "the
 * branch keys on the status code, not the error string"). */
export type ApiErrorBody = {
  error?: string;
  detail?: string;
  field?: string;
  [key: string]: unknown;
} | null;

export class ApiError extends Error {
  readonly status: number;
  readonly body: ApiErrorBody;
  readonly method: string;
  readonly path: string;
  readonly credential: CredentialKind;
  /** True for a real `504`, a bodyless proxy-style `504`, and a browser fetch
   * timeout alike — all three take C4's branch (§5.3). */
  readonly isTimeout: boolean;

  constructor(opts: {
    status: number;
    body: ApiErrorBody;
    method: string;
    path: string;
    credential: CredentialKind;
    isTimeout?: boolean;
  }) {
    super(`${opts.method} ${opts.path} -> ${opts.status}${opts.body?.error ? ` ${opts.body.error}` : ''}`);
    this.name = 'ApiError';
    this.status = opts.status;
    this.body = opts.body;
    this.method = opts.method;
    this.path = opts.path;
    this.credential = opts.credential;
    this.isTimeout = opts.isTimeout ?? opts.status === 504;
  }
}

// The one client-facing knob for "how long before we treat this as a browser
// fetch timeout" (§5.3 C4's third shape, alongside a named `504` body and a
// bodyless proxy `504`). Not otherwise part of any plan-stated contract.
export const DEFAULT_REQUEST_TIMEOUT_MS = 20_000;

export interface ApiFetchOptions {
  method: 'GET' | 'POST';
  path: string;
  credential: CredentialKind;
  /** Pre-built `Authorization` header value, e.g. via
   * `participantAuthHeader`/`presenterAuthHeader` — this module does not
   * know how either credential is shaped. */
  authHeader?: string;
  body?: unknown;
  query?: Record<string, string | number | undefined>;
  timeoutMs?: number;
  signal?: AbortSignal;
}

function buildUrl(path: string, query?: Record<string, string | number | undefined>): string {
  if (!query) return path;
  const params = new URLSearchParams();
  for (const [key, value] of Object.entries(query)) {
    if (value !== undefined) params.set(key, String(value));
  }
  const qs = params.toString();
  return qs ? `${path}?${qs}` : path;
}

/** Combines an optional caller-supplied `signal` with a fresh timeout, using
 * a manual `AbortController` rather than `AbortSignal.any`/`.timeout()` — both
 * are broadly supported today, but this keeps the module's only environment
 * assumption at plain `AbortController`, which every target here (evergreen
 * browsers, jsdom under Vitest) implements identically. */
function timedSignal(external: AbortSignal | undefined, timeoutMs: number) {
  const controller = new AbortController();
  let timedOut = false;
  const timer = setTimeout(() => {
    timedOut = true;
    controller.abort();
  }, timeoutMs);
  const onExternalAbort = () => controller.abort();
  external?.addEventListener('abort', onExternalAbort);
  return {
    signal: controller.signal,
    isTimeout: () => timedOut,
    cleanup: () => {
      clearTimeout(timer);
      external?.removeEventListener('abort', onExternalAbort);
    },
  };
}

export async function apiFetch<T>(opts: ApiFetchOptions): Promise<T> {
  const url = buildUrl(opts.path, opts.query);
  const headers: Record<string, string> = {};
  if (opts.body !== undefined) headers['Content-Type'] = 'application/json';
  if (opts.authHeader) headers.Authorization = opts.authHeader;

  const timeoutMs = opts.timeoutMs ?? DEFAULT_REQUEST_TIMEOUT_MS;
  const { signal, isTimeout, cleanup } = timedSignal(opts.signal, timeoutMs);

  let response: Response;
  try {
    response = await fetch(url, {
      method: opts.method,
      headers,
      body: opts.body !== undefined ? JSON.stringify(opts.body) : undefined,
      signal,
    });
  } catch (err) {
    cleanup();
    if (isTimeout()) {
      // §5.3 C4's third shape: "a fetch that times out in the browser takes
      // the same branch" as a named/bodyless `504`.
      throw new ApiError({
        status: 504,
        body: null,
        method: opts.method,
        path: opts.path,
        credential: opts.credential,
        isTimeout: true,
      });
    }
    // An external cancellation (e.g. React Query unmount/refetch supersede) —
    // not a response, not a rule this dispatches on. Let it propagate as-is.
    throw err;
  }
  cleanup();

  if (response.ok) {
    if (response.status === 204) return undefined as T;
    const text = await response.text();
    return (text ? JSON.parse(text) : undefined) as T;
  }

  let body: ApiErrorBody = null;
  try {
    body = (await response.json()) as ApiErrorBody;
  } catch {
    // A bodyless/HTML proxy error (§5.3 C4's second shape) — `body` stays
    // `null`, and the status code alone must still carry the branch.
    body = null;
  }

  throw new ApiError({
    status: response.status,
    body,
    method: opts.method,
    path: opts.path,
    credential: opts.credential,
  });
}
