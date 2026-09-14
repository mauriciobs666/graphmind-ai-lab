// Optimistic-send helpers for `ChatView.tsx`. Kept as plain functions over
// `usePostMessage()`'s own already-computed mutation state (never a second
// source of truth): `POST /shop/api/messages` returns the posted row itself
// (`api/endpoints.ts`'s `postMessage` — `Promise<MessageRow>`), so a
// successful mutation already holds the real, server-assigned row — no
// client-fabricated id is needed. The transcript renders it a poll interval
// early, and stops needing it the moment `GET /shop/api/messages` catches up
// (by `msgId`, so there is never a duplicate).
import type { MessageRow } from '../../api/endpoints';

/** Appends `sent` to `rows` when the mutation just succeeded and the polled
 * transcript has not caught up to it yet. `undefined`/`null` `sent` (no
 * mutation has ever resolved, or one is currently pending) is a no-op. */
export function withOptimisticRow(
  rows: readonly MessageRow[],
  sent: MessageRow | null | undefined,
): MessageRow[] {
  if (!sent) return [...rows];
  if (rows.some((row) => row.msgId === sent.msgId)) return [...rows];
  return [...rows, sent];
}
