import { describe, expect, it } from 'vitest';
import type { ErrorAction } from '../../api/dispatch';
import { composerNoticeFor } from './composerNotice';

describe('composerNoticeFor', () => {
  it('returns null with no action', () => {
    expect(composerNoticeFor(null, null)).toBeNull();
  });

  it('C6a — turnInProgressRetain is informational', () => {
    const notice = composerNoticeFor({ kind: 'turnInProgressRetain' }, null);
    expect(notice).toEqual({ tone: 'info', text: 'Still working on your last message…' });
  });

  it('C11 — fieldError for the user-supplied `text` field is surfaced', () => {
    const action: ErrorAction = { kind: 'fieldError', field: 'text', audience: 'user' };
    expect(composerNoticeFor(action, null)?.tone).toBe('error');
  });

  it('C11 — a dev-audience fieldError is not shown to the shopper', () => {
    const action: ErrorAction = { kind: 'fieldError', field: 'mentions', audience: 'dev' };
    expect(composerNoticeFor(action, null)).toBeNull();
  });

  it('C9 — nothingChangedRetry says nothing was sent', () => {
    const notice = composerNoticeFor({ kind: 'nothingChangedRetry', scope: 'user' }, null);
    expect(notice?.tone).toBe('warning');
    expect(notice?.text).toMatch(/not sent/i);
  });

  it('C4 — reread + nothingCommitted reads as safe to retry', () => {
    const action: ErrorAction = { kind: 'reread', via: { endpoint: 'messagesAndState' } };
    const notice = composerNoticeFor(action, 'nothingCommitted');
    expect(notice?.text).toMatch(/not sent/i);
  });

  it('C4 — reread + turnRunning reads as already sent, reply on its way', () => {
    const action: ErrorAction = { kind: 'reread', via: { endpoint: 'messagesAndState' } };
    const notice = composerNoticeFor(action, 'turnRunning');
    expect(notice?.tone).toBe('info');
    expect(notice?.text).toMatch(/on its way/i);
  });

  it('C4 — reread + turnLost warns that resending duplicates the line', () => {
    const action: ErrorAction = { kind: 'reread', via: { endpoint: 'messagesAndState' } };
    const notice = composerNoticeFor(action, 'turnLost');
    expect(notice?.text).toMatch(/new line/i);
  });

  // C4 — `reconciliation` is `usePostMessage()`'s own state, initialised to
  // `null` and only ever set once the reread's `getMessages`/`getState`
  // round trip resolves; `action.kind === 'reread'` is set synchronously,
  // one render ahead of that. `null` is therefore the actual, reachable
  // in-between value on a `reread` action — not a placeholder for "no other
  // case matched" — confirmed live (`falkor-chat/docs/reviews/
  // salesperson-ui-s13.md`'s Major: a gated-promise `ChatView` reproduction
  // showed this copy rendering in that window on every `504`-triggered
  // reread). `ChatView.test.tsx`'s own gated-reread case exercises the same
  // window end to end; this one pins the classification in isolation.
  it('C4 — reread + null (the reachable in-between value, before the reread resolves) reads as still checking', () => {
    const action: ErrorAction = { kind: 'reread', via: { endpoint: 'messagesAndState' } };
    const notice = composerNoticeFor(action, null);
    expect(notice?.tone).toBe('warning');
    expect(notice?.text).toMatch(/checking/i);
  });

  it('C14 — deadTurn says the message sent but no reply will come', () => {
    const notice = composerNoticeFor({ kind: 'deadTurn' }, null);
    expect(notice?.text).toMatch(/no reply will be generated/i);
  });

  it('C6b — unscopedAlarm reads as a failure, not busy or success', () => {
    const notice = composerNoticeFor({ kind: 'unscopedAlarm' }, null);
    expect(notice?.tone).toBe('error');
  });

  it('C3 — clearParticipant has nothing to show inline (navigating away)', () => {
    expect(composerNoticeFor({ kind: 'clearParticipant' }, null)).toBeNull();
  });

  it('C13 — unhandled is loud, carrying the status code', () => {
    const action: ErrorAction = { kind: 'unhandled', route: 'messagesPost', status: 418 };
    const notice = composerNoticeFor(action, null);
    expect(notice?.text).toContain('418');
  });
});
