import { afterEach, describe, expect, it } from 'vitest';
import type { ErrorAction } from '../../api/dispatch';
import i18n from '../../i18n/config';
import { composerNoticeFor } from './composerNotice';

// §4.13 (v1.39) — `composerNoticeFor` now takes a threaded `t: TFunction`
// third parameter instead of hardcoded English literals (the file stays
// outside the React tree on purpose — see this file's own top comment on
// mutation-testing cheapness — so tests call it directly with a real `t`
// rather than mounting a component). `i18n.t` from the shared instance is
// used directly: no DOM/React tree needed, i18next itself is plain JS.
// `.bind(i18n)` matters — `i18n.t` reads `this.translator` internally, so an
// unbound reference throws the moment it's called standalone.
const t = i18n.t.bind(i18n);

afterEach(async () => {
  await i18n.changeLanguage('en');
});

describe('composerNoticeFor', () => {
  it('returns null with no action', () => {
    expect(composerNoticeFor(null, null, t)).toBeNull();
  });

  it('C6a — turnInProgressRetain is informational', () => {
    const notice = composerNoticeFor({ kind: 'turnInProgressRetain' }, null, t);
    expect(notice).toEqual({ tone: 'info', text: 'Still working on your last message…' });
  });

  it('C11 — fieldError for the user-supplied `text` field is surfaced', () => {
    const action: ErrorAction = { kind: 'fieldError', field: 'text', audience: 'user' };
    expect(composerNoticeFor(action, null, t)?.tone).toBe('error');
  });

  it('C11 — a dev-audience fieldError is not shown to the shopper', () => {
    const action: ErrorAction = { kind: 'fieldError', field: 'mentions', audience: 'dev' };
    expect(composerNoticeFor(action, null, t)).toBeNull();
  });

  it('C9 — nothingChangedRetry says nothing was sent', () => {
    const notice = composerNoticeFor({ kind: 'nothingChangedRetry', scope: 'user' }, null, t);
    expect(notice?.tone).toBe('warning');
    expect(notice?.text).toMatch(/not sent/i);
  });

  it('C4 — reread + nothingCommitted reads as safe to retry', () => {
    const action: ErrorAction = { kind: 'reread', via: { endpoint: 'messagesAndState' } };
    const notice = composerNoticeFor(action, 'nothingCommitted', t);
    expect(notice?.text).toMatch(/not sent/i);
  });

  it('C4 — reread + turnRunning reads as already sent, reply on its way', () => {
    const action: ErrorAction = { kind: 'reread', via: { endpoint: 'messagesAndState' } };
    const notice = composerNoticeFor(action, 'turnRunning', t);
    expect(notice?.tone).toBe('info');
    expect(notice?.text).toMatch(/on its way/i);
  });

  it('C4 — reread + turnLost warns that resending duplicates the line', () => {
    const action: ErrorAction = { kind: 'reread', via: { endpoint: 'messagesAndState' } };
    const notice = composerNoticeFor(action, 'turnLost', t);
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
    const notice = composerNoticeFor(action, null, t);
    expect(notice?.tone).toBe('warning');
    expect(notice?.text).toMatch(/checking/i);
  });

  it('C14 — deadTurn says the message sent but no reply will come', () => {
    const notice = composerNoticeFor({ kind: 'deadTurn' }, null, t);
    expect(notice?.text).toMatch(/no reply will be generated/i);
  });

  it('C6b — unscopedAlarm reads as a failure, not busy or success', () => {
    const notice = composerNoticeFor({ kind: 'unscopedAlarm' }, null, t);
    expect(notice?.tone).toBe('error');
  });

  it('C3 — clearParticipant has nothing to show inline (navigating away)', () => {
    expect(composerNoticeFor({ kind: 'clearParticipant' }, null, t)).toBeNull();
  });

  it('C13 — unhandled is loud, carrying the status code', () => {
    const action: ErrorAction = { kind: 'unhandled', route: 'messagesPost', status: 418 };
    const notice = composerNoticeFor(action, null, t);
    expect(notice?.text).toContain('418');
  });

  // §4.13's Layer 2, applied here rather than via a mounted component (this
  // file stays a plain-function contract): every branch's copy actually
  // comes from the active bundle, not a hardcoded literal — switching
  // locale changes the text, and the English literal disappears.
  it('routes every branch through the active i18next bundle — switches to pt-BR, English literals gone', async () => {
    await i18n.changeLanguage('pt-BR');
    const ptT = i18n.t.bind(i18n);

    expect(composerNoticeFor({ kind: 'turnInProgressRetain' }, null, ptT)?.text).toBe(
      'Ainda processando sua última mensagem…',
    );
    expect(
      composerNoticeFor({ kind: 'fieldError', field: 'text', audience: 'user' }, null, ptT)?.text,
    ).toBe('Essa mensagem é muito longa. Encurte-a e envie novamente.');
    expect(
      composerNoticeFor({ kind: 'nothingChangedRetry', scope: 'user' }, null, ptT)?.text,
    ).toBe('Algo deu errado e sua mensagem não foi enviada. Tente novamente.');
    const rereadAction: ErrorAction = { kind: 'reread', via: { endpoint: 'messagesAndState' } };
    expect(composerNoticeFor(rereadAction, 'nothingCommitted', ptT)?.text).toBe(
      'Sua mensagem não foi enviada. Tente novamente.',
    );
    expect(composerNoticeFor(rereadAction, 'turnRunning', ptT)?.text).toBe(
      'Sua mensagem foi enviada — uma resposta está a caminho.',
    );
    expect(composerNoticeFor(rereadAction, 'turnLost', ptT)?.text).toBe(
      'Não conseguimos confirmar se uma resposta chegou. Enviar novamente adicionará uma nova linha ao chat.',
    );
    expect(composerNoticeFor(rereadAction, null, ptT)?.text).toBe(
      'Não conseguimos confirmar se sua mensagem foi entregue — verificando…',
    );
    expect(composerNoticeFor({ kind: 'deadTurn' }, null, ptT)?.text).toBe(
      'Sua mensagem foi enviada, mas nenhuma resposta será gerada agora. Envie-a novamente em breve.',
    );
    expect(composerNoticeFor({ kind: 'unscopedAlarm' }, null, ptT)?.text).toBe(
      'Sua sessão não está mais vinculada a esta loja. Recarregue a página para continuar.',
    );
    expect(
      composerNoticeFor({ kind: 'unhandled', route: 'messagesPost', status: 418 }, null, ptT)
        ?.text,
    ).toBe('Resposta inesperada do servidor (status 418). Tente novamente.');
  });
});
