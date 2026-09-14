import { describe, expect, it } from 'vitest';
import type { MessageRow } from '../../api/endpoints';
import { WELCOME_ROW_ID, withWelcomeRow } from './welcome';

function row(msgId: string, text = 'hi'): MessageRow {
  return { msgId, threadId: 't', authorId: 'p', text, role: 'user', createdAt: 1, mentions: [] };
}

describe('withWelcomeRow', () => {
  it('prepends the welcome line as an assistant-styled row when present', () => {
    const result = withWelcomeRow([row('1')], 'Welcome to the store, Ada.');
    expect(result.map((r) => r.msgId)).toEqual([WELCOME_ROW_ID, '1']);
    expect(result[0]).toMatchObject({ text: 'Welcome to the store, Ada.', role: 'assistant' });
  });

  it('is a no-op with no welcome line (null)', () => {
    expect(withWelcomeRow([row('1')], null)).toEqual([row('1')]);
  });

  it('is a no-op with an empty-string welcome line', () => {
    expect(withWelcomeRow([row('1')], '')).toEqual([row('1')]);
  });

  it('returns a fresh array, never the same reference as the input', () => {
    const input = [row('1')];
    expect(withWelcomeRow(input, null)).not.toBe(input);
  });

  it('prepends onto an otherwise-empty transcript', () => {
    const result = withWelcomeRow([], 'Bem-vindo à loja, Ada.');
    expect(result).toHaveLength(1);
    expect(result[0].msgId).toBe(WELCOME_ROW_ID);
  });
});
