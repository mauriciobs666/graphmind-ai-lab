import { describe, expect, it } from 'vitest';
import type { MessageRow } from '../../api/endpoints';
import { withOptimisticRow } from './optimistic';

function row(msgId: string, text = 'hi'): MessageRow {
  return { msgId, threadId: 't', authorId: 'p', text, role: 'user', createdAt: 1, mentions: [] };
}

describe('withOptimisticRow', () => {
  it('appends the sent row when it is not yet in the polled list', () => {
    const result = withOptimisticRow([row('1')], row('2', 'new'));
    expect(result.map((r) => r.msgId)).toEqual(['1', '2']);
  });

  it('does not duplicate once the polled list already contains it (by msgId)', () => {
    const result = withOptimisticRow([row('1'), row('2', 'new')], row('2', 'new'));
    expect(result.map((r) => r.msgId)).toEqual(['1', '2']);
  });

  it('is a no-op with no sent row (nothing pending/successful yet)', () => {
    expect(withOptimisticRow([row('1')], null)).toEqual([row('1')]);
    expect(withOptimisticRow([row('1')], undefined)).toEqual([row('1')]);
  });

  it('returns a fresh array, never the same reference as the input', () => {
    const input = [row('1')];
    expect(withOptimisticRow(input, null)).not.toBe(input);
  });
});
