// §4.12 (v1.37) — the welcome turn. Kept as a plain function, same reason as
// `optimistic.ts`'s `withOptimisticRow`: cheap to mutation-test with no
// rendered tree in the loop. Prepends the join response's one-shot `welcome`
// line (`session/SessionContext.tsx`'s `welcomeMessage`, captured once by
// `ChatView.tsx`) as the transcript's oldest entry, styled like an
// assistant turn (`role: 'assistant'`) since that is how it reads — the
// store greeting the participant, before they have sent anything.
import type { MessageRow } from '../../api/endpoints';

export const WELCOME_ROW_ID = '__welcome__';

export function withWelcomeRow(
  rows: readonly MessageRow[],
  welcome: string | null,
): MessageRow[] {
  if (!welcome) return [...rows];
  const welcomeRow: MessageRow = {
    msgId: WELCOME_ROW_ID,
    threadId: '',
    authorId: 'agent',
    text: welcome,
    role: 'assistant',
    createdAt: 0,
    mentions: [],
  };
  return [welcomeRow, ...rows];
}
