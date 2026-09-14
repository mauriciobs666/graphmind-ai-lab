// One transcript row. `{row.text}` is a JSX text child — React always
// renders it as a text node (`Node.textContent`, never `innerHTML`), which is
// what satisfies `salesperson/AGENTS.md`'s hard constraint #4 and
// docs/plans/salesperson-ui.md §4.2's `textContent`-only rule: agent-emitted
// markup (e.g. a literal `<b>` in the reply) is never parsed as HTML here,
// with no `dangerouslySetInnerHTML` anywhere in this tree.
import { useLocale } from '../../i18n/useLocale';
import { formatDate } from '../../i18n/format';
import type { MessageRow } from '../../api/endpoints';

export function MessageBubble({
  row,
  pending = false,
  showTimestamp = true,
}: {
  row: Pick<MessageRow, 'msgId' | 'text' | 'role' | 'createdAt'>;
  /** The optimistic "sending" echo — not yet confirmed by the server. */
  pending?: boolean;
  /** The welcome turn (§4.12) carries no real `createdAt` (it is minted
   * client-side from the join response, not a transcript row with a wire
   * timestamp) — `false` there, so the caption doesn't show a misleading
   * clock time for "epoch 0". */
  showTimestamp?: boolean;
}) {
  const { locale } = useLocale();
  const isOwn = row.role !== 'assistant';

  return (
    <li
      data-testid="message-row"
      data-pending={pending || undefined}
      className={`flex ${isOwn ? 'justify-end' : 'justify-start'}`}
    >
      <div
        className={`max-w-[85%] rounded-2xl px-3 py-2 text-sm shadow-sm ${
          isOwn
            ? 'rounded-br-sm bg-slate-900 text-white dark:bg-slate-100 dark:text-slate-900'
            : 'rounded-bl-sm bg-white text-slate-800 ring-1 ring-slate-200 dark:bg-slate-800 dark:text-slate-100 dark:ring-slate-700'
        } ${pending ? 'opacity-60' : ''}`}
      >
        <p className="whitespace-pre-wrap break-words">{row.text}</p>
        {(pending || showTimestamp) && (
          <p
            className={`mt-1 text-[11px] ${
              isOwn ? 'text-slate-300 dark:text-slate-600' : 'text-slate-400 dark:text-slate-500'
            }`}
          >
            {pending ? 'Sending…' : formatDate(row.createdAt, locale, { timeStyle: 'short' })}
          </p>
        )}
      </div>
    </li>
  );
}
