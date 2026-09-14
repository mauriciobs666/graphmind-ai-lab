// The scrollable message list — autoscroll-when-at-bottom, mirroring
// `falkor-chat/web/app.js`'s `pollMessages`/`renderMessages`: a 40px
// tolerance around the bottom edge, and a new row only pulls the view down
// when the viewer was already there (someone scrolled up to re-read history
// is never yanked back down by an incoming reply).
//
// Whether the viewer is "at the bottom" is tracked continuously via a
// `scroll` listener into a ref (not recomputed from the post-update DOM,
// which by the time any effect can read it already reflects the *new*
// `scrollHeight` — too late to tell "was" from "is now"), so the decision at
// append time is always the viewer's true position immediately beforehand.
import { useEffect, useRef } from 'react';
import { useTranslation } from 'react-i18next';
import type { MessageRow } from '../../api/endpoints';
import { MessageBubble } from './MessageBubble';
import { WELCOME_ROW_ID } from './welcome';

const AT_BOTTOM_PX = 40;

export function Transcript({
  rows,
  pendingText,
}: {
  rows: readonly MessageRow[];
  /** The composer's own text while a send is in flight — rendered as a
   * trailing, unconfirmed echo (optimistic send). */
  pendingText?: string | null;
}) {
  const { t } = useTranslation();
  const containerRef = useRef<HTMLUListElement | null>(null);
  const atBottomRef = useRef(true);

  useEffect(() => {
    const el = containerRef.current;
    if (!el) return;
    function handleScroll() {
      if (!el) return;
      atBottomRef.current = el.scrollHeight - el.scrollTop - el.clientHeight < AT_BOTTOM_PX;
    }
    el.addEventListener('scroll', handleScroll);
    return () => el.removeEventListener('scroll', handleScroll);
  }, []);

  useEffect(() => {
    const el = containerRef.current;
    if (el && atBottomRef.current) {
      el.scrollTop = el.scrollHeight;
    }
  }, [rows.length, pendingText]);

  const isEmpty = rows.length === 0 && !pendingText;

  return (
    <ul
      ref={containerRef}
      aria-label={t('chat.transcript.label')}
      aria-live="polite"
      className="flex min-h-0 flex-1 flex-col gap-2 overflow-y-auto px-3 py-3"
    >
      {isEmpty && (
        <li className="px-2 py-8 text-center text-sm text-slate-400 dark:text-slate-500">
          {t('chat.transcript.empty')}
        </li>
      )}
      {rows.map((row) => (
        <MessageBubble
          key={row.msgId}
          row={row}
          // §4.12 — the welcome turn carries no real wire `createdAt`.
          showTimestamp={row.msgId !== WELCOME_ROW_ID}
        />
      ))}
      {pendingText && (
        <MessageBubble
          key="__pending__"
          // `createdAt` is never read while `pending` — `MessageBubble`
          // shows "Sending…" instead of a formatted time — so this is a
          // fixed placeholder, not `Date.now()` (an impure call during
          // render, and a real timestamp would be misleading anyway: it
          // isn't the moment the server accepts the message).
          row={{ msgId: '__pending__', text: pendingText, role: 'user', createdAt: 0 }}
          pending
        />
      )}
    </ul>
  );
}
