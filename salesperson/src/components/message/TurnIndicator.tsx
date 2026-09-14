// The thinking/queued indicator, driven entirely by `turn` (docs/plans/
// salesperson-ui.md §5.2 *The queue position*). `queuePosition` is
// **always present** on the wire and is `0` exactly when the participant is
// first in line — never "no queue": a queued turn with `queuePosition: 0`
// must render as *first in line*, so the branch below tests `=== 0`
// explicitly rather than treating the number as a truthy/falsy flag (the
// mutation `docs/plans/salesperson-ui-impl.md`-style bug this guards
// against is `turn.queuePosition && <ahead-of-you text>`, which renders
// nothing at all when the position is `0`).
import { useTranslation } from 'react-i18next';
import type { TurnBlock } from '../../api/endpoints';

export function TurnIndicator({ turn }: { turn: TurnBlock | undefined }) {
  const { t } = useTranslation();
  if (!turn || turn.state === 'idle') return null;

  if (turn.state === 'thinking') {
    return (
      <p
        role="status"
        className="flex items-center gap-2 px-1 text-xs text-slate-500 dark:text-slate-400"
      >
        <span className="flex gap-0.5" aria-hidden="true">
          <span className="h-1.5 w-1.5 animate-bounce rounded-full bg-slate-400 [animation-delay:-0.3s] motion-reduce:animate-none dark:bg-slate-500" />
          <span className="h-1.5 w-1.5 animate-bounce rounded-full bg-slate-400 [animation-delay:-0.15s] motion-reduce:animate-none dark:bg-slate-500" />
          <span className="h-1.5 w-1.5 animate-bounce rounded-full bg-slate-400 motion-reduce:animate-none dark:bg-slate-500" />
        </span>
        {t('chat.turn.thinking')}
      </p>
    );
  }

  // turn.state === 'queued'
  const text =
    turn.queuePosition === 0
      ? t('chat.turn.firstInLine')
      : t('chat.turn.aheadOfYou', { count: turn.queuePosition });

  return (
    <p role="status" className="px-1 text-xs text-slate-500 dark:text-slate-400">
      {text}
    </p>
  );
}
