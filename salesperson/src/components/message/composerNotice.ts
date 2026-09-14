// Pure classification of the composer's inline banner from a `POST
// /shop/api/messages` mutation's own `action`/`reconciliation` (both already
// computed by `api/hooks.ts`'s `usePostMessage()` — this file performs no
// dispatch of its own, only picks the copy). Kept apart from `ChatView.tsx`
// for the same reason `api/dispatch.ts` is kept apart from the hooks that
// consume it (docs/plans/salesperson-ui.md §5.1's S12a row's own comment):
// a plain function is what makes the mutation-testing requirement (break the
// rule, watch the test go red) cheap to run, with no fetch/DOM/React tree in
// the loop.
import type { TFunction } from 'i18next';
import type { ErrorAction } from '../../api/dispatch';
import type { PostMessageReconciliation } from '../../api/hooks';

export type ComposerNoticeTone = 'info' | 'warning' | 'error';

export interface ComposerNotice {
  tone: ComposerNoticeTone;
  text: string;
}

/** C6a's first half — a `409 TurnInProgress` is routine, not a failure: the
 * composer text is retained and re-enables on its own once `turn.state`
 * returns to `idle` (`ChatView.tsx` drives `disabled` from `turn.state`
 * directly). This banner is informational only. */
export function composerNoticeFor(
  action: ErrorAction | null,
  reconciliation: PostMessageReconciliation | null,
  t: TFunction,
): ComposerNotice | null {
  if (!action) return null;

  switch (action.kind) {
    case 'turnInProgressRetain':
      return { tone: 'info', text: t('chat.notice.turnInProgress') };

    case 'fieldError':
      // C11 — dispatched on the field; a `dev` audience value (never
      // user-supplied) is a defence-in-depth branch, not shopper copy.
      return action.audience === 'user'
        ? { tone: 'error', text: t('chat.notice.messageTooLong') }
        : null;

    case 'nothingChangedRetry':
      // C9 — the 503 default: nothing was written, safe to retry.
      return {
        tone: 'warning',
        text: t('chat.notice.sendFailedRetry'),
      };

    case 'reread':
      // C4 — a 504 on the write: ambiguous, resolved by the hook's own
      // re-read + `reconcilePostMessageFailure`.
      switch (reconciliation) {
        case 'nothingCommitted':
          return { tone: 'warning', text: t('chat.notice.messageNotSent') };
        case 'turnRunning':
          return { tone: 'info', text: t('chat.notice.sentAwaitingReply') };
        case 'turnLost':
          return {
            tone: 'warning',
            text: t('chat.notice.replyUnconfirmed'),
          };
        default:
          return { tone: 'warning', text: t('chat.notice.checkingDelivery') };
      }

    case 'deadTurn':
      // C14 — the one 503 that is not "nothing changed": the message was
      // written, but no reply will be generated for it.
      return {
        tone: 'warning',
        text: t('chat.notice.messageSentNoReply'),
      };

    case 'unscopedAlarm':
      // C6b — an alarm; never rendered as busy or success.
      return {
        tone: 'error',
        text: t('chat.notice.sessionUnscoped'),
      };

    case 'clearParticipant':
      // C3 — the credential is already cleared and the client is navigating
      // to join; nothing to show inline.
      return null;

    case 'unhandled':
      // C13 — loud, never a silent generic fallback.
      return {
        tone: 'error',
        text: t('chat.notice.unexpectedStatus', { status: action.status }),
      };

    default:
      return null;
  }
}
