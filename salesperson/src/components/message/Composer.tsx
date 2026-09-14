// The message input + send control. Purely presentational — text ownership,
// the submit handler and the disabled/notice computation all live in
// `../../views/ChatView.tsx`, so this component never itself decides C6a's
// "retain the text on a 409" rule; it only renders whatever `ChatView` hands
// it. That split is what keeps this file free of any hook, and therefore
// trivial to render with a handful of props in a test.
import type { FormEvent } from 'react';
import { useTranslation } from 'react-i18next';
import type { ComposerNotice } from './composerNotice';

const MAX_LENGTH = 2000; // §5.2 `POST /shop/api/messages` bound (C11)

const NOTICE_STYLES: Record<ComposerNotice['tone'], string> = {
  info: 'bg-slate-100 text-slate-600 dark:bg-slate-800 dark:text-slate-300',
  warning: 'bg-amber-100 text-amber-900 dark:bg-amber-950/40 dark:text-amber-200',
  error: 'bg-red-100 text-red-800 dark:bg-red-950/40 dark:text-red-200',
};

export function Composer({
  value,
  onChange,
  onSubmit,
  disabled,
  notice,
}: {
  value: string;
  onChange: (value: string) => void;
  onSubmit: (event: FormEvent) => void;
  disabled: boolean;
  notice: ComposerNotice | null;
}) {
  const { t } = useTranslation();
  return (
    <div
      // `sticky bottom-0` is defensive: it keeps the composer reachable at
      // the viewport's bottom edge even in a browser/ancestor combination
      // where `Transcript`'s own `overflow-y-auto` does not end up bounded
      // (a nested-flex `min-height` gotcha whose fix, if needed, is
      // `layout/Shell.tsx`'s `flex-1` wrapper — out of this file's reach) and
      // the page scrolls as a whole instead of just the transcript.
      className="sticky bottom-0 z-20 border-t border-slate-200/80 bg-white/90 pb-[env(safe-area-inset-bottom)] backdrop-blur supports-[backdrop-filter]:bg-white/70 dark:border-slate-800/80 dark:bg-slate-950/90 dark:supports-[backdrop-filter]:bg-slate-950/70"
    >
      {notice && (
        <p
          role={notice.tone === 'error' ? 'alert' : 'status'}
          className={`mx-3 mt-2 rounded-md px-3 py-2 text-sm ${NOTICE_STYLES[notice.tone]}`}
        >
          {notice.text}
        </p>
      )}
      <form onSubmit={onSubmit} className="flex items-end gap-2 px-3 py-3">
        <label className="sr-only" htmlFor="chat-composer-input">
          {t('chat.composer.messageLabel')}
        </label>
        <textarea
          id="chat-composer-input"
          value={value}
          onChange={(event) => onChange(event.target.value)}
          onKeyDown={(event) => {
            if (event.key === 'Enter' && !event.shiftKey) {
              event.preventDefault();
              (event.currentTarget.form as HTMLFormElement | null)?.requestSubmit();
            }
          }}
          maxLength={MAX_LENGTH}
          rows={1}
          placeholder={t('chat.composer.placeholder')}
          disabled={disabled}
          className="max-h-32 min-h-11 flex-1 resize-none rounded-2xl border border-slate-300 px-4 py-2.5 text-base leading-snug outline-none focus:border-slate-500 disabled:cursor-not-allowed disabled:opacity-60 dark:border-slate-600 dark:bg-slate-900"
        />
        <button
          type="submit"
          disabled={disabled || value.trim().length === 0}
          aria-label={t('chat.composer.send')}
          className="flex h-11 w-11 shrink-0 items-center justify-center rounded-full bg-slate-900 text-white transition disabled:cursor-not-allowed disabled:opacity-40 dark:bg-slate-100 dark:text-slate-900"
        >
          <svg
            viewBox="0 0 24 24"
            fill="none"
            stroke="currentColor"
            strokeWidth={1.75}
            strokeLinecap="round"
            strokeLinejoin="round"
            className="h-5 w-5"
            aria-hidden="true"
          >
            <path d="M4 12h15M13 6l6 6-6 6" />
          </svg>
        </button>
      </form>
    </div>
  );
}
