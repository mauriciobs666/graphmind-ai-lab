// The route shell (docs/plans/salesperson-ui.md §5.1's S12a row: "route
// shell for join / chat / presenter"). The screens below are deliberately
// minimal — enough to prove the join -> chat round trip end to end and to
// exercise every hook in `./api/hooks` — not the polished, styled surface
// the plan's later steps own: S13 replaces the chat surface with
// `src/views/Chat*` + `src/components/message/**`, S14 adds
// `src/views/{Cart,Order,Profile,Catalog}*`, and S12d adds
// `src/views/Presenter*`. Session/API wiring, and everything not owned by
// one of those three, stays this file's job (§5.0's shared-file map).
import { type FormEvent, useState } from 'react';
import { Navigate, createBrowserRouter } from 'react-router-dom';
import {
  useHealth,
  useJoin,
  useMessages,
  usePostMessage,
  usePresenterLogin,
  usePresenterParticipants,
  useResetMine,
  useShopState,
} from './api/hooks';
import { APP_PATHS } from './routePaths';
import { useSession } from './session/SessionContext';

function JoinScreen() {
  const { pendingLanguageStep } = useSession();
  const health = useHealth();
  const join = useJoin();
  const locales = health.data?.locales ?? ['en'];
  const isLanguageStep = pendingLanguageStep !== null;

  const [displayName, setDisplayName] = useState('');
  const [language, setLanguage] = useState(pendingLanguageStep ?? locales[0] ?? 'en');

  function handleSubmit(event: FormEvent) {
    event.preventDefault();
    join.mutate({ displayName: displayName.trim(), language });
  }

  const fieldError =
    join.action?.kind === 'fieldError' && join.action.audience === 'user' ? join.action : null;
  const transientFailure = join.action?.kind === 'nothingChangedRetry' ? join.action : null;
  const lostJoin = join.action?.kind === 'reread' && join.action.via.endpoint === 'none';

  return (
    <main className="mx-auto flex min-h-svh max-w-md flex-col justify-center gap-6 px-6 py-10">
      <div className="text-center">
        <h1 className="text-2xl font-semibold text-slate-900 dark:text-slate-50">
          {isLanguageStep ? 'Choose your language' : 'Welcome to the store'}
        </h1>
        {!isLanguageStep && (
          <p className="mt-1 text-sm text-slate-500 dark:text-slate-400">
            Tell us your name to start chatting.
          </p>
        )}
      </div>

      {lostJoin && (
        <p role="alert" className="rounded-md bg-amber-100 px-3 py-2 text-sm text-amber-900">
          Your join may not have completed — please join again.
        </p>
      )}
      {transientFailure && (
        <p role="alert" className="rounded-md bg-amber-100 px-3 py-2 text-sm text-amber-900">
          Something went wrong and nothing was saved. Please try again.
        </p>
      )}

      <form onSubmit={handleSubmit} className="flex flex-col gap-4">
        {!isLanguageStep && (
          <label className="flex flex-col gap-1 text-sm text-slate-700 dark:text-slate-200">
            Your name
            <input
              className="rounded-md border border-slate-300 px-3 py-2 text-base outline-none focus:border-slate-500 dark:border-slate-600 dark:bg-slate-900"
              value={displayName}
              onChange={(event) => setDisplayName(event.target.value)}
              maxLength={60}
              required
            />
            {fieldError?.field === 'displayName' && (
              <span className="text-xs text-red-600">Please enter a name.</span>
            )}
          </label>
        )}

        <label className="flex flex-col gap-1 text-sm text-slate-700 dark:text-slate-200">
          Language
          <select
            className="rounded-md border border-slate-300 px-3 py-2 text-base outline-none focus:border-slate-500 dark:border-slate-600 dark:bg-slate-900"
            value={language}
            onChange={(event) => setLanguage(event.target.value)}
          >
            {locales.map((locale) => (
              <option key={locale} value={locale}>
                {locale}
              </option>
            ))}
          </select>
        </label>

        <button
          type="submit"
          disabled={join.isPending || (!isLanguageStep && displayName.trim().length === 0)}
          className="rounded-md bg-slate-900 px-4 py-2 text-sm font-medium text-white transition disabled:cursor-not-allowed disabled:opacity-50 dark:bg-slate-100 dark:text-slate-900"
        >
          {join.isPending ? 'Joining…' : isLanguageStep ? 'Continue' : 'Join'}
        </button>
      </form>
    </main>
  );
}

function ChatScreen() {
  const { participant } = useSession();
  const shopState = useShopState();
  const messages = useMessages();
  const postMessage = usePostMessage();
  const resetMine = useResetMine();
  const [text, setText] = useState('');

  const turnState = shopState.data?.turn.state ?? 'idle';
  const composerDisabled = turnState !== 'idle' || postMessage.isPending;

  function handleSubmit(event: FormEvent) {
    event.preventDefault();
    const trimmed = text.trim();
    if (!trimmed) return;
    postMessage.mutate(trimmed, {
      onSuccess: () => setText(''),
      // C6a — a 409 retains the composer text; do not clear it here.
    });
  }

  return (
    <main className="mx-auto flex min-h-svh max-w-lg flex-col gap-4 px-4 py-6">
      <header className="flex items-center justify-between">
        <div>
          <p className="text-sm text-slate-500 dark:text-slate-400">Signed in as</p>
          <p className="font-medium text-slate-900 dark:text-slate-50">
            {participant?.displayName}
          </p>
        </div>
        <button
          type="button"
          onClick={() => {
            if (window.confirm('Reset your session? This clears your cart and chat.')) {
              resetMine.mutate();
            }
          }}
          className="rounded-md border border-slate-300 px-3 py-1.5 text-xs text-slate-600 dark:border-slate-600 dark:text-slate-300"
        >
          Reset
        </button>
      </header>

      {shopState.deadTurnNotice && (
        <p role="status" className="rounded-md bg-amber-100 px-3 py-2 text-sm text-amber-900">
          The reply never arrived — send again.
        </p>
      )}
      {postMessage.action?.kind === 'turnInProgressRetain' && (
        <p role="status" className="text-xs text-slate-500">
          Still working on the last message…
        </p>
      )}

      <ul className="flex-1 space-y-2 overflow-y-auto" aria-label="Transcript">
        {(messages.data ?? []).map((row) => (
          <li
            key={row.msgId}
            className="rounded-md bg-slate-100 px-3 py-2 text-sm text-slate-800 dark:bg-slate-800 dark:text-slate-100"
          >
            {row.text}
          </li>
        ))}
      </ul>

      <form onSubmit={handleSubmit} className="flex gap-2">
        <input
          className="flex-1 rounded-md border border-slate-300 px-3 py-2 text-base outline-none focus:border-slate-500 dark:border-slate-600 dark:bg-slate-900"
          value={text}
          onChange={(event) => setText(event.target.value)}
          maxLength={2000}
          placeholder="Type a message…"
          disabled={composerDisabled}
        />
        <button
          type="submit"
          disabled={composerDisabled || text.trim().length === 0}
          className="rounded-md bg-slate-900 px-4 py-2 text-sm font-medium text-white disabled:cursor-not-allowed disabled:opacity-50 dark:bg-slate-100 dark:text-slate-900"
        >
          Send
        </button>
      </form>
    </main>
  );
}

function ParticipantRoute() {
  const { participant, pendingLanguageStep } = useSession();
  if (!participant || pendingLanguageStep !== null) {
    return <JoinScreen />;
  }
  return <ChatScreen />;
}

function PresenterKeyScreen() {
  const login = usePresenterLogin();
  const [key, setKey] = useState('');
  const keyRejected = login.action?.kind === 'presenterKeyRejected';

  function handleSubmit(event: FormEvent) {
    event.preventDefault();
    login.mutate(key);
  }

  return (
    <main className="mx-auto flex min-h-svh max-w-sm flex-col justify-center gap-4 px-6">
      <h1 className="text-xl font-semibold text-slate-900 dark:text-slate-50">Presenter key</h1>
      <form onSubmit={handleSubmit} className="flex flex-col gap-3">
        <input
          type="password"
          className="rounded-md border border-slate-300 px-3 py-2 text-base outline-none focus:border-slate-500 dark:border-slate-600 dark:bg-slate-900"
          value={key}
          onChange={(event) => setKey(event.target.value)}
          required
        />
        {keyRejected && <p className="text-xs text-red-600">That key was not accepted.</p>}
        <button
          type="submit"
          disabled={login.isPending}
          className="rounded-md bg-slate-900 px-4 py-2 text-sm font-medium text-white disabled:opacity-50 dark:bg-slate-100 dark:text-slate-900"
        >
          {login.isPending ? 'Checking…' : 'Enter'}
        </button>
      </form>
    </main>
  );
}

function PresenterRoster() {
  const roster = usePresenterParticipants();
  return (
    <main className="mx-auto max-w-2xl px-4 py-6">
      <h1 className="text-xl font-semibold text-slate-900 dark:text-slate-50">Participants</h1>
      {roster.data && roster.data.length === 0 && (
        <p className="mt-4 text-sm text-slate-500">No one has joined yet.</p>
      )}
      <ul className="mt-4 space-y-2">
        {(roster.data ?? []).map((row) => (
          <li
            key={row.participantId}
            className="rounded-md border border-slate-200 px-3 py-2 text-sm dark:border-slate-700"
          >
            {row.displayName} · {row.language}
          </li>
        ))}
      </ul>
    </main>
  );
}

function PresenterRoute() {
  const { presenter } = useSession();
  return presenter ? <PresenterRoster /> : <PresenterKeyScreen />;
}

export const router = createBrowserRouter(
  [
    { path: APP_PATHS.participant, element: <ParticipantRoute /> },
    { path: APP_PATHS.presenter, element: <PresenterRoute /> },
    { path: '*', element: <Navigate to={APP_PATHS.participant} replace /> },
  ],
  { basename: '/shop' },
);
