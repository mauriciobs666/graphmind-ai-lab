// Autoscroll-when-at-bottom (mirroring `falkor-chat/web/app.js`'s
// `pollMessages`): jsdom never lays out real scroll dimensions, so these
// tests stage `scrollHeight`/`clientHeight`/`scrollTop` on the transcript
// element directly (the standard way to exercise this behaviour without a
// real browser) and drive the component's own `scroll` listener via
// `fireEvent.scroll` to set its "am I at the bottom" tracking, exactly as a
// real scroll would.
import { fireEvent, render, screen } from '@testing-library/react';
import { describe, expect, it } from 'vitest';
import type { MessageRow } from '../../api/endpoints';
import { Transcript } from './Transcript';
import { WELCOME_ROW_ID } from './welcome';

function row(msgId: string, text: string): MessageRow {
  return { msgId, threadId: 't', authorId: 'p', text, role: 'user', createdAt: 1, mentions: [] };
}

function stageScroll(
  el: HTMLElement,
  { scrollHeight, clientHeight, scrollTop }: { scrollHeight: number; clientHeight: number; scrollTop: number },
) {
  Object.defineProperty(el, 'scrollHeight', { value: scrollHeight, configurable: true });
  Object.defineProperty(el, 'clientHeight', { value: clientHeight, configurable: true });
  Object.defineProperty(el, 'scrollTop', { value: scrollTop, writable: true, configurable: true });
}

describe('Transcript', () => {
  it('shows the empty state with no rows and nothing pending', () => {
    render(<Transcript rows={[]} />);
    expect(screen.getByText(/no messages yet/i)).toBeInTheDocument();
  });

  it('renders the pending echo (optimistic send) alongside any real rows', () => {
    render(<Transcript rows={[row('1', 'hi')]} pendingText="on its way" />);
    expect(screen.getByText('hi')).toBeInTheDocument();
    expect(screen.getByText('on its way')).toBeInTheDocument();
    expect(screen.getByText('Sending…')).toBeInTheDocument();
  });

  // §4.12 — the welcome turn (identified by `WELCOME_ROW_ID`) carries no
  // real wire timestamp; `Transcript` must suppress the caption for that one
  // row without touching any other row's.
  it('suppresses the timestamp caption for the welcome row only', () => {
    const welcomeRow: MessageRow = {
      msgId: WELCOME_ROW_ID, threadId: '', authorId: 'agent',
      text: 'Welcome to the store, Ada.', role: 'assistant', createdAt: 0, mentions: [],
    };
    render(<Transcript rows={[welcomeRow, row('1', 'hi')]} />);
    expect(screen.getByText('Welcome to the store, Ada.')).toBeInTheDocument();
    // The ordinary row still gets its caption…
    const ordinaryBubble = screen.getByText('hi').parentElement;
    expect(ordinaryBubble?.children.length).toBe(2);
    // …the welcome row's does not.
    const welcomeBubble = screen.getByText('Welcome to the store, Ada.').parentElement;
    expect(welcomeBubble?.children.length).toBe(1);
  });

  it('scrolls to the bottom on a new row when the viewer was already there', () => {
    const { rerender } = render(<Transcript rows={[row('1', 'first')]} />);
    const list = screen.getByRole('list', { name: 'Transcript' });

    stageScroll(list, { scrollHeight: 500, clientHeight: 500, scrollTop: 470 });
    fireEvent.scroll(list); // 500 - 470 - 500 < 40 -> "at bottom"

    rerender(<Transcript rows={[row('1', 'first'), row('2', 'second')]} />);

    expect(list.scrollTop).toBe(500);
  });

  it('does not yank the view down when the viewer had scrolled up to read history', () => {
    const { rerender } = render(<Transcript rows={[row('1', 'first')]} />);
    const list = screen.getByRole('list', { name: 'Transcript' });

    stageScroll(list, { scrollHeight: 1000, clientHeight: 300, scrollTop: 100 });
    fireEvent.scroll(list); // far from the bottom (1000 - 100 - 300 = 600 >= 40)

    rerender(<Transcript rows={[row('1', 'first'), row('2', 'second')]} />);

    expect(list.scrollTop).toBe(100);
  });
});
