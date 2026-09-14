// salesperson/AGENTS.md hard constraint #4 / docs/plans/salesperson-ui.md
// §4.2: agent-emitted markup must render as literal text, never parsed as
// HTML — no `dangerouslySetInnerHTML` anywhere in this tree.
import { render, screen } from '@testing-library/react';
import { describe, expect, it } from 'vitest';
import { MessageBubble } from './MessageBubble';

describe('MessageBubble', () => {
  it('renders agent-emitted markup as literal text, not parsed HTML', () => {
    const markup = '<b>bold</b><img src="x" onerror="window.__pwned = true">';
    render(
      <ul>
        <MessageBubble
          row={{ msgId: '1', text: markup, role: 'assistant', createdAt: Date.now() }}
        />
      </ul>,
    );

    // The literal string is visible as text…
    expect(screen.getByText(markup)).toBeInTheDocument();
    // …and was never parsed as an element: no <img> exists, and the escape
    // hatch this repo bans is not present in the rendered subtree.
    expect(screen.queryByRole('img')).not.toBeInTheDocument();
    expect(document.querySelector('img')).toBeNull();
    expect(document.querySelector('b')).toBeNull();
    expect((globalThis as { __pwned?: boolean }).__pwned).toBeUndefined();
  });

  it('aligns an assistant row to the start and a participant row to the end', () => {
    const { container: assistantContainer } = render(
      <ul>
        <MessageBubble row={{ msgId: '1', text: 'hi', role: 'assistant', createdAt: 1 }} />
      </ul>,
    );
    expect(assistantContainer.querySelector('li')).toHaveClass('justify-start');

    const { container: userContainer } = render(
      <ul>
        <MessageBubble row={{ msgId: '2', text: 'hi', role: 'user', createdAt: 1 }} />
      </ul>,
    );
    expect(userContainer.querySelector('li')).toHaveClass('justify-end');
  });

  it('shows a "Sending…" caption instead of a timestamp while pending', () => {
    render(
      <ul>
        <MessageBubble
          row={{ msgId: '__pending__', text: 'hi', role: 'user', createdAt: Date.now() }}
          pending
        />
      </ul>,
    );
    expect(screen.getByText('Sending…')).toBeInTheDocument();
  });
});
