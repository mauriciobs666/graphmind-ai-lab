// S12a's placeholder for App.tsx's layout-shell slot
// (docs/plans/salesperson-ui.md §5.1's S12a row: "App.tsx exposes ... a
// layout-shell slot ... with a no-op default, so that S12b ... can mount
// into [it] without editing [App.tsx]").
//
// `src/layout/**` is S12b's owned subtree (§5.0's shared-file map) — S12b
// replaces this file's content with the mobile shell (sticky header,
// bottom-sheet overlays, safe-area insets). Until then it is a pass-through.
import type { ReactNode } from 'react';

export function LayoutShell({ children }: { children: ReactNode }) {
  return <>{children}</>;
}
