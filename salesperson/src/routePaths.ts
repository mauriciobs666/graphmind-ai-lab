// The two top-level URL trees this SPA mounts (docs/plans/salesperson-ui.md
// §3/§5.1: "route shell for join / chat / presenter"). `App.tsx`'s router is
// given `basename="/shop"` (the FastAPI mount point, §4.1), so these are
// paths *inside* that mount, not full URLs.
export const APP_PATHS = {
  /** Join screen when no participant session exists; the same screen also
   * renders the post-reset language step (C7) and, once a session exists,
   * is where S13's chat view mounts. */
  participant: '/',
  /** Presenter key entry when no presenter session exists; the roster/reset
   * view once one does (S12d). */
  presenter: '/presenter',
} as const;
