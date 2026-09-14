// The two client-side credentials (docs/plans/salesperson-ui.md §5.3's
// credentials table). Neither type carries `welcome` (join's one-shot
// greeting) or the presenter *key* (never stored — only the token exchanged
// for it) — see the table for why.

export interface ParticipantSession {
  participantId: string;
  token: string;
  displayName: string;
  language: string;
}

export interface PresenterSession {
  token: string;
}
