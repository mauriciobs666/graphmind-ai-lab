// Minimal stroke icon set for the header + sheet chrome — no icon library
// dependency, matching the plan's "no branding/design-system pass beyond
// polish" framing (§4.2, salesperson/AGENTS.md). Every icon is decorative
// (`aria-hidden`); the interactive element around it supplies the
// accessible name.
export type IconProps = { className?: string };

const SHARED = {
  viewBox: '0 0 24 24',
  fill: 'none',
  stroke: 'currentColor',
  strokeWidth: 1.75,
  strokeLinecap: 'round' as const,
  strokeLinejoin: 'round' as const,
  'aria-hidden': true as const,
};

export function CatalogIcon({ className }: IconProps) {
  return (
    <svg {...SHARED} className={className}>
      <rect x="3" y="3" width="7" height="7" rx="1.5" />
      <rect x="14" y="3" width="7" height="7" rx="1.5" />
      <rect x="3" y="14" width="7" height="7" rx="1.5" />
      <rect x="14" y="14" width="7" height="7" rx="1.5" />
    </svg>
  );
}

export function CartIcon({ className }: IconProps) {
  return (
    <svg {...SHARED} className={className}>
      <path d="M2.5 3h2l2.2 11.4a2 2 0 0 0 2 1.6h8.4a2 2 0 0 0 2-1.6L21 7H6" />
      <circle cx="9" cy="20" r="1.4" fill="currentColor" stroke="none" />
      <circle cx="18" cy="20" r="1.4" fill="currentColor" stroke="none" />
    </svg>
  );
}

export function OrderIcon({ className }: IconProps) {
  return (
    <svg {...SHARED} className={className}>
      <path d="M7 3h10l1 4H6l1-4Z" />
      <path d="M5 7h14l-1.2 12.1a2 2 0 0 1-2 1.9H8.2a2 2 0 0 1-2-1.9L5 7Z" />
      <path d="M9.5 11.5h5" />
    </svg>
  );
}

export function ProfileIcon({ className }: IconProps) {
  return (
    <svg {...SHARED} className={className}>
      <circle cx="12" cy="8" r="3.5" />
      <path d="M4.5 20a7.5 7.5 0 0 1 15 0" />
    </svg>
  );
}

export function CloseIcon({ className }: IconProps) {
  return (
    <svg {...SHARED} className={className}>
      <path d="M6 6l12 12M18 6 6 18" />
    </svg>
  );
}
