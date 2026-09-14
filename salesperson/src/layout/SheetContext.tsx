// Local UI state for which bottom sheet (if any) is open. Created and owned
// entirely within this step's own subtree (`layout/**`,
// docs/plans/salesperson-ui.md §5.1's S12b row) — pure client UI state, no
// participant/session data involved, so it is safe to establish from
// `LayoutShell` even though (per `./Shell.tsx`'s top comment) that component
// sits outside `SessionContext`/TanStack Query/the router in App.tsx's fixed
// composition.
import { createContext, useContext, useMemo, useState, type ReactNode } from 'react';

export type SheetKey = 'catalog' | 'cart' | 'order' | 'profile';

interface SheetContextValue {
  openSheet: SheetKey | null;
  open: (key: SheetKey) => void;
  close: () => void;
}

const SheetContext = createContext<SheetContextValue | null>(null);

export function SheetStateProvider({ children }: { children: ReactNode }) {
  const [openSheet, setOpenSheet] = useState<SheetKey | null>(null);

  const value = useMemo<SheetContextValue>(
    () => ({
      openSheet,
      open: (key: SheetKey) => setOpenSheet(key),
      close: () => setOpenSheet(null),
    }),
    [openSheet],
  );

  return <SheetContext.Provider value={value}>{children}</SheetContext.Provider>;
}

export function useSheetState(): SheetContextValue {
  const ctx = useContext(SheetContext);
  if (!ctx) {
    throw new Error('useSheetState must be used within a SheetStateProvider');
  }
  return ctx;
}
