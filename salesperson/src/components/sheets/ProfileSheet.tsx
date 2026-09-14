import { useSheetState } from '../../layout/SheetContext';
import { ProfilePanel } from '../../views/ProfilePanel';
import { BottomSheet } from './BottomSheet';
import { ResetControl } from './ResetControl';

export function ProfileSheet() {
  const { openSheet, close } = useSheetState();
  return (
    <BottomSheet open={openSheet === 'profile'} title="Profile" onClose={close}>
      <div className="flex flex-col gap-6">
        <ProfilePanel />
        <hr className="border-slate-100 dark:border-slate-800" />
        {/* AC-5's participant half lives in this sheet's chrome, not
         * ProfilePanel (S14's subtree) — see ResetControl's top comment. */}
        <ResetControl onReset={close} />
      </div>
    </BottomSheet>
  );
}
