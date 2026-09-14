import { useSheetState } from '../../layout/SheetContext';
import { CartPanel } from '../../views/CartPanel';
import { BottomSheet } from './BottomSheet';

export function CartSheet() {
  const { openSheet, close } = useSheetState();
  return (
    <BottomSheet open={openSheet === 'cart'} title="Cart" onClose={close}>
      <CartPanel />
    </BottomSheet>
  );
}
