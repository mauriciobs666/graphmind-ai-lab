import { useTranslation } from 'react-i18next';
import { useSheetState } from '../../layout/SheetContext';
import { CartPanel } from '../../views/CartPanel';
import { BottomSheet } from './BottomSheet';

export function CartSheet() {
  const { t } = useTranslation();
  const { openSheet, close } = useSheetState();
  return (
    <BottomSheet open={openSheet === 'cart'} title={t('layout.sheet.title.cart')} onClose={close}>
      <CartPanel />
    </BottomSheet>
  );
}
