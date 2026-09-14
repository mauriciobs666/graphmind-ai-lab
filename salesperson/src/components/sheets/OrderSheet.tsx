import { useTranslation } from 'react-i18next';
import { useSheetState } from '../../layout/SheetContext';
import { OrderPanel } from '../../views/OrderPanel';
import { BottomSheet } from './BottomSheet';

export function OrderSheet() {
  const { t } = useTranslation();
  const { openSheet, close } = useSheetState();
  return (
    <BottomSheet
      open={openSheet === 'order'}
      title={t('layout.sheet.title.order')}
      onClose={close}
    >
      <OrderPanel />
    </BottomSheet>
  );
}
