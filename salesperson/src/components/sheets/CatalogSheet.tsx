import { useTranslation } from 'react-i18next';
import { useSheetState } from '../../layout/SheetContext';
import { CatalogPanel } from '../../views/CatalogPanel';
import { BottomSheet } from './BottomSheet';

export function CatalogSheet() {
  const { t } = useTranslation();
  const { openSheet, close } = useSheetState();
  return (
    <BottomSheet
      open={openSheet === 'catalog'}
      title={t('layout.sheet.title.catalog')}
      onClose={close}
    >
      <CatalogPanel />
    </BottomSheet>
  );
}
