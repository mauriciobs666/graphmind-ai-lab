import { useSheetState } from '../../layout/SheetContext';
import { CatalogPanel } from '../../views/CatalogPanel';
import { BottomSheet } from './BottomSheet';

export function CatalogSheet() {
  const { openSheet, close } = useSheetState();
  return (
    <BottomSheet open={openSheet === 'catalog'} title="Catalog" onClose={close}>
      <CatalogPanel />
    </BottomSheet>
  );
}
