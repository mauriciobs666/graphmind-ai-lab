const BASE = import.meta.env.VITE_API_BASE;
const DEFAULT_ITEM = 1;

export async function getDefaultItem() {
  const res = await fetch(`${BASE}/api/items/${DEFAULT_ITEM}`);
  return res.json();
}
