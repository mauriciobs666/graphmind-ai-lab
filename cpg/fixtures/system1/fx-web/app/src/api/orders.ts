import { API_PREFIX, ORDER_PATH, SAMPLE_ORDER } from "./endpoints";

export async function getOrder() {
  const res = await fetch(API_PREFIX + ORDER_PATH + "/" + SAMPLE_ORDER);
  return res.json();
}
