import { API_PREFIX } from "./endpoints";

const README = "readme.txt";

export async function fetchFile() {
  const res = await fetch(`${API_PREFIX}/files/${README}`);
  return res.text();
}
