// Thin wrapper over fetch: every api call goes through request().
export async function request(method: string, path: string) {
  const res = await fetch(path, { method });
  if (!res.ok) {
    throw new Error(`request failed: ${res.status}`);
  }
  return res.json();
}
