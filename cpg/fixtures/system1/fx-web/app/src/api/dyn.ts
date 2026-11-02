const base = import.meta.env.VITE_API_BASE;

// Every caller supplies its own path, so the target is not knowable here.
export async function go(path: string) {
  return fetch(base + path);
}

export async function forward(...args: Parameters<typeof fetch>) {
  return fetch(...args);
}
