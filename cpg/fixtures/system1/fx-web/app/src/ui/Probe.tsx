// Probes that exercise URL shapes; each returns the raw fetch promise.
export function probeEncoded() {
  return fetch("/api/v1/%2E%2E/health");
}

export function probeDeep() {
  return fetch("/api/items/%2E%2E/%2E%2E/api/health");
}

export function probeUnreserved() {
  return fetch("/api/%68ealth");
}

export function probeReserved() {
  return fetch("/api/files/a%2fb");
}

export function probeToken() {
  return fetch("/api/hooks/x7k2%2Dq9w4%2Dm1p8%2Dz3v6");
}

export function probeVerb() {
  return fetch("/api/items/7", { method: "DELETE" });
}

export function probePrefix() {
  return fetch("/items/7");
}
