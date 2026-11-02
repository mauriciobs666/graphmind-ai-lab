// Decoy: "e2e" in a file name, not a directory segment.
export function pad(value: number, width: number): string {
  return String(value).padStart(width, "0");
}
