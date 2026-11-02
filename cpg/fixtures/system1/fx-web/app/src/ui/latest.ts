// Decoy: the word "test" sits inside the basename, not in a test suffix.
export function latest<T>(values: T[]): T | undefined {
  return values[values.length - 1];
}
