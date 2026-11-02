// Build-script check; lives outside the unit's rootPath.
export function buildTest(): boolean {
  return process.argv.length >= 2;
}
