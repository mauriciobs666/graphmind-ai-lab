import { Box, Plain, café } from './a';
export function run(): number {
  const s = "é"; const b = new Box(2);
  new Plain();
  return café() + b.put();
}
export function astral(): number {
  const s = "😀"; return café();
}
