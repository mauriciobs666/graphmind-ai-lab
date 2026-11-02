import { callTool } from "../mcp/client";

export function onReindex() {
  return callTool("reindex");
}

export function Admin() {
  return <button onClick={onReindex}>Rebuild index</button>;
}
