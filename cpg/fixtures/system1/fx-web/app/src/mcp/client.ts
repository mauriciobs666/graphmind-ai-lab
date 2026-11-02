import { Client } from "@modelcontextprotocol/sdk/client/index.js";
import { StreamableHTTPClientTransport } from "@modelcontextprotocol/sdk/client/streamableHttp.js";
import { TOOL_LIST_ORDERS } from "./names";

let client: Client | undefined;

export async function connect(): Promise<Client> {
  if (client) {
    return client;
  }
  const transport = new StreamableHTTPClientTransport(new URL("http://localhost:8200/mcp"));
  client = new Client({ name: "fx-web", version: "1.0.0" });
  await client.connect(transport);
  return client;
}

export async function callTool(name: string, args: Record<string, unknown> = {}) {
  const c = await connect();
  return c.callTool({ name, arguments: args });
}

export async function loadItem() {
  const c = await connect();
  return c.callTool({ name: "get_item", arguments: { id: 1 } });
}

export async function loadOrders() {
  const c = await connect();
  return c.callTool({ name: TOOL_LIST_ORDERS, arguments: {} });
}
