| # | item | anchor type | config element | mechanism | askable | now | store (ConfigEntry under keyPath / Link BOUND_BY it) | answer |
|---|---|---|---|---|---|---|---|---|
| 0 | 129dc9 | unit | compose.yaml#services.server | server URL (compose service; unit anchor) | no | not asked | 11 / 0 | not asked |
| 1 | 284562 | unit | falkor-chat-demo.json#mcpServers.falkor-chat | server URL (kiro mcpServers; unit anchor) | no | not asked | 1 / 0 | not asked |
| 2 | 2ae2af | unit | config.example.toml#server.falkor-chat | server URL (mcp-monitor server block; unit anchor) | no | not asked | 2 / 0 | not asked |
| 3 | 32da04 | contract | falkor-chat-demo.json#allowedTools[0] | tool name in an allow-list (kiro allowedTools) | yes | miss | 1 / 0 | 22 items, 0 of type config |
| 4 | 3efb7f | contract | falkor-chat-demo.json#tools[0] | tool name in a tools list (kiro tools) | yes | miss | 1 / 0 | 22 items, 0 of type config |
| 5 | 81fd8c | unit | .mcp.json#mcpServers.falkor-chat-agent-team | server URL (.mcp.json; unit anchor) | no | not asked | 2 / 0 | not asked |
| 6 | 9c7140 | symbol | opencode.json#agent.tank | launches a script (tank bash allow-list path) | yes | miss | 48 / 0 | 0 items |
| 7 | ae355d | symbol | config.example.toml#server.fake-test | launches a script (command in TOML server block) | yes | miss | 4 / 0 | 0 items |
| 8 | da060c | symbol | opencode.json#agent.tank | launches a script (tank prompt + allow-list) | yes | miss | 48 / 0 | 0 items |
| 9 | f16850 | symbol | opencode.json#agent.tank | launches a script (tank prompt + allow-list) | yes | miss | 48 / 0 | 0 items |
| 10 | f8844c | contract | falkor-chat-demo.json#prompt | tool name in a prompt (kiro `prompt` field) | yes | miss | 1 / 0 | 22 items, 0 of type config |
