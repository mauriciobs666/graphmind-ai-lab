"""ccg's side of the common LLM gateway (docs/plans/llm-gateway.md §3.2, §3.6): `BrokerTransport`, the gateway
`Transport` that sends every model call through the egress broker, and `ManifestMarkers`, the `ScopeMarkerSource`
backed by the installed manifest. Importing this package needs the `gateway` extra (`llmgateway`); `ccg.broker`
never imports it."""
