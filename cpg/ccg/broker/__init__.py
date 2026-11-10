"""The egress broker (C-1001 P1-29): one host process per user that owns the allow-lists, run tokens, rule
evaluation, the upstream connection and the single audit writer. Spec: ``docs/plans/connected-code-graph-egress-spec3.md``
section 2-5. Modules: `policy` (rule table), `runs` (tokens), `audit` (records), `addr` (address validation),
`structured` (the upstream exchange), `framing` (the socket's frames), `server` (the process), `client`
(`BrokerClient`, the only module another component imports)."""
