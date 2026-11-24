"""`BrokerTransport`: the gateway `Transport` that sends every model call through the egress broker
(docs/plans/llm-gateway.md §3.2, §3.3, §3.4 "ccg path"; egress-spec5 §2-§3).

One `WireRequest` becomes one broker `request` (component from the call site, `purpose: model-call`, the touched
repositories read from the `ccg:repo:` scopes, `gatewayRunId` from the run id). The broker decides (R1 local, R7
hosted) and, for R7, forwards to the key holder; this process never reaches a provider. `reaches_hosted` is True in
that sense only: the broker's policy and the holder's validation stay authoritative, and a restricted scope never
gets a hosted chain from the gateway's policy in the first place.

Outcomes, all through `llmgateway.failures` (§3.3):
  * decision deny                            -> `BROKER_DENIAL` (policy_refused), never retried
  * closeReason upstream-error / too-large / timeout -> `BROKER_UPSTREAM_ERROR` / `BROKER_RESPONSE_TOO_LARGE` / `BROKER_TIMEOUT`
  * `BrokerUnavailable`, `BrokerError`        -> `BROKER_UNAVAILABLE` (config)
  * a completed exchange (any status, a holder refusal's 499 included) -> `WireResponse`, classified by the engine
    through the provider's `classify` (which checks the refusal content type first)

`BrokerClient.request` drops the reply's `closeReason`, which the timeout row needs, so the exchange goes through
`BrokerClient._call` with the same message `request` builds.
"""
from __future__ import annotations

import base64
import json
from collections.abc import Callable
from typing import Any

from ccg.broker.client import BrokerClient, BrokerError, BrokerUnavailable
from ccg.gateway.manifestmarkers import REPO_PREFIX
from llmgateway.failures import Condition
from llmgateway.transport import TransportFailure, WireRequest, WireResponse
from llmgateway.types import CallFailed, FailureClass

PURPOSE = "model-call"
_CLOSE_CONDITIONS = {
    "upstream-error": Condition.BROKER_UPSTREAM_ERROR,
    "response-too-large": Condition.BROKER_RESPONSE_TOO_LARGE,
    "timeout": Condition.BROKER_TIMEOUT,
}


class BrokerTransport:
    """`token` returns the broker run token for this call (the caller owns the run: `open_run`/`close_run`)."""

    reaches_hosted = True

    def __init__(self, client: BrokerClient, token: Callable[[], str]) -> None:
        self._client, self._token = client, token

    def exchange(self, request: WireRequest) -> WireResponse:
        body: Any = None
        if request.body:
            try:
                body = json.loads(request.body)
            except ValueError:
                raise CallFailed(FailureClass.CONFIG, f"{request.provider_id}: body is not JSON") from None
        req: dict[str, Any] = {
            "component": request.component,
            "purpose": PURPOSE,
            "payloadClass": "none" if body is None else "repo-derived",
            "endpoint": request.base_url,
            "target": {"method": request.method, "path": request.path},
            "codebases": [s[len(REPO_PREFIX):] for s in request.scopes if s.startswith(REPO_PREFIX)],
        }
        if request.run_id is not None:
            req["gatewayRunId"] = request.run_id
        try:
            reply = self._client._call({"op": "request", "token": self._token(), "request": req, "body": body})
        except (BrokerUnavailable, BrokerError) as exc:
            raise TransportFailure(Condition.BROKER_UNAVAILABLE, f"broker: {type(exc).__name__}") from None
        decision = reply["decision"]
        if decision["decision"] != "allow":
            raise TransportFailure(
                Condition.BROKER_DENIAL, f"broker denied the call ({decision['rule']}: {decision['reason']})"
            )
        resp = reply.get("response")
        if resp is None:
            reason = reply.get("closeReason")
            raise TransportFailure(_CLOSE_CONDITIONS.get(reason, Condition.BROKER_UPSTREAM_ERROR), f"broker: {reason}")
        return WireResponse(resp["status"], base64.b64decode(resp["bodyB64"]), resp["contentType"])
