# fx-lib

Small shared pricing library with a gRPC service, an in-process event bus and a command-line consumer.

- `pricing/` - the library (`com.acme.fxlib`), the `.proto` contract and the in-process bus.
- `quote-cli/` - a command-line consumer of the library, built against the same version.

These sources are a parse-only fixture: they are not buildable with Maven (no protobuf plugin generates the `.proto` classes, and `quote-cli` imports a vendor jar that is deliberately absent).
