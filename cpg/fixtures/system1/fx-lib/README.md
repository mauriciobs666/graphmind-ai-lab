# fx-lib

Small shared pricing library with a gRPC service, an in-process event bus and a command-line consumer.

- `pricing/` - the library (`com.acme.fxlib`), the `.proto` contract and the in-process bus.
- `quote-cli/` - a command-line consumer of the library, built against the same version.

Build each module with Maven (`mvn -f pricing/pom.xml install`, then `mvn -f quote-cli/pom.xml package`).
