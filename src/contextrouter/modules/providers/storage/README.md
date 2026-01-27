# Brain Storage Provider

The Brain provider delegates retrieval and storage operations to the `ContextBrain` service. It is the primary way `ContextRouter` interacts with indexed knowledge.

## Configuration

The provider supports two modes of operation, controlled by environment variables or configuration:

| Variable | Description | Default |
|----------|-------------|---------|
| `BRAIN_MODE` | Integration mode: `local` or `grpc` | `local` |
| `BRAIN_GRPC_ENDPOINT` | Address of the Brain gRPC service (required for `grpc` mode) | `localhost:50051` |
| `BRAIN_DATABASE_URL` | Database connection string (required for `local` mode) | - |

### Local Mode (`local`)

In this mode, `ContextRouter` imports `contextbrain` as a library and runs the `BrainService` logic within its own process. This is ideal for monolithic deployments or local development where you want to avoid network overhead.

**Requirements**:
-   `contextbrain` must be installed in the same environment.
-   Database credentials must be provided directly to the Router process.

### gRPC Mode (`grpc`)

In this mode, `ContextRouter` acts as a client and sends requests to a standalone `ContextBrain` service via gRPC. This is ideal for microservices architectures and allows scaling Brain independently from Router.

**Requirements**:
-   `ContextBrain` service must be running and accessible at the specified endpoint.
-   Protos must be compiled in `contextcore`.

## Usage

To use this provider in the RAG pipeline, set the following in your `.env`:

```bash
RAG_PROVIDER=brain
BRAIN_MODE=grpc
BRAIN_GRPC_ENDPOINT=10.0.0.5:50051
```
