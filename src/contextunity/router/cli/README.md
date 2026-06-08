# Router CLI (`contextrouter`) — Legacy

> **Status**: Deprecated. Will be replaced by the unified `contextunity` CLI (see [CU-244](../../../../../../../planner/roadmap/phase-00/tasks/cu-244-cli-handoff-spine-and-local-bootstrap-contract.md) and [cli_blueprint.md](../../../../../../../planner/reference/appendixes/cli_blueprint.md)).

## Current Commands

| Command | What it does | Notes |
|---------|-------------|-------|
| `contextrouter serve` | Starts the Router gRPC server | Production use. Will migrate to `contextunity router serve`. |
| `contextrouter models generate` | Smoke-tests a single LLM adapter | Dev utility. Calls `model_registry.create_llm()` directly, bypasses Router pipeline. Uses bootstrap `get_auth_metadata()` to satisfy SecureNode token checks. |
| `contextrouter models test-multimodal` | Tests text/image/audio/video/stream across providers | Same as above but iterates all configured providers. |

## Entry Point

Registered in `pyproject.toml` as:

```toml
[project.scripts]
contextrouter = "contextunity.router.cli.app:main"
```

## What Replaces This

The `contextunity` metapackage CLI defined in planner phases:

- **Phase 0** — CLI spine and local bootstrap contract (CU-244)
- **Phase 4** — `contextunity router dlq`, Event Journal, TestMan surfaces
- **Phase 5** — Headless Lab Kernel, graph memory CLI
- **Phase 6** — Shield command surface, service token broker

The `models generate` / `test-multimodal` commands have no direct successor in the roadmap. If needed, they can be reimplemented as `contextunity dev test-llm` under the new CLI.
