# Contributing

Thanks for contributing to **ContextRouter**.

## Development setup

```bash
cd contextrouter
uv pip install -e '.[dev]'
```

If you work on ingestion code/CLI:

```bash
uv pip install -e '.[dev,ingestion]'
```

## Pre-commit

Install the git hooks once:

```bash
pre-commit install
```

Run on-demand:

```bash
pre-commit run --all-files
```

## Linting & tests

```bash
uv run ruff check . --fix
uv run ruff format .
uv run python -m pytest -q
```

## Branching & GitHub workflow

### Branch naming

- **Features**: `feat/<short-topic>`
- **Fixes**: `fix/<short-topic>`
- **Chores**: `chore/<short-topic>`
- **Docs**: `contextrouter_docs/<short-topic>`
- **Refactors**: `refactor/<short-topic>`

Keep names lowercase, dash-separated. Example: `feat/ingestion-runner`.

### PR flow (recommended)

- Branch off `main`
- Open a PR early (Draft is fine)
- Keep PRs small and focused; avoid mixing unrelated changes
- Before requesting review:
  - `pre-commit run --all-files`
  - ensure CI is green (lint + tests)

### Merge strategy

- Prefer **Squash & merge** into `main` (keeps history clean)
- Use **Conventional Commits** style in the squash commit title:
  - `feat: ...`, `fix: ...`, `docs: ...`, `refactor: ...`, `chore: ...`, `test: ...`

### Releases

- Bump version in `pyproject.toml` (SemVer)
- Tag releases as `vX.Y.Z`

## Error/exception conventions

- All internal exceptions that cross module/transport boundaries should inherit from
  `contextrouter.core.exceptions.ContextrouterError`.
- Every `ContextrouterError` must have a stable, **non-empty** `code` string.
- Prefer raising typed errors close to the boundary (providers/connectors/modules), and let
  transports map `code` to their own protocol.

## Architecture constraints (summary)

- `contextrouter.core` is the kernel: keep it knowledge-agnostic.
- RAG-specific types/settings live under `modules/retrieval/rag/`.
- Ingestion-specific code lives under `modules/ingestion/`.
- Graph wiring lives in `cortex/graphs/`; business logic in `cortex/steps/`.
