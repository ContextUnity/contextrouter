# syntax=docker/dockerfile:1
FROM python:3.13-slim

WORKDIR /app

# Install uv
COPY --from=ghcr.io/astral-sh/uv:latest /uv /usr/local/bin/uv

# Copy entire workspace
COPY pyproject.toml uv.lock ./
COPY packages packages/
COPY services services/

# Sync all workspace members (frozen to lockfile)
RUN uv sync --frozen

# Run router service
WORKDIR /app/services/router
CMD ["uv", "run", "contextrouter", "serve"]
