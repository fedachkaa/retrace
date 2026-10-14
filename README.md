# RETRACE

RETRACE is a privacy-first personal memory application for capturing notes,
conversations, observations, and events. The planned AI memory engine will help users
extract structured information, review evidence-based connections, identify potential
inconsistencies, and explore timelines. RETRACE is not a lie detector or surveillance
tool.

This repository currently implements a minimal FastAPI service, an importable
memory-engine package, a local Ollama-backed extraction pipeline, and development
tooling. Persistence, a frontend, the Flutter app, and the local companion service
are planned but not implemented.

## Monorepo structure

```text
apps/web/                 FastAPI web application
packages/memory-engine/   Reusable Python memory-engine package
evals/                    Future AI evaluation datasets
docs/                     Architecture and product documentation
.github/workflows/        Continuous integration
```

Both Python projects use a `src/` layout and are managed from the root uv workspace.

## Requirements

- [uv](https://docs.astral.sh/uv/)
- Python 3.12 (uv can install the required interpreter)

No database or external API is required. Ollama is optional and is needed only for
the local extraction demo.

## Installation

From the repository root:

```shell
uv python install 3.12
uv sync --locked --all-packages
```

## Start the API

```shell
uv run uvicorn retrace_web.main:app --reload
```

The health endpoint is available at `http://127.0.0.1:8000/health` and returns
`{"status":"ok"}`.

## Tests

```shell
uv run pytest
```

The test suite measures branch coverage for both Python packages and requires 100%
coverage for the current minimal foundation.

## Linting and formatting

Check linting and formatting:

```shell
uv run ruff check .
uv run ruff format --check .
```

Apply automatic fixes and formatting during development:

```shell
uv run ruff check --fix .
uv run ruff format .
```

## Privacy direction

RETRACE is designed toward local-first processing so personal memories can remain
under the user's control. AI-generated claims and events retain exact supporting
evidence and remain reviewable by the user. See
[Local AI memory extraction](docs/local-ai-extraction.md) for Ollama setup, the demo,
privacy boundaries, and limitations.
