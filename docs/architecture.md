# RETRACE architecture

## Current implementation

RETRACE is a uv-managed Python monorepo targeting Python 3.12. It currently contains
two independently packaged workspace members:

- `apps/web` exposes a minimal FastAPI application and a health endpoint.
- `packages/memory-engine` exposes the importable `retrace_memory` package.

The web application depends on the memory engine through an explicit uv workspace
dependency. This verifies the package boundary without adding business logic. There
is no database, model runtime, external API, authentication, or user interface.

## Planned boundaries

### Web application

The web application will provide the prototype's HTTP API and eventual web UI. HTTP
concerns should remain in this application. Domain and AI memory behavior that needs
to be reusable should live in the memory engine.

### Memory engine

The memory engine is intended to own reusable processing for memories, including
future structured extraction, evidence-backed relationships, inconsistency signals,
and timeline construction. None of those capabilities are implemented yet. The
engine should remain independent of FastAPI so other clients can reuse it.

### Flutter application

A Flutter mobile application is planned as a separate monorepo application. It will
consume a stable API or communicate with a local companion service. It has not been
created in this foundation.

### Local companion service

An optional Python companion service is planned for local processing and private
model execution. It should reuse `retrace_memory` without depending on the web
application. It has not been created in this foundation.

## Privacy and evidence

The architecture is intended to support local-first processing and explicit user
control over personal data. Future AI-generated relationships or inconsistency
signals must preserve their supporting source evidence and be reviewable rather than
presented as authoritative judgments. RETRACE is not intended for surveillance or
lie detection.
