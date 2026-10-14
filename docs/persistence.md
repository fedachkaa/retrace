# SQLite persistence and memory linking

RETRACE stores memories and extraction results in one local SQLite database through
`MemoryStore`. The store uses Python's built-in `sqlite3`; it has no FastAPI or Ollama
dependency and does not store prompts or raw model responses.

## Schema and initialization

Opening `MemoryStore(path)` creates schema version 1 when the file is new. SQLite's
`user_version` and the `schema_info` table record the version. A database created by a
newer unsupported version is rejected instead of being interpreted optimistically.

The normalized schema contains:

- `memories` for exact original content and metadata;
- `extractions` for one snapshot per memory;
- global `entities` plus `extraction_entities` associations;
- `claims`, `events`, ordered participants, and ordered source references;
- temporal precision, exact instants, and original temporal expressions;
- extraction diagnostics.

Foreign key enforcement is enabled on every connection. Indexes cover recent-memory
ordering, shared-entity lookup, and exact-event timeline ordering.

## Store API

```python
from retrace_memory import MemoryStore

with MemoryStore("local/retrace.sqlite3") as store:
    store.save_memory(memory)
    store.save_extraction(result)
    restored_memory = store.get_memory(memory.id)
    restored_result = store.get_extraction(memory.id)
    recent = store.list_memories(limit=20)
    connections = store.list_connections(memory.id)
    timeline = store.get_timeline()
```

`MemoryStore` is also explicitly closeable with `close()`.

## Atomic writes and reprocessing

`save_extraction()` creates one immutable snapshot. A second save for the same memory
raises `DuplicateRecordError`; records are never merged or overwritten implicitly.
All entity, claim, event, evidence, participant, and diagnostic inserts run in one
transaction, so any failure rolls back the complete snapshot.

Reprocessing is explicit through `replace_extraction()`. It validates the complete new
result first, then atomically removes the old snapshot and inserts the replacement.
Failure restores the old snapshot. Entities no longer referenced by any extraction are
removed after a successful replacement.

## Evidence integrity

Before writing an extraction, the store loads its original memory and applies
`validate_source_reference()` to every claim and event reference. It rejects missing
memories, mismatched memory IDs, invalid offsets, and non-matching quotations. Missing
evidence is never synthesized or repaired. Database foreign keys supplement these
domain checks; they do not replace them.

Claim speakers and event participants must reference entities in the same extraction.
UUIDs and timezone-aware timestamps are stored as lossless text and reconstructed as
the existing Pydantic domain models.

## Identity and linking

Entities retain their application-owned UUIDs. Names, aliases, and types are never used
to merge identity. Two entities named “Alex” remain unrelated when their UUIDs differ.

`list_connections()` reports only entity UUIDs explicitly shared by two extraction
snapshots. Its `basis` is `shared_entity_id`; no textual similarity, contradiction,
identity resolution, or graph inference is performed. Reusing an entity UUID with
conflicting entity data is rejected.

Although the domain package defines `Relationship`, the current `ExtractionResult`
contract has no relationship collection. Persistence therefore stores no standalone
relationships yet; adding them should follow a future extraction-contract change rather
than create a second, inconsistent snapshot shape here.

## Timeline precision

`get_timeline()` returns `TimelineResult` with two collections:

- `exact` contains only events with an actual timezone-aware exact instant and sorts
  those instants chronologically;
- `ambiguous` contains undated, relative, approximate, day-, month-, year-, and
  unknown-precision events in deterministic identifier order.

Expressions such as “next Friday” and “November” remain unchanged. RETRACE does not
guess dates, years, or timezones. Every timeline entry includes its complete event,
evidence, status, and source memory identifier.

## Local data handling

The caller chooses the database path. The SQLite file remains on that filesystem, but
local storage does not provide encryption at rest. Filesystem permissions, full-disk
encryption, backups, and secure deletion remain the operator's responsibility. Back up
the database only to locations appropriate for its private contents. To delete the
local dataset, close the store and remove the database file and any copies or backups.

## Persistence demo

Run the synthetic, Ollama-free smoke test with an explicit path outside the repository:

```shell
uv run python -m retrace_memory.demo_persistence C:\temp\retrace-demo.sqlite3
```

The demo creates one synthetic memory and extraction, closes and reopens the database,
then prints the restored records, conservative links, and timeline. Use a new database
path for each run because duplicate memory and extraction writes are intentionally
rejected.
