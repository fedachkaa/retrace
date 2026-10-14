# Memory domain models

## Overview

The `retrace_memory` package defines immutable Pydantic models for user-created
memories and interpretations derived from them. These are domain models, not HTTP or
persistence schemas.

- `Memory` preserves the original user-provided content and distinguishes creation
  time from the optional time described by the memory.
- `Entity` identifies a mentioned person, place, organization, or other subject.
- `Claim` records an attributed statement without asserting that it is true.
- `Event` records a planned, reported, occurred, or uncertain event.
- `Relationship` records an explicit or inferred connection between two entities.
- `SourceReference` traces an interpretation back to a source memory.
- `TemporalContext` represents exact and unresolved temporal information.

All models forbid unknown fields and are frozen after creation. Updates create new
records rather than mutating original data.

## Ownership and relationships

Memories are independent source records. Entities, claims, events, and relationships
have their own UUIDs and refer to other records by UUID. Identical display names do not
merge entities: two people called Alex remain distinct until a future, explicit entity
resolution process says otherwise.

Claims, events, and relationships contain one or more source references. Their
`origin` records whether the information was explicitly present in user-provided
material or inferred by AI. Extracted records never replace or modify a `Memory`.

## Claims are reports, not facts

A claim means that a statement was recorded. It does not establish objective truth.
For example, a claim that Anna said she would move to Berlin proves only that the
source memory records that statement. A later, conflicting claim remains a separate
report with separate evidence; the models do not decide which report is correct.

`Certainty` is qualitative. It captures whether wording was stated, uncertain, or did
not communicate certainty. It is not a probability and must not be presented as one.

## Provenance and evidence validation

`SourceReference` offsets use zero-based, end-exclusive Python string indices. For
example, offsets `0` and `4` select `text[0:4]`. Both offsets must be supplied together,
the end must be greater than the start, and a supplied quote must have the same
character length as the range.

These structural checks do not require storage access. When the source `Memory` is
available, call `validate_source_reference(reference, memory)` to verify:

- the referenced memory ID matches;
- offsets remain within the original content;
- a quote matches the selected text; or
- a quote without offsets occurs somewhere in the content.

This separation keeps the models reusable without hiding persistence or lookup work.

## Temporal ambiguity

An exact temporal context contains a timezone-aware `instant`. Imprecise values retain
their original expression and an explicit precision such as `month`, `relative`, or
`approximate`. For example, `"next Friday"` remains unresolved rather than being
silently converted using an unknown reference date. `unknown` contains neither an
instant nor a fabricated expression.

## JSON serialization

```python
from datetime import UTC, datetime

from retrace_memory import (
    Claim,
    InformationOrigin,
    Memory,
    MemorySourceType,
    SourceReference,
)

memory = Memory(
    content="Anna said she may move to Berlin.",
    created_at=datetime.now(UTC),
    source_type=MemorySourceType.TEXT,
)
reference = SourceReference(
    memory_id=memory.id,
    start_offset=14,
    end_offset=32,
    quote="may move to Berlin",
)
claim = Claim(
    statement="Anna may move to Berlin.",
    evidence=(reference,),
    origin=InformationOrigin.USER_PROVIDED,
)

payload = claim.model_dump_json()
restored = Claim.model_validate_json(payload)
```

UUIDs, timezone-aware datetimes, enums, tuples, and nested models reconstruct through
Pydantic without losing their domain types.

## Future extension points

Later phases may add persistence mappings, extraction services, entity-resolution
decisions, contradiction signals, and API-specific schemas around these models. Those
features should preserve source references and must not reinterpret a claim as a
verified fact. Audio storage, temporal parsing, and automated inference are outside
this model layer.
