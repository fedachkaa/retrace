import argparse
import json
from datetime import UTC, datetime
from pathlib import Path

from retrace_memory.extraction import ExtractionResult
from retrace_memory.models import (
    Entity,
    EntityType,
    Event,
    EventStatus,
    EvidenceScope,
    InformationOrigin,
    Memory,
    MemorySourceType,
    SourceReference,
    TemporalContext,
    TemporalPrecision,
)
from retrace_memory.persistence import MemoryStore


def main(arguments: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run the RETRACE SQLite demo")
    parser.add_argument("database", type=Path, help="path for the local SQLite file")
    options = parser.parse_args(arguments)

    content = "Sam recorded a synthetic project milestone in November."
    memory = Memory(
        content=content,
        created_at=datetime(2026, 10, 14, 12, 0, tzinfo=UTC),
        source_type=MemorySourceType.TEXT,
        language="en",
    )
    entity = Entity(display_name="Sam", entity_type=EntityType.PERSON)
    quote = "synthetic project milestone"
    start = content.index(quote)
    evidence = SourceReference(
        memory_id=memory.id,
        scope=EvidenceScope.TEXT_SPAN,
        start_offset=start,
        end_offset=start + len(quote),
        quote=quote,
    )
    event = Event(
        title="Synthetic project milestone",
        participant_entity_ids=(entity.id,),
        temporal_context=TemporalContext(
            precision=TemporalPrecision.MONTH,
            expression="November",
        ),
        evidence=(evidence,),
        status=EventStatus.REPORTED,
        origin=InformationOrigin.AI_INFERRED,
    )
    result = ExtractionResult(
        memory_id=memory.id,
        entities=(entity,),
        events=(event,),
    )

    with MemoryStore(options.database) as store:
        store.save_memory(memory)
        store.save_extraction(result)

    with MemoryStore(options.database) as store:
        output = {
            "memory": store.get_memory(memory.id).model_dump(mode="json"),
            "extraction": store.get_extraction(memory.id).model_dump(mode="json"),
            "connections": [
                connection.model_dump(mode="json")
                for connection in store.list_connections(memory.id)
            ],
            "timeline": store.get_timeline().model_dump(mode="json"),
        }

    print(json.dumps(output, indent=2))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
