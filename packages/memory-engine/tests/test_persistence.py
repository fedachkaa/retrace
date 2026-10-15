import sqlite3
from datetime import UTC, datetime
from pathlib import Path
from uuid import UUID

import pytest

import retrace_memory.persistence as persistence
from retrace_memory import (
    Certainty,
    Claim,
    DatabaseError,
    DuplicateRecordError,
    Entity,
    EntityType,
    Event,
    EventStatus,
    EvidenceScope,
    ExtractionDiagnostic,
    ExtractionResult,
    InformationOrigin,
    InvalidExtractionError,
    InvalidIdentityConfirmationError,
    Memory,
    MemorySourceType,
    MemoryStore,
    MissingEntityError,
    MissingIdentityConfirmationError,
    MissingMemoryError,
    SourceReference,
    TemporalContext,
    TemporalPrecision,
    UnsupportedSchemaVersionError,
)


def uid(value: int) -> UUID:
    return UUID(int=value)


def memory(
    value: int = 1,
    *,
    content: str = "Alex met Robin in Berlin.",
    created_at: datetime | None = None,
) -> Memory:
    return Memory(
        id=uid(value),
        content=content,
        created_at=created_at or datetime(2026, 10, 14, 12, 0, tzinfo=UTC),
        occurred_at=datetime(2026, 10, 13, 18, 30, tzinfo=UTC),
        source_type=MemorySourceType.TEXT,
        language="en",
    )


def reference(source: Memory, quote: str = "Alex") -> SourceReference:
    start = source.content.index(quote)
    return SourceReference(
        memory_id=source.id,
        scope=EvidenceScope.TEXT_SPAN,
        start_offset=start,
        end_offset=start + len(quote),
        quote=quote,
    )


def extraction(
    source: Memory,
    *,
    entity: Entity | None = None,
    event_temporal: TemporalContext | None = None,
) -> ExtractionResult:
    person = entity or Entity(
        id=uid(10 + source.id.int),
        display_name="Alex",
        entity_type=EntityType.PERSON,
        aliases=("A",),
    )
    place = Entity(
        id=uid(20 + source.id.int),
        display_name="Berlin",
        entity_type=EntityType.PLACE,
    )
    evidence = (
        reference(source),
        SourceReference(memory_id=source.id, scope=EvidenceScope.WHOLE_MEMORY),
    )
    return ExtractionResult(
        memory_id=source.id,
        entities=(person, place),
        claims=(
            Claim(
                id=uid(30 + source.id.int),
                statement="Alex was mentioned.",
                speaker_entity_id=person.id,
                evidence=evidence,
                temporal_context=TemporalContext(
                    precision=TemporalPrecision.RELATIVE,
                    expression="yesterday",
                ),
                certainty=Certainty.STATED,
                origin=InformationOrigin.AI_INFERRED,
            ),
        ),
        events=(
            Event(
                id=uid(40 + source.id.int),
                title="Meeting",
                participant_entity_ids=(person.id, place.id),
                temporal_context=event_temporal,
                evidence=evidence,
                status=EventStatus.REPORTED,
                origin=InformationOrigin.AI_INFERRED,
            ),
        ),
        diagnostics=(
            ExtractionDiagnostic(
                item_type="claim",
                item_index=1,
                code="synthetic_test",
                message="Synthetic diagnostic",
            ),
        ),
    )


def test_memory_round_trips_with_uuid_timestamps_and_exact_content(
    tmp_path: Path,
) -> None:
    source = memory(content="  Original text stays exact.\n")

    with MemoryStore(tmp_path / "memory.sqlite3") as store:
        store.save_memory(source)
        restored = store.get_memory(source.id)

    assert restored == source
    assert restored.content == "  Original text stays exact.\n"
    assert restored.created_at.tzinfo is not None
    assert restored.occurred_at.tzinfo is not None


def test_complete_extraction_round_trips_exactly(tmp_path: Path) -> None:
    source = memory()
    original = extraction(
        source,
        event_temporal=TemporalContext(
            precision=TemporalPrecision.MONTH, expression="November"
        ),
    )

    with MemoryStore(tmp_path / "memory.sqlite3") as store:
        store.save_memory(source)
        store.save_extraction(original)

        assert store.get_extraction(source.id) == original


def test_database_persists_across_close_and_reopen(tmp_path: Path) -> None:
    database = tmp_path / "memory.sqlite3"
    source = memory()
    original = extraction(source)

    with MemoryStore(database) as store:
        store.save_memory(source)
        store.save_extraction(original)

    with MemoryStore(database) as reopened:
        assert reopened.schema_version == 2
        assert reopened.foreign_keys_enabled
        assert reopened.get_memory(source.id) == source
        assert reopened.get_extraction(source.id) == original


def test_empty_database_queries_are_deterministic(tmp_path: Path) -> None:
    with MemoryStore(tmp_path / "memory.sqlite3") as store:
        assert store.list_memories() == ()
        assert store.get_timeline().exact == ()
        assert store.get_timeline().ambiguous == ()
        with pytest.raises(MissingMemoryError, match="memory does not exist"):
            store.get_memory(uid(999))
        with pytest.raises(MissingMemoryError, match="snapshot does not exist"):
            store.get_extraction(uid(999))
        with pytest.raises(ValueError, match="limit"):
            store.list_memories(0)


def test_memories_are_listed_recent_first_with_a_limit(tmp_path: Path) -> None:
    older = memory(1, created_at=datetime(2026, 1, 1, tzinfo=UTC))
    newer = memory(2, created_at=datetime(2026, 2, 1, tzinfo=UTC))

    with MemoryStore(tmp_path / "memory.sqlite3") as store:
        store.save_memory(older)
        store.save_memory(newer)

        assert store.list_memories(limit=1) == (newer,)


def test_duplicate_memory_and_extraction_are_rejected(tmp_path: Path) -> None:
    source = memory()
    result = extraction(source)

    with MemoryStore(tmp_path / "memory.sqlite3") as store:
        store.save_memory(source)
        with pytest.raises(DuplicateRecordError, match="memory already exists"):
            store.save_memory(source)
        store.save_extraction(result)
        with pytest.raises(DuplicateRecordError, match="snapshot already exists"):
            store.save_extraction(result)


def test_explicit_reextraction_atomically_replaces_the_snapshot(tmp_path: Path) -> None:
    source = memory()
    original = extraction(source)
    replacement = ExtractionResult(
        memory_id=source.id,
        entities=(),
        claims=(),
        events=(),
        diagnostics=(),
    )

    with MemoryStore(tmp_path / "memory.sqlite3") as store:
        store.save_memory(source)
        with pytest.raises(MissingMemoryError, match="snapshot does not exist"):
            store.replace_extraction(original)
        store.save_extraction(original)
        store.replace_extraction(replacement)

        assert store.get_extraction(source.id) == replacement


def test_extraction_save_rolls_back_every_record_on_database_failure(
    tmp_path: Path,
) -> None:
    database = tmp_path / "memory.sqlite3"
    source = memory()
    result = extraction(source)
    store = MemoryStore(database)
    store.save_memory(source)
    with sqlite3.connect(database) as connection:
        connection.execute(
            """
            CREATE TRIGGER reject_claim BEFORE INSERT ON claims
            BEGIN SELECT RAISE(ABORT, 'synthetic failure'); END
            """
        )

    with pytest.raises(DuplicateRecordError, match="conflicting records"):
        store.save_extraction(result)

    with sqlite3.connect(database) as connection:
        assert connection.execute("SELECT COUNT(*) FROM extractions").fetchone()[0] == 0
        assert connection.execute("SELECT COUNT(*) FROM entities").fetchone()[0] == 0
    store.close()


def test_missing_memory_and_invalid_evidence_are_rejected(tmp_path: Path) -> None:
    database = tmp_path / "memory.sqlite3"
    source = memory()
    other = memory(2)
    good = extraction(source)
    wrong_memory = good.claims[0].model_copy(update={"evidence": (reference(other),)})
    bad_quote = SourceReference(
        memory_id=source.id,
        scope=EvidenceScope.TEXT_SPAN,
        start_offset=0,
        end_offset=4,
        quote="Nope",
    )
    tampered = good.claims[0].model_copy(update={"evidence": (bad_quote,)})

    with MemoryStore(database) as store:
        with pytest.raises(InvalidExtractionError, match="memory does not exist"):
            store.save_extraction(good)
        store.save_memory(source)
        with pytest.raises(InvalidExtractionError, match="different memory"):
            store.save_extraction(good.model_copy(update={"claims": (wrong_memory,)}))
        with pytest.raises(InvalidExtractionError, match="invalid evidence"):
            store.save_extraction(good.model_copy(update={"claims": (tampered,)}))


def test_unknown_entity_references_and_duplicate_entities_are_rejected(
    tmp_path: Path,
) -> None:
    source = memory()
    good = extraction(source)
    unknown = uid(999)
    bad_claim = good.claims[0].model_copy(update={"speaker_entity_id": unknown})
    bad_event = good.events[0].model_copy(update={"participant_entity_ids": (unknown,)})

    with MemoryStore(tmp_path / "memory.sqlite3") as store:
        store.save_memory(source)
        with pytest.raises(InvalidExtractionError, match="claim references"):
            store.save_extraction(good.model_copy(update={"claims": (bad_claim,)}))
        with pytest.raises(InvalidExtractionError, match="event references"):
            store.save_extraction(good.model_copy(update={"events": (bad_event,)}))
        with pytest.raises(InvalidExtractionError, match="duplicate entity"):
            store.save_extraction(
                good.model_copy(update={"entities": (good.entities[0],) * 2})
            )


def test_shared_entity_ids_create_links_but_matching_names_do_not(
    tmp_path: Path,
) -> None:
    first = memory(1)
    second = memory(2)
    third = memory(3)
    shared = Entity(id=uid(500), display_name="Alex", entity_type=EntityType.PERSON)
    same_name = Entity(id=uid(501), display_name="Alex", entity_type=EntityType.PERSON)

    with MemoryStore(tmp_path / "memory.sqlite3") as store:
        for source in (first, second, third):
            store.save_memory(source)
        store.save_extraction(extraction(first, entity=shared))
        store.save_extraction(extraction(second, entity=shared))
        store.save_extraction(extraction(third, entity=same_name))

        connections = store.list_connections(first.id)

        assert len(connections) == 1
        assert connections[0].connected_memory_id == second.id
        assert connections[0].shared_entity_ids == (shared.id,)
        assert connections[0].basis == "shared_entity_id"
        assert connections[0].explanations[0].basis == "shared_entity_id"
        assert connections[0].explanations[0].entity_id == shared.id
        assert store.list_connections(third.id) == ()


def test_entity_match_candidates_normalize_names_without_asserting_identity(
    tmp_path: Path,
) -> None:
    first = memory(1)
    second = memory(2)
    third = memory(3)
    alex = Entity(
        id=uid(501),
        display_name="  Alex   Morgan ",
        entity_type=EntityType.PERSON,
    )
    alias_match = Entity(
        id=uid(502),
        display_name="A. Morgan",
        entity_type=EntityType.PERSON,
        aliases=("alex morgan",),
    )
    incompatible = Entity(
        id=uid(503),
        display_name="Alex Morgan",
        entity_type=EntityType.ORGANIZATION,
    )
    different_name = Entity(
        id=uid(504), display_name="Robin", entity_type=EntityType.PERSON
    )
    same_memory_match = Entity(
        id=uid(505), display_name="alex morgan", entity_type=EntityType.PERSON
    )
    fourth = memory(4)

    with MemoryStore(tmp_path / "memory.sqlite3") as store:
        first_result = extraction(first, entity=alex)
        store.save_memory(first)
        store.save_extraction(
            first_result.model_copy(
                update={"entities": (*first_result.entities, same_memory_match)}
            )
        )
        for source, entity in (
            (second, alias_match),
            (third, incompatible),
            (fourth, different_name),
        ):
            store.save_memory(source)
            store.save_extraction(extraction(source, entity=entity))

        candidates = store.list_entity_match_candidates(alex.id)

        assert len(candidates) == 1
        assert candidates[0].candidate_entity_id == alias_match.id
        assert candidates[0].normalized_names == ("alex morgan",)
        assert candidates[0].memory_ids == (first.id,)
        assert candidates[0].candidate_memory_ids == (second.id,)
        assert store.list_connections(first.id) == ()


def test_confirmed_identity_groups_are_transitive_and_link_memories(
    tmp_path: Path,
) -> None:
    sources = (memory(1), memory(2), memory(3))
    entities = tuple(
        Entity(
            id=uid(600 + index),
            display_name=f"Alex {index}",
            entity_type=EntityType.PERSON,
        )
        for index in range(3)
    )

    with MemoryStore(tmp_path / "memory.sqlite3") as store:
        for source, entity in zip(sources, entities, strict=True):
            store.save_memory(source)
            store.save_extraction(extraction(source, entity=entity))
        store.confirm_entity_identity(entities[0].id, entities[1].id)
        store.confirm_entity_identity(entities[1].id, entities[2].id)

        assert store.get_confirmed_identity_group(entities[0].id) == tuple(
            sorted(entity.id for entity in entities)
        )
        connections = store.list_connections(sources[0].id)
        assert [connection.connected_memory_id for connection in connections] == [
            sources[1].id,
            sources[2].id,
        ]
        assert all(
            connection.basis == "confirmed_identity" for connection in connections
        )
        assert connections[1].explanations[0].entity_id == entities[0].id
        assert connections[1].explanations[0].connected_entity_id == entities[2].id


def test_revocation_preserves_alternate_identity_paths(tmp_path: Path) -> None:
    sources = (memory(1), memory(2), memory(3))
    entities = tuple(
        Entity(
            id=uid(700 + index),
            display_name="Alex",
            entity_type=EntityType.PERSON,
        )
        for index in range(3)
    )

    with MemoryStore(tmp_path / "memory.sqlite3") as store:
        for source, entity in zip(sources, entities, strict=True):
            store.save_memory(source)
            store.save_extraction(extraction(source, entity=entity))
        store.confirm_entity_identity(entities[0].id, entities[1].id)
        store.confirm_entity_identity(entities[1].id, entities[2].id)
        store.confirm_entity_identity(entities[0].id, entities[2].id)

        store.revoke_entity_identity(entities[0].id, entities[1].id)
        assert set(store.get_confirmed_identity_group(entities[0].id)) == {
            entity.id for entity in entities
        }

        store.revoke_entity_identity(entities[0].id, entities[2].id)
        assert store.get_confirmed_identity_group(entities[0].id) == (entities[0].id,)
        assert store.list_connections(sources[0].id) == ()


def test_confirmations_persist_across_reopen_and_candidates_stop_suggesting_them(
    tmp_path: Path,
) -> None:
    database = tmp_path / "memory.sqlite3"
    first = memory(1)
    second = memory(2)
    first_entity = Entity(
        id=uid(801), display_name="Alex", entity_type=EntityType.PERSON
    )
    second_entity = Entity(
        id=uid(802), display_name="ALEX", entity_type=EntityType.PERSON
    )
    with MemoryStore(database) as store:
        for source, entity in ((first, first_entity), (second, second_entity)):
            store.save_memory(source)
            store.save_extraction(extraction(source, entity=entity))
        assert store.list_entity_match_candidates(first_entity.id)
        store.confirm_entity_identity(first_entity.id, second_entity.id)

    with MemoryStore(database) as reopened:
        assert reopened.schema_version == 2
        assert reopened.list_entity_match_candidates(first_entity.id) == ()
        assert reopened.get_confirmed_identity_group(first_entity.id) == (
            first_entity.id,
            second_entity.id,
        )
        assert reopened.list_connections(first.id)[0].basis == "confirmed_identity"


def test_identity_operations_validate_entities_and_direct_edges(tmp_path: Path) -> None:
    source = memory()
    result = extraction(source)
    first_entity = result.entities[0]
    second_entity = Entity(
        id=uid(900), display_name="Robin", entity_type=EntityType.PERSON
    )
    result = result.model_copy(update={"entities": (*result.entities, second_entity)})

    with MemoryStore(tmp_path / "memory.sqlite3") as store:
        store.save_memory(source)
        store.save_extraction(result)
        with pytest.raises(MissingEntityError, match="entity does not exist"):
            store.list_entity_match_candidates(uid(999))
        with pytest.raises(MissingEntityError, match="entity does not exist"):
            store.confirm_entity_identity(first_entity.id, uid(999))
        with pytest.raises(ValueError, match="two different entities"):
            store.confirm_entity_identity(first_entity.id, first_entity.id)

        store.confirm_entity_identity(first_entity.id, second_entity.id)
        with pytest.raises(DuplicateRecordError, match="already exists"):
            store.confirm_entity_identity(second_entity.id, first_entity.id)
        with pytest.raises(MissingIdentityConfirmationError, match="does not exist"):
            store.revoke_entity_identity(first_entity.id, uid(999))


def test_cross_type_confirmation_is_rejected_without_creating_an_edge(
    tmp_path: Path,
) -> None:
    source = memory()
    result = extraction(source)
    person = result.entities[0]
    place = result.entities[1]

    with MemoryStore(tmp_path / "memory.sqlite3") as store:
        store.save_memory(source)
        store.save_extraction(result)

        with pytest.raises(
            InvalidIdentityConfirmationError, match="matching entity types"
        ):
            store.confirm_entity_identity(person.id, place.id)

        assert store.get_confirmed_identity_group(person.id) == (person.id,)
        assert store.get_confirmed_identity_group(place.id) == (place.id,)
        with pytest.raises(MissingIdentityConfirmationError, match="does not exist"):
            store.revoke_entity_identity(person.id, place.id)


def test_identity_database_errors_are_sanitized(tmp_path: Path) -> None:
    store = MemoryStore(tmp_path / "memory.sqlite3")
    store.close()

    with pytest.raises(DatabaseError, match="confirming entity identity failed"):
        store.confirm_entity_identity(uid(1), uid(2))
    with pytest.raises(DatabaseError, match="revoking entity identity failed"):
        store.revoke_entity_identity(uid(1), uid(2))


def test_version_one_database_is_migrated_without_changing_snapshots(
    tmp_path: Path,
) -> None:
    database = tmp_path / "memory.sqlite3"
    source = memory()
    original = extraction(source)
    with MemoryStore(database) as store:
        store.save_memory(source)
        store.save_extraction(original)
    with sqlite3.connect(database) as connection:
        connection.execute("DROP TABLE entity_identity_confirmations")
        connection.execute("UPDATE schema_info SET version = 1")
        connection.execute("PRAGMA user_version = 1")

    with MemoryStore(database) as migrated:
        assert migrated.schema_version == 2
        assert migrated.get_extraction(source.id) == original
        assert migrated.get_confirmed_identity_group(original.entities[0].id) == (
            original.entities[0].id,
        )


def test_conflicting_data_for_a_shared_entity_id_is_rejected(tmp_path: Path) -> None:
    first = memory(1)
    second = memory(2)
    shared = Entity(id=uid(500), display_name="Alex", entity_type=EntityType.PERSON)
    conflict = Entity(
        id=shared.id, display_name="Different Alex", entity_type=EntityType.PERSON
    )

    with MemoryStore(tmp_path / "memory.sqlite3") as store:
        store.save_memory(first)
        store.save_memory(second)
        store.save_extraction(extraction(first, entity=shared))
        with pytest.raises(InvalidExtractionError, match="conflicting data"):
            store.save_extraction(extraction(second, entity=conflict))
        with pytest.raises(MissingMemoryError):
            store.get_extraction(second.id)


def test_timeline_sorts_exact_instants_and_separates_ambiguous_events(
    tmp_path: Path,
) -> None:
    first = memory(1)
    second = memory(2)
    third = memory(3)
    fourth = memory(4)
    early = TemporalContext(
        precision=TemporalPrecision.EXACT,
        instant=datetime(2026, 1, 1, tzinfo=UTC),
    )
    late = TemporalContext(
        precision=TemporalPrecision.EXACT,
        instant=datetime(2026, 2, 1, tzinfo=UTC),
    )
    relative = TemporalContext(
        precision=TemporalPrecision.RELATIVE, expression="next Friday"
    )
    month = TemporalContext(precision=TemporalPrecision.MONTH, expression="November")

    with MemoryStore(tmp_path / "memory.sqlite3") as store:
        for source in (first, second, third, fourth):
            store.save_memory(source)
        store.save_extraction(extraction(first, event_temporal=late))
        store.save_extraction(extraction(second, event_temporal=early))
        store.save_extraction(extraction(third, event_temporal=relative))
        store.save_extraction(extraction(fourth, event_temporal=month))

        timeline = store.get_timeline()

        assert [entry.memory_id for entry in timeline.exact] == [second.id, first.id]
        assert [entry.memory_id for entry in timeline.ambiguous] == [
            third.id,
            fourth.id,
        ]
        assert timeline.ambiguous[0].event.temporal_context.expression == "next Friday"
        assert timeline.ambiguous[1].event.temporal_context.expression == "November"


def test_future_schema_version_is_rejected(tmp_path: Path) -> None:
    database = tmp_path / "future.sqlite3"
    with sqlite3.connect(database) as connection:
        connection.execute("PRAGMA user_version = 3")

    with pytest.raises(UnsupportedSchemaVersionError, match="newer than supported"):
        MemoryStore(database)


def test_initialization_database_errors_are_sanitized(tmp_path: Path) -> None:
    with pytest.raises(DatabaseError, match="initialization failed"):
        MemoryStore(tmp_path)


def test_failed_initialization_rolls_back_and_can_be_retried(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    database = tmp_path / "memory.sqlite3"
    schema = persistence._SCHEMA
    monkeypatch.setattr(
        persistence,
        "_SCHEMA",
        f"{schema}\nINSERT INTO table_that_does_not_exist VALUES (1);",
    )

    with pytest.raises(DatabaseError, match="initialization failed"):
        MemoryStore(database)

    with sqlite3.connect(database) as connection:
        tables = connection.execute(
            "SELECT name FROM sqlite_master WHERE type = 'table'"
        ).fetchall()
        assert tables == []
        assert connection.execute("PRAGMA user_version").fetchone()[0] == 0

    monkeypatch.setattr(persistence, "_SCHEMA", schema)
    with MemoryStore(database) as store:
        assert store.schema_version == persistence.SCHEMA_VERSION


def test_store_rejects_connections_without_foreign_key_enforcement(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class DisabledForeignKeys:
        row_factory: object = None
        closed = False

        def execute(self, statement: str) -> "DisabledForeignKeys":
            return self

        def fetchone(self) -> tuple[int]:
            return (0,)

        def close(self) -> None:
            self.closed = True

    connection = DisabledForeignKeys()
    monkeypatch.setattr(persistence.sqlite3, "connect", lambda path: connection)

    with pytest.raises(DatabaseError, match="could not be enabled"):
        MemoryStore("unused.sqlite3")

    assert connection.closed


def test_save_memory_database_errors_are_sanitized(tmp_path: Path) -> None:
    store = MemoryStore(tmp_path / "memory.sqlite3")
    store.close()

    with pytest.raises(DatabaseError, match="saving memory failed"):
        store.save_memory(memory())


def test_save_extraction_database_errors_are_sanitized(tmp_path: Path) -> None:
    database = tmp_path / "memory.sqlite3"
    source = memory()
    result = extraction(source)
    store = MemoryStore(database)
    store.save_memory(source)
    with sqlite3.connect(database) as connection:
        connection.execute(
            """
            CREATE TRIGGER break_claim BEFORE INSERT ON claims
            BEGIN SELECT unavailable_function(); END
            """
        )

    with pytest.raises(DatabaseError, match="saving extraction failed"):
        store.save_extraction(result)

    with pytest.raises(MissingMemoryError):
        store.get_extraction(source.id)
    store.close()


def test_replace_extraction_rolls_back_invalid_conflicts(tmp_path: Path) -> None:
    first = memory(1)
    second = memory(2)
    shared = Entity(id=uid(500), display_name="Alex", entity_type=EntityType.PERSON)
    original = extraction(first, entity=shared)
    other = extraction(second, entity=shared)
    conflict = Entity(
        id=shared.id, display_name="Different Alex", entity_type=EntityType.PERSON
    )

    with MemoryStore(tmp_path / "memory.sqlite3") as store:
        store.save_memory(first)
        store.save_memory(second)
        store.save_extraction(original)
        store.save_extraction(other)

        with pytest.raises(InvalidExtractionError, match="conflicting data"):
            store.replace_extraction(extraction(first, entity=conflict))

        assert store.get_extraction(first.id) == original


def test_replace_extraction_reports_duplicate_record_conflicts(tmp_path: Path) -> None:
    first = memory(1)
    second = memory(2)
    original = extraction(first)
    other = extraction(second)
    conflicting_claim = original.claims[0].model_copy(update={"id": other.claims[0].id})
    replacement = original.model_copy(update={"claims": (conflicting_claim,)})

    with MemoryStore(tmp_path / "memory.sqlite3") as store:
        store.save_memory(first)
        store.save_memory(second)
        store.save_extraction(original)
        store.save_extraction(other)

        with pytest.raises(DuplicateRecordError, match="conflicting records"):
            store.replace_extraction(replacement)

        assert store.get_extraction(first.id) == original


def test_replace_extraction_database_errors_are_sanitized(tmp_path: Path) -> None:
    database = tmp_path / "memory.sqlite3"
    source = memory()
    original = extraction(source)
    store = MemoryStore(database)
    store.save_memory(source)
    store.save_extraction(original)
    with sqlite3.connect(database) as connection:
        connection.execute(
            """
            CREATE TRIGGER break_replacement BEFORE INSERT ON claims
            BEGIN SELECT unavailable_function(); END
            """
        )

    with pytest.raises(DatabaseError, match="replacing extraction failed"):
        store.replace_extraction(original)

    assert store.get_extraction(source.id) == original
    store.close()
