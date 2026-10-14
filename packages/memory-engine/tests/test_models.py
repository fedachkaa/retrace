from datetime import UTC, datetime
from uuid import UUID, uuid4

import pytest
from pydantic import ValidationError

from retrace_memory import (
    Certainty,
    Claim,
    Entity,
    EntityType,
    Event,
    EventStatus,
    InformationOrigin,
    Memory,
    MemorySourceType,
    Relationship,
    SourceReference,
    TemporalContext,
    TemporalPrecision,
    validate_source_reference,
)


def aware_datetime() -> datetime:
    return datetime(2026, 10, 14, 12, 30, tzinfo=UTC)


def memory(content: str = "Anna said she might move to Berlin next Friday.") -> Memory:
    return Memory(
        content=content,
        created_at=aware_datetime(),
        source_type=MemorySourceType.TEXT,
        language="en",
    )


def evidence(source: Memory, quote: str = "Anna") -> SourceReference:
    start = source.content.index(quote)
    return SourceReference(
        memory_id=source.id,
        start_offset=start,
        end_offset=start + len(quote),
        quote=quote,
    )


def test_memory_round_trips_through_json_without_losing_source_data() -> None:
    original = Memory(
        id=uuid4(),
        content="Anna said the meeting happened yesterday.",
        created_at=aware_datetime(),
        occurred_at=datetime(2026, 10, 13, 18, 0, tzinfo=UTC),
        source_type=MemorySourceType.VOICE_TRANSCRIPT,
        language=" en-US ",
    )

    restored = Memory.model_validate_json(original.model_dump_json())

    assert restored == original
    assert restored.language == "en-US"


@pytest.mark.parametrize("content", ["", "   ", "\n\t"])
def test_memory_rejects_empty_content(content: str) -> None:
    with pytest.raises(ValidationError, match="memory content must not be empty"):
        Memory(
            content=content,
            created_at=aware_datetime(),
            source_type=MemorySourceType.TEXT,
        )


@pytest.mark.parametrize("field", ["created_at", "occurred_at"])
def test_memory_rejects_naive_datetimes(field: str) -> None:
    values = {
        "content": "A memory",
        "created_at": aware_datetime(),
        "source_type": MemorySourceType.TEXT,
        field: datetime(2026, 10, 14, 12, 30),
    }

    with pytest.raises(ValidationError, match="timezone"):
        Memory(**values)


@pytest.mark.parametrize("language", ["", "english", "en_US"])
def test_memory_rejects_invalid_language_tags(language: str) -> None:
    with pytest.raises(ValidationError, match="valid language tag"):
        Memory(
            content="A memory",
            created_at=aware_datetime(),
            source_type=MemorySourceType.TEXT,
            language=language,
        )


def test_memory_original_content_is_immutable() -> None:
    original = memory()

    with pytest.raises(ValidationError, match="frozen"):
        original.content = "AI replacement"  # type: ignore[misc]


def test_optional_language_accepts_explicit_none() -> None:
    original = Memory(
        content="A memory without a known language",
        created_at=aware_datetime(),
        source_type=MemorySourceType.TEXT,
        language=None,
    )

    assert original.language is None


def test_entities_with_the_same_name_remain_distinct() -> None:
    first = Entity(
        display_name=" Alex ",
        entity_type=EntityType.PERSON,
        aliases=("Al", " al "),
    )
    second = Entity(display_name="Alex", entity_type=EntityType.PERSON)

    assert first.display_name == "Alex"
    assert first.aliases == ("Al",)
    assert first.id != second.id


@pytest.mark.parametrize("aliases", [("",), ("Alex",)])
def test_entity_rejects_invalid_aliases(aliases: tuple[str, ...]) -> None:
    with pytest.raises(ValidationError):
        Entity(
            display_name="Alex",
            entity_type=EntityType.PERSON,
            aliases=aliases,
        )


def test_entity_rejects_whitespace_display_name() -> None:
    with pytest.raises(ValidationError, match="display name"):
        Entity(display_name=" ", entity_type=EntityType.OTHER)


def test_claim_requires_evidence() -> None:
    with pytest.raises(ValidationError, match="at least 1 item"):
        Claim(
            statement="Anna plans to move.",
            evidence=(),
            origin=InformationOrigin.USER_PROVIDED,
        )


def test_claim_preserves_attribution_uncertainty_and_ambiguous_time() -> None:
    source = memory()
    anna = Entity(display_name="Anna", entity_type=EntityType.PERSON)
    claim = Claim(
        statement="Anna might move to Berlin.",
        speaker_entity_id=anna.id,
        evidence=(evidence(source, "might move to Berlin"),),
        temporal_context=TemporalContext(
            precision=TemporalPrecision.RELATIVE,
            expression="next Friday",
        ),
        certainty=Certainty.UNCERTAIN,
        origin=InformationOrigin.USER_PROVIDED,
    )

    assert claim.speaker_entity_id == anna.id
    assert claim.temporal_context.expression == "next Friday"
    assert claim.certainty is Certainty.UNCERTAIN


def test_conflicting_claims_remain_independent_reports() -> None:
    first_memory = memory("Anna said she would move to Berlin in November.")
    second_memory = memory("Anna later said she would stay in Prague.")

    claims = (
        Claim(
            statement="Anna intends to move to Berlin.",
            evidence=(evidence(first_memory, "move to Berlin"),),
            origin=InformationOrigin.USER_PROVIDED,
        ),
        Claim(
            statement="Anna intends to stay in Prague.",
            evidence=(evidence(second_memory, "stay in Prague"),),
            origin=InformationOrigin.USER_PROVIDED,
        ),
    )

    assert claims[0].evidence[0].memory_id != claims[1].evidence[0].memory_id
    assert all(claim.certainty is Certainty.UNKNOWN for claim in claims)


def test_event_preserves_planned_status_and_unresolved_month() -> None:
    source = memory("Anna told me she was moving to Berlin in November.")
    anna_id = uuid4()
    event = Event(
        title="Anna moves to Berlin",
        participant_entity_ids=(anna_id, anna_id),
        temporal_context=TemporalContext(
            precision=TemporalPrecision.MONTH,
            expression="November",
        ),
        evidence=(evidence(source, "moving to Berlin in November"),),
        status=EventStatus.PLANNED,
        origin=InformationOrigin.AI_INFERRED,
    )

    assert event.status is EventStatus.PLANNED
    assert event.participant_entity_ids == (anna_id,)
    assert event.temporal_context.instant is None


def test_relationship_requires_evidence_and_distinct_entities() -> None:
    source_id = uuid4()
    target_id = uuid4()

    with pytest.raises(ValidationError, match="at least 1 item"):
        Relationship(
            source_entity_id=source_id,
            target_entity_id=target_id,
            relationship_type="colleague",
            evidence=(),
            origin=InformationOrigin.AI_INFERRED,
        )

    with pytest.raises(ValidationError, match="source and target must differ"):
        Relationship(
            source_entity_id=source_id,
            target_entity_id=source_id,
            relationship_type="colleague",
            evidence=(SourceReference(memory_id=uuid4()),),
            origin=InformationOrigin.USER_PROVIDED,
        )


def test_relationship_records_whether_it_was_inferred() -> None:
    source = memory("Anna and Jo work on the same team.")
    relationship = Relationship(
        source_entity_id=uuid4(),
        target_entity_id=uuid4(),
        relationship_type="works_with",
        evidence=(evidence(source, "work on the same team"),),
        origin=InformationOrigin.AI_INFERRED,
    )

    assert relationship.origin is InformationOrigin.AI_INFERRED


@pytest.mark.parametrize(
    ("start", "end", "quote"),
    [
        (0, None, None),
        (None, 4, None),
        (4, 4, None),
        (5, 4, None),
        (-1, 4, None),
        (0, -1, None),
        (0, 4, "too long"),
        (None, None, ""),
    ],
)
def test_source_reference_rejects_inconsistent_ranges(
    start: int | None,
    end: int | None,
    quote: str | None,
) -> None:
    with pytest.raises(ValidationError):
        SourceReference(
            memory_id=uuid4(),
            start_offset=start,
            end_offset=end,
            quote=quote,
        )


def test_source_reference_validates_quote_against_memory() -> None:
    source = memory()
    reference = evidence(source, "Berlin")

    validate_source_reference(reference, source)

    with pytest.raises(ValueError, match="different memory"):
        validate_source_reference(reference, memory())

    with pytest.raises(ValueError, match="does not match"):
        validate_source_reference(
            reference.model_copy(update={"quote": "Friday"}),
            source,
        )

    with pytest.raises(ValueError, match="exceed"):
        validate_source_reference(
            SourceReference(
                memory_id=source.id,
                start_offset=len(source.content),
                end_offset=len(source.content) + 1,
                quote="x",
            ),
            source,
        )


def test_source_reference_without_offsets_requires_quote_to_occur() -> None:
    source = memory()
    validate_source_reference(
        SourceReference(memory_id=source.id, quote="Berlin"),
        source,
    )

    with pytest.raises(ValueError, match="does not occur"):
        validate_source_reference(
            SourceReference(memory_id=source.id, quote="Paris"),
            source,
        )


@pytest.mark.parametrize(
    "context",
    [
        {"precision": TemporalPrecision.EXACT},
        {
            "precision": TemporalPrecision.MONTH,
            "instant": aware_datetime(),
            "expression": "November 2026",
        },
        {"precision": TemporalPrecision.MONTH},
        {"precision": TemporalPrecision.MONTH, "expression": " "},
        {"precision": TemporalPrecision.UNKNOWN, "expression": "unknown"},
    ],
)
def test_temporal_context_rejects_inconsistent_representations(
    context: dict[str, object],
) -> None:
    with pytest.raises(ValidationError):
        TemporalContext(**context)


def test_exact_and_unknown_temporal_contexts_are_explicit() -> None:
    exact = TemporalContext(
        precision=TemporalPrecision.EXACT,
        instant=aware_datetime(),
        expression="2026-10-14 at 12:30 UTC",
    )
    unknown = TemporalContext(precision=TemporalPrecision.UNKNOWN)

    assert exact.instant == aware_datetime()
    assert unknown.expression is None


def test_temporal_context_accepts_an_explicitly_absent_expression() -> None:
    context = TemporalContext(
        precision=TemporalPrecision.UNKNOWN,
        expression=None,
    )

    assert context.expression is None


@pytest.mark.parametrize(
    ("model", "field", "invalid_value"),
    [
        (Memory, "source_type", "email"),
        (Entity, "entity_type", "animal"),
        (Event, "status", "confirmed"),
        (Relationship, "origin", "verified"),
    ],
)
def test_models_reject_invalid_enum_values(
    model: type[Memory] | type[Entity] | type[Event] | type[Relationship],
    field: str,
    invalid_value: str,
) -> None:
    source = memory()
    values: dict[str, object] = {
        Memory: {
            "content": "A memory",
            "created_at": aware_datetime(),
            "source_type": MemorySourceType.TEXT,
        },
        Entity: {"display_name": "Alex", "entity_type": EntityType.PERSON},
        Event: {
            "title": "A meeting",
            "evidence": (evidence(source),),
            "status": EventStatus.REPORTED,
            "origin": InformationOrigin.USER_PROVIDED,
        },
        Relationship: {
            "source_entity_id": uuid4(),
            "target_entity_id": uuid4(),
            "relationship_type": "knows",
            "evidence": (evidence(source),),
            "origin": InformationOrigin.USER_PROVIDED,
        },
    }[model]
    values[field] = invalid_value

    with pytest.raises(ValidationError):
        model(**values)


def test_models_reject_malformed_uuids() -> None:
    with pytest.raises(ValidationError, match="UUID"):
        SourceReference(memory_id="not-a-uuid")


def test_nested_claim_round_trips_through_json() -> None:
    source = memory("Anna said she might move to Berlin next Friday.")
    claim = Claim(
        statement="Anna might move to Berlin.",
        evidence=(evidence(source, "might move to Berlin"),),
        temporal_context=TemporalContext(
            precision=TemporalPrecision.RELATIVE,
            expression="next Friday",
        ),
        certainty=Certainty.UNCERTAIN,
        origin=InformationOrigin.AI_INFERRED,
    )

    restored = Claim.model_validate_json(claim.model_dump_json())

    assert restored == claim
    assert isinstance(restored.id, UUID)


@pytest.mark.parametrize(
    ("model", "values", "message"),
    [
        (
            Claim,
            {
                "statement": " ",
                "evidence": (SourceReference(memory_id=uuid4()),),
                "origin": InformationOrigin.AI_INFERRED,
            },
            "claim statement",
        ),
        (
            Event,
            {
                "title": " ",
                "evidence": (SourceReference(memory_id=uuid4()),),
                "status": EventStatus.UNCERTAIN,
                "origin": InformationOrigin.AI_INFERRED,
            },
            "event title",
        ),
        (
            Relationship,
            {
                "source_entity_id": uuid4(),
                "target_entity_id": uuid4(),
                "relationship_type": " ",
                "evidence": (SourceReference(memory_id=uuid4()),),
                "origin": InformationOrigin.AI_INFERRED,
            },
            "relationship type",
        ),
    ],
)
def test_interpretation_models_reject_blank_text(
    model: type[Claim] | type[Event] | type[Relationship],
    values: dict[str, object],
    message: str,
) -> None:
    with pytest.raises(ValidationError, match=message):
        model(**values)
