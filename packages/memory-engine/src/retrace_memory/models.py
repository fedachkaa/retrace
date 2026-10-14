import re
from enum import StrEnum
from typing import Self
from uuid import UUID, uuid4

from pydantic import (
    AwareDatetime,
    BaseModel,
    ConfigDict,
    Field,
    field_validator,
    model_validator,
)


class MemorySourceType(StrEnum):
    TEXT = "text"
    VOICE_TRANSCRIPT = "voice_transcript"
    MANUAL_RECORD = "manual_record"


class EntityType(StrEnum):
    PERSON = "person"
    PLACE = "place"
    ORGANIZATION = "organization"
    OTHER = "other"


class InformationOrigin(StrEnum):
    USER_PROVIDED = "user_provided"
    AI_INFERRED = "ai_inferred"


class Certainty(StrEnum):
    STATED = "stated"
    UNCERTAIN = "uncertain"
    UNKNOWN = "unknown"


class EventStatus(StrEnum):
    PLANNED = "planned"
    REPORTED = "reported"
    OCCURRED = "occurred"
    UNCERTAIN = "uncertain"


class TemporalPrecision(StrEnum):
    EXACT = "exact"
    DAY = "day"
    MONTH = "month"
    YEAR = "year"
    RELATIVE = "relative"
    APPROXIMATE = "approximate"
    UNKNOWN = "unknown"


class EvidenceScope(StrEnum):
    WHOLE_MEMORY = "whole_memory"
    TEXT_SPAN = "text_span"


class Memory(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    id: UUID = Field(default_factory=uuid4)
    content: str
    created_at: AwareDatetime
    occurred_at: AwareDatetime | None = None
    source_type: MemorySourceType
    language: str | None = None

    @field_validator("content")
    @classmethod
    def validate_content(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("memory content must not be empty")
        return value

    @field_validator("language")
    @classmethod
    def validate_language(cls, value: str | None) -> str | None:
        if value is None:
            return None
        normalized = value.strip()
        if not re.fullmatch(r"[A-Za-z]{2,3}(?:-[A-Za-z0-9]{2,8})*", normalized):
            raise ValueError("language must be a valid language tag")
        return normalized


class Entity(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    id: UUID = Field(default_factory=uuid4)
    display_name: str = Field(min_length=1)
    entity_type: EntityType
    aliases: tuple[str, ...] = ()

    @field_validator("display_name")
    @classmethod
    def normalize_display_name(cls, value: str) -> str:
        normalized = value.strip()
        if not normalized:
            raise ValueError("entity display name must not be empty")
        return normalized

    @field_validator("aliases")
    @classmethod
    def normalize_aliases(cls, aliases: tuple[str, ...]) -> tuple[str, ...]:
        normalized_aliases: list[str] = []
        seen: set[str] = set()
        for alias in aliases:
            normalized = alias.strip()
            if not normalized:
                raise ValueError("entity aliases must not be empty")
            comparison_key = normalized.casefold()
            if comparison_key not in seen:
                normalized_aliases.append(normalized)
                seen.add(comparison_key)
        return tuple(normalized_aliases)

    @model_validator(mode="after")
    def reject_display_name_alias(self) -> Self:
        display_name = self.display_name.casefold()
        if any(alias.casefold() == display_name for alias in self.aliases):
            raise ValueError("an alias must differ from the display name")
        return self


class SourceReference(BaseModel):
    model_config = ConfigDict(
        frozen=True,
        extra="forbid",
        revalidate_instances="always",
    )

    memory_id: UUID
    scope: EvidenceScope
    start_offset: int | None = Field(default=None, ge=0)
    end_offset: int | None = Field(default=None, ge=0)
    quote: str | None = None

    @model_validator(mode="after")
    def validate_scope(self) -> Self:
        if self.scope is EvidenceScope.WHOLE_MEMORY:
            if any(
                value is not None
                for value in (self.start_offset, self.end_offset, self.quote)
            ):
                raise ValueError("whole-memory evidence cannot contain span details")
            return self
        if None in (self.start_offset, self.end_offset, self.quote):
            raise ValueError("text-span evidence requires offsets and a quote")
        if self.end_offset <= self.start_offset:
            raise ValueError("evidence end offset must be greater than start offset")
        if not self.quote:
            raise ValueError("evidence quote must not be empty")
        if len(self.quote) != self.end_offset - self.start_offset:
            raise ValueError("evidence quote length must match its offset range")
        return self


class TemporalContext(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    precision: TemporalPrecision
    instant: AwareDatetime | None = None
    expression: str | None = None

    @field_validator("expression")
    @classmethod
    def normalize_expression(cls, value: str | None) -> str | None:
        if value is None:
            return None
        normalized = value.strip()
        if not normalized:
            raise ValueError("temporal expression must not be empty")
        return normalized

    @model_validator(mode="after")
    def validate_representation(self) -> Self:
        if self.precision is TemporalPrecision.EXACT:
            if self.instant is None:
                raise ValueError("exact temporal context requires an instant")
            return self
        if self.instant is not None:
            raise ValueError("only exact temporal context may contain an instant")
        if self.precision is TemporalPrecision.UNKNOWN:
            if self.expression is not None:
                raise ValueError(
                    "unknown temporal context cannot contain an expression"
                )
            return self
        if self.expression is None:
            raise ValueError("imprecise temporal context requires an expression")
        return self


class Claim(BaseModel):
    model_config = ConfigDict(
        frozen=True,
        extra="forbid",
        revalidate_instances="always",
    )

    id: UUID = Field(default_factory=uuid4)
    statement: str = Field(min_length=1)
    speaker_entity_id: UUID | None = None
    evidence: tuple[SourceReference, ...] = Field(min_length=1)
    temporal_context: TemporalContext | None = None
    certainty: Certainty = Certainty.UNKNOWN
    origin: InformationOrigin

    @field_validator("statement")
    @classmethod
    def normalize_statement(cls, value: str) -> str:
        normalized = value.strip()
        if not normalized:
            raise ValueError("claim statement must not be empty")
        return normalized


class Event(BaseModel):
    model_config = ConfigDict(
        frozen=True,
        extra="forbid",
        revalidate_instances="always",
    )

    id: UUID = Field(default_factory=uuid4)
    title: str = Field(min_length=1)
    participant_entity_ids: tuple[UUID, ...] = ()
    temporal_context: TemporalContext | None = None
    evidence: tuple[SourceReference, ...] = Field(min_length=1)
    status: EventStatus
    origin: InformationOrigin

    @field_validator("title")
    @classmethod
    def normalize_title(cls, value: str) -> str:
        normalized = value.strip()
        if not normalized:
            raise ValueError("event title must not be empty")
        return normalized

    @field_validator("participant_entity_ids")
    @classmethod
    def deduplicate_participants(cls, entity_ids: tuple[UUID, ...]) -> tuple[UUID, ...]:
        return tuple(dict.fromkeys(entity_ids))


class Relationship(BaseModel):
    model_config = ConfigDict(
        frozen=True,
        extra="forbid",
        revalidate_instances="always",
    )

    id: UUID = Field(default_factory=uuid4)
    source_entity_id: UUID
    target_entity_id: UUID
    relationship_type: str = Field(min_length=1)
    evidence: tuple[SourceReference, ...] = Field(min_length=1)
    origin: InformationOrigin

    @field_validator("relationship_type")
    @classmethod
    def normalize_relationship_type(cls, value: str) -> str:
        normalized = value.strip()
        if not normalized:
            raise ValueError("relationship type must not be empty")
        return normalized

    @model_validator(mode="after")
    def reject_self_relationship(self) -> Self:
        if self.source_entity_id == self.target_entity_id:
            raise ValueError("relationship source and target must differ")
        return self


def validate_source_reference(reference: SourceReference, memory: Memory) -> None:
    reference = SourceReference.model_validate(reference)
    if reference.memory_id != memory.id:
        raise ValueError("evidence references a different memory")
    if reference.scope is EvidenceScope.WHOLE_MEMORY:
        return
    if reference.end_offset > len(memory.content):
        raise ValueError("evidence offsets exceed memory content")
    if memory.content[reference.start_offset : reference.end_offset] != reference.quote:
        raise ValueError("evidence quote does not match memory content at offsets")
