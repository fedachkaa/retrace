import json
import sqlite3
from collections.abc import Iterable
from pathlib import Path
from typing import Literal, Self
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field

from retrace_memory.extraction import ExtractionDiagnostic, ExtractionResult
from retrace_memory.models import (
    Certainty,
    Claim,
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
    validate_source_reference,
)

SCHEMA_VERSION = 1


class StoreError(RuntimeError):
    pass


class MissingMemoryError(StoreError):
    pass


class DuplicateRecordError(StoreError):
    pass


class InvalidExtractionError(StoreError):
    pass


class DatabaseError(StoreError):
    pass


class UnsupportedSchemaVersionError(StoreError):
    pass


class MemoryConnection(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    memory_id: UUID
    connected_memory_id: UUID
    basis: Literal["shared_entity_id"] = "shared_entity_id"
    shared_entity_ids: tuple[UUID, ...] = Field(min_length=1)


class TimelineEntry(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    memory_id: UUID
    event: Event


class TimelineResult(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    exact: tuple[TimelineEntry, ...] = ()
    ambiguous: tuple[TimelineEntry, ...] = ()


_SCHEMA = """
CREATE TABLE schema_info (
    version INTEGER NOT NULL
);
CREATE TABLE memories (
    id TEXT PRIMARY KEY,
    content TEXT NOT NULL,
    created_at TEXT NOT NULL,
    occurred_at TEXT,
    source_type TEXT NOT NULL,
    language TEXT
);
CREATE TABLE extractions (
    memory_id TEXT PRIMARY KEY REFERENCES memories(id) ON DELETE CASCADE
);
CREATE TABLE entities (
    id TEXT PRIMARY KEY,
    display_name TEXT NOT NULL,
    entity_type TEXT NOT NULL,
    aliases_json TEXT NOT NULL
);
CREATE TABLE extraction_entities (
    memory_id TEXT NOT NULL REFERENCES extractions(memory_id) ON DELETE CASCADE,
    entity_id TEXT NOT NULL REFERENCES entities(id),
    ordinal INTEGER NOT NULL,
    PRIMARY KEY (memory_id, entity_id),
    UNIQUE (memory_id, ordinal)
);
CREATE TABLE claims (
    id TEXT PRIMARY KEY,
    memory_id TEXT NOT NULL REFERENCES extractions(memory_id) ON DELETE CASCADE,
    ordinal INTEGER NOT NULL,
    statement TEXT NOT NULL,
    speaker_entity_id TEXT,
    temporal_precision TEXT,
    temporal_instant TEXT,
    temporal_expression TEXT,
    certainty TEXT NOT NULL,
    origin TEXT NOT NULL,
    FOREIGN KEY (memory_id, speaker_entity_id)
        REFERENCES extraction_entities(memory_id, entity_id)
);
CREATE TABLE claim_evidence (
    claim_id TEXT NOT NULL REFERENCES claims(id) ON DELETE CASCADE,
    ordinal INTEGER NOT NULL,
    memory_id TEXT NOT NULL REFERENCES memories(id),
    scope TEXT NOT NULL,
    start_offset INTEGER,
    end_offset INTEGER,
    quote TEXT,
    PRIMARY KEY (claim_id, ordinal)
);
CREATE TABLE events (
    id TEXT PRIMARY KEY,
    memory_id TEXT NOT NULL REFERENCES extractions(memory_id) ON DELETE CASCADE,
    ordinal INTEGER NOT NULL,
    title TEXT NOT NULL,
    temporal_precision TEXT,
    temporal_instant TEXT,
    temporal_expression TEXT,
    status TEXT NOT NULL,
    origin TEXT NOT NULL
);
CREATE TABLE event_participants (
    event_id TEXT NOT NULL REFERENCES events(id) ON DELETE CASCADE,
    ordinal INTEGER NOT NULL,
    memory_id TEXT NOT NULL,
    entity_id TEXT NOT NULL,
    PRIMARY KEY (event_id, ordinal),
    FOREIGN KEY (memory_id, entity_id)
        REFERENCES extraction_entities(memory_id, entity_id)
);
CREATE TABLE event_evidence (
    event_id TEXT NOT NULL REFERENCES events(id) ON DELETE CASCADE,
    ordinal INTEGER NOT NULL,
    memory_id TEXT NOT NULL REFERENCES memories(id),
    scope TEXT NOT NULL,
    start_offset INTEGER,
    end_offset INTEGER,
    quote TEXT,
    PRIMARY KEY (event_id, ordinal)
);
CREATE TABLE extraction_diagnostics (
    memory_id TEXT NOT NULL REFERENCES extractions(memory_id) ON DELETE CASCADE,
    ordinal INTEGER NOT NULL,
    item_type TEXT NOT NULL,
    item_index INTEGER NOT NULL,
    code TEXT NOT NULL,
    message TEXT NOT NULL,
    PRIMARY KEY (memory_id, ordinal)
);
CREATE INDEX idx_memories_created_at ON memories(created_at DESC, id);
CREATE INDEX idx_extraction_entities_entity
    ON extraction_entities(entity_id, memory_id);
CREATE INDEX idx_events_exact_timeline
    ON events(temporal_instant, id)
    WHERE temporal_precision = 'exact';
"""


class MemoryStore:
    def __init__(self, database_path: str | Path) -> None:
        self.database_path = Path(database_path)
        try:
            self._connection = sqlite3.connect(self.database_path)
            self._connection.row_factory = sqlite3.Row
            self._connection.execute("PRAGMA foreign_keys = ON")
            enabled = self._connection.execute("PRAGMA foreign_keys").fetchone()[0]
            if enabled != 1:
                raise DatabaseError(
                    "SQLite foreign key enforcement could not be enabled"
                )
            self._initialize_schema()
        except StoreError:
            self._connection.close()
            raise
        except sqlite3.Error as error:
            raise DatabaseError("SQLite store initialization failed") from error

    def __enter__(self) -> Self:
        return self

    def __exit__(self, *args: object) -> None:
        self.close()

    def close(self) -> None:
        self._connection.close()

    @property
    def schema_version(self) -> int:
        return self._connection.execute("PRAGMA user_version").fetchone()[0]

    @property
    def foreign_keys_enabled(self) -> bool:
        return self._connection.execute("PRAGMA foreign_keys").fetchone()[0] == 1

    def _initialize_schema(self) -> None:
        version = self.schema_version
        if version > SCHEMA_VERSION:
            raise UnsupportedSchemaVersionError(
                f"database schema version {version} is newer than supported version "
                f"{SCHEMA_VERSION}"
            )
        if version == 0:
            with self._connection:
                self._connection.executescript(_SCHEMA)
                self._connection.execute(
                    "INSERT INTO schema_info(version) VALUES (?)", (SCHEMA_VERSION,)
                )
                self._connection.execute(f"PRAGMA user_version = {SCHEMA_VERSION}")

    def save_memory(self, memory: Memory) -> None:
        try:
            with self._connection:
                self._connection.execute(
                    """
                    INSERT INTO memories(
                        id, content, created_at, occurred_at, source_type, language
                    ) VALUES (?, ?, ?, ?, ?, ?)
                    """,
                    (
                        str(memory.id),
                        memory.content,
                        memory.created_at.isoformat(),
                        memory.occurred_at.isoformat() if memory.occurred_at else None,
                        memory.source_type.value,
                        memory.language,
                    ),
                )
        except sqlite3.IntegrityError as error:
            raise DuplicateRecordError("memory already exists") from error
        except sqlite3.Error as error:
            raise DatabaseError("saving memory failed") from error

    def get_memory(self, memory_id: UUID) -> Memory:
        row = self._connection.execute(
            "SELECT * FROM memories WHERE id = ?", (str(memory_id),)
        ).fetchone()
        if row is None:
            raise MissingMemoryError("memory does not exist")
        return Memory(
            id=row["id"],
            content=row["content"],
            created_at=row["created_at"],
            occurred_at=row["occurred_at"],
            source_type=MemorySourceType(row["source_type"]),
            language=row["language"],
        )

    def list_memories(self, limit: int = 50) -> tuple[Memory, ...]:
        if limit <= 0:
            raise ValueError("memory list limit must be greater than zero")
        rows = self._connection.execute(
            "SELECT id FROM memories ORDER BY created_at DESC, id LIMIT ?", (limit,)
        ).fetchall()
        return tuple(self.get_memory(UUID(row["id"])) for row in rows)

    def save_extraction(self, result: ExtractionResult) -> None:
        if self._has_extraction(result.memory_id):
            raise DuplicateRecordError("an extraction snapshot already exists")
        self._validate_extraction(result)
        self._write_extraction(result)

    def replace_extraction(self, result: ExtractionResult) -> None:
        if not self._has_extraction(result.memory_id):
            raise MissingMemoryError("extraction snapshot does not exist")
        self._validate_extraction(result)
        try:
            with self._connection:
                self._connection.execute(
                    "DELETE FROM extractions WHERE memory_id = ?",
                    (str(result.memory_id),),
                )
                self._insert_extraction(result)
                self._remove_orphan_entities()
        except InvalidExtractionError:
            raise
        except sqlite3.IntegrityError as error:
            raise DuplicateRecordError(
                "extraction contains conflicting records"
            ) from error
        except sqlite3.Error as error:
            raise DatabaseError("replacing extraction failed") from error

    def get_extraction(self, memory_id: UUID) -> ExtractionResult:
        if not self._has_extraction(memory_id):
            raise MissingMemoryError("extraction snapshot does not exist")
        entities = self._read_entities(memory_id)
        claims = self._read_claims(memory_id)
        events = self._read_events(memory_id)
        diagnostics = self._read_diagnostics(memory_id)
        return ExtractionResult(
            memory_id=memory_id,
            entities=entities,
            claims=claims,
            events=events,
            diagnostics=diagnostics,
        )

    def list_connections(self, memory_id: UUID) -> tuple[MemoryConnection, ...]:
        self.get_memory(memory_id)
        rows = self._connection.execute(
            """
            SELECT other.memory_id AS connected_memory_id, current.entity_id
            FROM extraction_entities AS current
            JOIN extraction_entities AS other
              ON other.entity_id = current.entity_id
             AND other.memory_id <> current.memory_id
            WHERE current.memory_id = ?
            ORDER BY other.memory_id, current.entity_id
            """,
            (str(memory_id),),
        ).fetchall()
        grouped: dict[UUID, list[UUID]] = {}
        for row in rows:
            grouped.setdefault(UUID(row["connected_memory_id"]), []).append(
                UUID(row["entity_id"])
            )
        return tuple(
            MemoryConnection(
                memory_id=memory_id,
                connected_memory_id=connected_id,
                shared_entity_ids=tuple(entity_ids),
            )
            for connected_id, entity_ids in grouped.items()
        )

    def get_timeline(self) -> TimelineResult:
        rows = self._connection.execute(
            """
            SELECT memory_id
            FROM extractions
            ORDER BY memory_id
            """
        ).fetchall()
        exact: list[TimelineEntry] = []
        ambiguous: list[TimelineEntry] = []
        for row in rows:
            result = self.get_extraction(UUID(row["memory_id"]))
            for event in result.events:
                entry = TimelineEntry(memory_id=result.memory_id, event=event)
                if (
                    event.temporal_context is not None
                    and event.temporal_context.precision is TemporalPrecision.EXACT
                ):
                    exact.append(entry)
                else:
                    ambiguous.append(entry)
        exact.sort(
            key=lambda entry: (
                entry.event.temporal_context.instant,
                str(entry.event.id),
            )
        )
        ambiguous.sort(key=lambda entry: (str(entry.memory_id), str(entry.event.id)))
        return TimelineResult(exact=tuple(exact), ambiguous=tuple(ambiguous))

    def _has_extraction(self, memory_id: UUID) -> bool:
        return (
            self._connection.execute(
                "SELECT 1 FROM extractions WHERE memory_id = ?", (str(memory_id),)
            ).fetchone()
            is not None
        )

    def _validate_extraction(self, result: ExtractionResult) -> None:
        try:
            memory = self.get_memory(result.memory_id)
        except MissingMemoryError as error:
            raise InvalidExtractionError("extraction memory does not exist") from error
        entity_ids = [entity.id for entity in result.entities]
        if len(entity_ids) != len(set(entity_ids)):
            raise InvalidExtractionError("extraction contains duplicate entity IDs")
        known_entities = set(entity_ids)
        for claim in result.claims:
            if (
                claim.speaker_entity_id is not None
                and claim.speaker_entity_id not in known_entities
            ):
                raise InvalidExtractionError("claim references an unknown entity")
            self._validate_evidence(claim.evidence, result.memory_id, memory)
        for event in result.events:
            if any(
                entity_id not in known_entities
                for entity_id in event.participant_entity_ids
            ):
                raise InvalidExtractionError("event references an unknown entity")
            self._validate_evidence(event.evidence, result.memory_id, memory)

    @staticmethod
    def _validate_evidence(
        references: Iterable[SourceReference], memory_id: UUID, memory: Memory
    ) -> None:
        for reference in references:
            if reference.memory_id != memory_id:
                raise InvalidExtractionError("evidence references a different memory")
            try:
                validate_source_reference(reference, memory)
            except ValueError as error:
                raise InvalidExtractionError(
                    "extraction contains invalid evidence"
                ) from error

    def _write_extraction(self, result: ExtractionResult) -> None:
        try:
            with self._connection:
                self._insert_extraction(result)
        except InvalidExtractionError:
            raise
        except sqlite3.IntegrityError as error:
            raise DuplicateRecordError(
                "extraction contains conflicting records"
            ) from error
        except sqlite3.Error as error:
            raise DatabaseError("saving extraction failed") from error

    def _insert_extraction(self, result: ExtractionResult) -> None:
        memory_id = str(result.memory_id)
        self._connection.execute(
            "INSERT INTO extractions(memory_id) VALUES (?)", (memory_id,)
        )
        for index, entity in enumerate(result.entities):
            self._insert_entity(memory_id, index, entity)
        for index, claim in enumerate(result.claims):
            temporal = self._temporal_values(claim.temporal_context)
            self._connection.execute(
                """
                INSERT INTO claims(
                    id, memory_id, ordinal, statement, speaker_entity_id,
                    temporal_precision, temporal_instant, temporal_expression,
                    certainty, origin
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    str(claim.id),
                    memory_id,
                    index,
                    claim.statement,
                    str(claim.speaker_entity_id) if claim.speaker_entity_id else None,
                    *temporal,
                    claim.certainty.value,
                    claim.origin.value,
                ),
            )
            self._insert_evidence(
                "claim_evidence", "claim_id", claim.id, claim.evidence
            )
        for index, event in enumerate(result.events):
            temporal = self._temporal_values(event.temporal_context)
            self._connection.execute(
                """
                INSERT INTO events(
                    id, memory_id, ordinal, title, temporal_precision, temporal_instant,
                    temporal_expression, status, origin
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    str(event.id),
                    memory_id,
                    index,
                    event.title,
                    *temporal,
                    event.status.value,
                    event.origin.value,
                ),
            )
            self._connection.executemany(
                """
                INSERT INTO event_participants(
                    event_id, ordinal, memory_id, entity_id
                ) VALUES (?, ?, ?, ?)
                """,
                (
                    (str(event.id), index, memory_id, str(entity_id))
                    for index, entity_id in enumerate(event.participant_entity_ids)
                ),
            )
            self._insert_evidence(
                "event_evidence", "event_id", event.id, event.evidence
            )
        self._connection.executemany(
            """
            INSERT INTO extraction_diagnostics(
                memory_id, ordinal, item_type, item_index, code, message
            ) VALUES (?, ?, ?, ?, ?, ?)
            """,
            (
                (
                    memory_id,
                    index,
                    diagnostic.item_type,
                    diagnostic.item_index,
                    diagnostic.code,
                    diagnostic.message,
                )
                for index, diagnostic in enumerate(result.diagnostics)
            ),
        )

    def _insert_entity(self, memory_id: str, ordinal: int, entity: Entity) -> None:
        stored = self._connection.execute(
            "SELECT * FROM entities WHERE id = ?", (str(entity.id),)
        ).fetchone()
        values = (
            entity.display_name,
            entity.entity_type.value,
            json.dumps(entity.aliases),
        )
        if stored is None:
            self._connection.execute(
                """
                INSERT INTO entities(id, display_name, entity_type, aliases_json)
                VALUES (?, ?, ?, ?)
                """,
                (str(entity.id), *values),
            )
        elif (
            stored["display_name"],
            stored["entity_type"],
            stored["aliases_json"],
        ) != values:
            raise InvalidExtractionError("shared entity ID has conflicting data")
        self._connection.execute(
            """
            INSERT INTO extraction_entities(memory_id, entity_id, ordinal)
            VALUES (?, ?, ?)
            """,
            (memory_id, str(entity.id), ordinal),
        )

    def _insert_evidence(
        self,
        table: Literal["claim_evidence", "event_evidence"],
        owner_column: Literal["claim_id", "event_id"],
        owner_id: UUID,
        evidence: tuple[SourceReference, ...],
    ) -> None:
        statement = f"""
            INSERT INTO {table}(
                {owner_column}, ordinal, memory_id, scope,
                start_offset, end_offset, quote
            ) VALUES (?, ?, ?, ?, ?, ?, ?)
        """
        self._connection.executemany(
            statement,
            (
                (
                    str(owner_id),
                    index,
                    str(reference.memory_id),
                    reference.scope.value,
                    reference.start_offset,
                    reference.end_offset,
                    reference.quote,
                )
                for index, reference in enumerate(evidence)
            ),
        )

    @staticmethod
    def _temporal_values(
        temporal: TemporalContext | None,
    ) -> tuple[str | None, str | None, str | None]:
        if temporal is None:
            return None, None, None
        return (
            temporal.precision.value,
            temporal.instant.isoformat() if temporal.instant else None,
            temporal.expression,
        )

    def _read_entities(self, memory_id: UUID) -> tuple[Entity, ...]:
        rows = self._connection.execute(
            """
            SELECT entity.*
            FROM entities AS entity
            JOIN extraction_entities AS link ON link.entity_id = entity.id
            WHERE link.memory_id = ?
            ORDER BY link.ordinal
            """,
            (str(memory_id),),
        ).fetchall()
        return tuple(
            Entity(
                id=row["id"],
                display_name=row["display_name"],
                entity_type=EntityType(row["entity_type"]),
                aliases=tuple(json.loads(row["aliases_json"])),
            )
            for row in rows
        )

    def _read_claims(self, memory_id: UUID) -> tuple[Claim, ...]:
        rows = self._connection.execute(
            "SELECT * FROM claims WHERE memory_id = ? ORDER BY ordinal",
            (str(memory_id),),
        ).fetchall()
        return tuple(
            Claim(
                id=row["id"],
                statement=row["statement"],
                speaker_entity_id=row["speaker_entity_id"],
                evidence=self._read_evidence("claim_evidence", "claim_id", row["id"]),
                temporal_context=self._read_temporal(row),
                certainty=Certainty(row["certainty"]),
                origin=InformationOrigin(row["origin"]),
            )
            for row in rows
        )

    def _read_events(self, memory_id: UUID) -> tuple[Event, ...]:
        rows = self._connection.execute(
            "SELECT * FROM events WHERE memory_id = ? ORDER BY ordinal",
            (str(memory_id),),
        ).fetchall()
        events: list[Event] = []
        for row in rows:
            participants = self._connection.execute(
                """
                SELECT entity_id FROM event_participants
                WHERE event_id = ? ORDER BY ordinal
                """,
                (row["id"],),
            ).fetchall()
            events.append(
                Event(
                    id=row["id"],
                    title=row["title"],
                    participant_entity_ids=tuple(
                        UUID(item["entity_id"]) for item in participants
                    ),
                    temporal_context=self._read_temporal(row),
                    evidence=self._read_evidence(
                        "event_evidence", "event_id", row["id"]
                    ),
                    status=EventStatus(row["status"]),
                    origin=InformationOrigin(row["origin"]),
                )
            )
        return tuple(events)

    def _read_evidence(
        self,
        table: Literal["claim_evidence", "event_evidence"],
        owner_column: Literal["claim_id", "event_id"],
        owner_id: str,
    ) -> tuple[SourceReference, ...]:
        rows = self._connection.execute(
            f"SELECT * FROM {table} WHERE {owner_column} = ? ORDER BY ordinal",
            (owner_id,),
        ).fetchall()
        return tuple(
            SourceReference(
                memory_id=row["memory_id"],
                scope=EvidenceScope(row["scope"]),
                start_offset=row["start_offset"],
                end_offset=row["end_offset"],
                quote=row["quote"],
            )
            for row in rows
        )

    @staticmethod
    def _read_temporal(row: sqlite3.Row) -> TemporalContext | None:
        if row["temporal_precision"] is None:
            return None
        return TemporalContext(
            precision=TemporalPrecision(row["temporal_precision"]),
            instant=row["temporal_instant"],
            expression=row["temporal_expression"],
        )

    def _read_diagnostics(self, memory_id: UUID) -> tuple[ExtractionDiagnostic, ...]:
        rows = self._connection.execute(
            """
            SELECT * FROM extraction_diagnostics
            WHERE memory_id = ? ORDER BY ordinal
            """,
            (str(memory_id),),
        ).fetchall()
        return tuple(
            ExtractionDiagnostic(
                item_type=row["item_type"],
                item_index=row["item_index"],
                code=row["code"],
                message=row["message"],
            )
            for row in rows
        )

    def _remove_orphan_entities(self) -> None:
        self._connection.execute(
            """
            DELETE FROM entities
            WHERE NOT EXISTS (
                SELECT 1 FROM extraction_entities
                WHERE extraction_entities.entity_id = entities.id
            )
            """
        )
