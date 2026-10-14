"""Core memory-engine package for RETRACE."""

from retrace_memory.extraction import (
    ExtractionDiagnostic,
    ExtractionError,
    ExtractionResult,
    MemoryExtractor,
)
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
    Relationship,
    SourceReference,
    TemporalContext,
    TemporalPrecision,
    validate_source_reference,
)
from retrace_memory.ollama import (
    OllamaClient,
    OllamaConfig,
    OllamaConfigurationError,
    OllamaError,
    OllamaResponseError,
    OllamaTimeoutError,
    OllamaUnavailableError,
)

__version__ = "0.1.0"

__all__ = [
    "Certainty",
    "Claim",
    "ExtractionDiagnostic",
    "ExtractionError",
    "ExtractionResult",
    "EvidenceScope",
    "Entity",
    "EntityType",
    "Event",
    "EventStatus",
    "InformationOrigin",
    "Memory",
    "MemoryExtractor",
    "MemorySourceType",
    "OllamaClient",
    "OllamaConfig",
    "OllamaConfigurationError",
    "OllamaError",
    "OllamaResponseError",
    "OllamaTimeoutError",
    "OllamaUnavailableError",
    "Relationship",
    "SourceReference",
    "TemporalContext",
    "TemporalPrecision",
    "validate_source_reference",
]
