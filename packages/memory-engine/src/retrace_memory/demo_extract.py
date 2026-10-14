import json
import os
import sys
from datetime import UTC, datetime

from retrace_memory.extraction import ExtractionError, MemoryExtractor
from retrace_memory.models import Memory, MemorySourceType
from retrace_memory.ollama import OllamaClient, OllamaConfig, OllamaError


def _read_stdin() -> str:
    buffer = getattr(sys.stdin, "buffer", None)
    if buffer is None:
        return sys.stdin.read()
    return buffer.read().decode("utf-8-sig")


def main() -> int:
    try:
        content = _read_stdin()
    except UnicodeDecodeError:
        print("Memory input must be UTF-8 text.", file=sys.stderr)
        return 2
    if not content.strip():
        print("Provide memory text on standard input.", file=sys.stderr)
        return 2

    try:
        config = OllamaConfig.from_env()
        debug = os.getenv("RETRACE_EXTRACTION_DEBUG", "").casefold() in {
            "1",
            "true",
            "yes",
            "on",
        }
        result = MemoryExtractor(
            OllamaClient(config, debug=debug), debug=debug
        ).extract(
            Memory(
                content=content,
                created_at=datetime.now(UTC),
                source_type=MemorySourceType.TEXT,
            )
        )
    except (OllamaError, ExtractionError) as error:
        print(str(error), file=sys.stderr)
        return 1

    print(json.dumps(result.model_dump(mode="json"), indent=2))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
