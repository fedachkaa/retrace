import io

import pytest

from retrace_memory import ExtractionResult, OllamaConfigurationError, demo_extract


def test_demo_requires_memory_on_standard_input(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.setattr(demo_extract.sys, "stdin", io.StringIO("  "))

    assert demo_extract.main() == 2
    assert "standard input" in capsys.readouterr().err


def test_demo_reads_utf8_input_and_removes_a_bom(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class BinaryInput:
        buffer = io.BytesIO("\ufeffMünchen".encode())

    monkeypatch.setattr(demo_extract.sys, "stdin", BinaryInput())

    assert demo_extract._read_stdin() == "München"


def test_demo_rejects_non_utf8_input(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    class BinaryInput:
        buffer = io.BytesIO(b"\xff")

    monkeypatch.setattr(demo_extract.sys, "stdin", BinaryInput())

    assert demo_extract.main() == 2
    assert capsys.readouterr().err == "Memory input must be UTF-8 text.\n"


def test_demo_prints_structured_extraction(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    class SuccessfulExtractor:
        def __init__(self, client: object, debug: bool = False) -> None:
            pass

        def extract(self, memory: object) -> ExtractionResult:
            return ExtractionResult(memory_id=memory.id)

    monkeypatch.setattr(demo_extract.sys, "stdin", io.StringIO("A memory"))
    monkeypatch.setattr(demo_extract, "MemoryExtractor", SuccessfulExtractor)

    assert demo_extract.main() == 0
    assert '"entities": []' in capsys.readouterr().out


def test_demo_reports_configuration_or_extraction_errors(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.setattr(demo_extract.sys, "stdin", io.StringIO("A memory"))
    monkeypatch.setattr(
        demo_extract.OllamaConfig,
        "from_env",
        lambda: (_ for _ in ()).throw(OllamaConfigurationError("bad config")),
    )

    assert demo_extract.main() == 1
    assert capsys.readouterr().err == "bad config\n"


def test_demo_enables_debug_only_from_an_explicit_environment_value(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    received: dict[str, bool] = {}

    class DebugClient:
        def __init__(self, config: object, debug: bool = False) -> None:
            received["client"] = debug

    class DebugExtractor:
        def __init__(self, client: object, debug: bool = False) -> None:
            received["extractor"] = debug

        def extract(self, memory: object) -> ExtractionResult:
            return ExtractionResult(memory_id=memory.id)

    monkeypatch.setattr(demo_extract.sys, "stdin", io.StringIO("A memory"))
    monkeypatch.setenv("RETRACE_EXTRACTION_DEBUG", "true")
    monkeypatch.setattr(demo_extract, "OllamaClient", DebugClient)
    monkeypatch.setattr(demo_extract, "MemoryExtractor", DebugExtractor)

    assert demo_extract.main() == 0
    assert received == {"client": True, "extractor": True}
