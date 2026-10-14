import json

import httpx
import pytest

from retrace_memory import (
    OllamaClient,
    OllamaConfig,
    OllamaConfigurationError,
    OllamaResponseError,
    OllamaTimeoutError,
    OllamaUnavailableError,
)


def test_ollama_client_requests_structured_non_streaming_output() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content)
        assert request.url == "http://127.0.0.1:11434/api/chat"
        assert body == {
            "model": "local-model",
            "stream": False,
            "format": {"type": "object"},
            "options": {"temperature": 0},
            "messages": [
                {"role": "system", "content": "system"},
                {"role": "user", "content": "user"},
            ],
        }
        return httpx.Response(200, json={"message": {"content": '{"ok":true}'}})

    client = OllamaClient(
        OllamaConfig(model="local-model"),
        transport=httpx.MockTransport(handler),
    )

    assert (
        client.complete(
            system_prompt="system",
            user_prompt="user",
            response_schema={"type": "object"},
        )
        == '{"ok":true}'
    )


def test_ollama_debug_mode_reports_selected_model_and_raw_response(
    capsys: pytest.CaptureFixture[str],
) -> None:
    transport = httpx.MockTransport(
        lambda request: httpx.Response(
            200, json={"message": {"content": '{"entities":[]}'}}
        )
    )

    OllamaClient(OllamaConfig(model="debug-model"), transport, debug=True).complete(
        system_prompt="system", user_prompt="user", response_schema={}
    )

    output = capsys.readouterr().err
    assert "selected model: debug-model" in output
    assert 'raw Ollama response: {"entities":[]}' in output


@pytest.mark.parametrize(
    ("values", "message"),
    [
        ({"base_url": "https://example.com"}, "loopback"),
        ({"base_url": "ftp://127.0.0.1"}, "loopback"),
        ({"model": "  "}, "model name"),
        ({"timeout_seconds": 0}, "greater than zero"),
    ],
)
def test_ollama_config_rejects_unsafe_or_invalid_values(
    values: dict[str, object], message: str
) -> None:
    with pytest.raises(OllamaConfigurationError, match=message):
        OllamaConfig(**values)


def test_ollama_config_reads_environment(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("RETRACE_OLLAMA_MODEL", "qwen-local")
    monkeypatch.setenv("RETRACE_OLLAMA_BASE_URL", "http://localhost:1234")
    monkeypatch.setenv("RETRACE_OLLAMA_TIMEOUT_SECONDS", "12.5")

    config = OllamaConfig.from_env()

    assert config == OllamaConfig(
        model="qwen-local",
        base_url="http://localhost:1234",
        timeout_seconds=12.5,
    )


def test_ollama_config_rejects_non_numeric_environment_timeout(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("RETRACE_OLLAMA_TIMEOUT_SECONDS", "later")

    with pytest.raises(OllamaConfigurationError, match="must be a number"):
        OllamaConfig.from_env()


@pytest.mark.parametrize(
    ("effect", "error_type", "message"),
    [
        (
            httpx.ConnectError("offline"),
            OllamaUnavailableError,
            "Cannot reach local Ollama",
        ),
        (httpx.ReadTimeout("slow"), OllamaTimeoutError, "timed out"),
    ],
)
def test_ollama_client_translates_transport_errors(
    effect: Exception, error_type: type[Exception], message: str
) -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        raise effect

    client = OllamaClient(OllamaConfig(), httpx.MockTransport(handler))

    with pytest.raises(error_type, match=message):
        client.complete(system_prompt="s", user_prompt="u", response_schema={})


def test_ollama_client_reports_http_errors_without_exposing_response_body() -> None:
    transport = httpx.MockTransport(
        lambda request: httpx.Response(404, text="private model details")
    )

    with pytest.raises(OllamaResponseError, match="HTTP 404") as raised:
        OllamaClient(OllamaConfig(), transport).complete(
            system_prompt="s", user_prompt="u", response_schema={}
        )

    assert "private model details" not in str(raised.value)


def test_ollama_client_rejects_invalid_chat_envelopes() -> None:
    transport = httpx.MockTransport(lambda request: httpx.Response(200, json={}))

    with pytest.raises(OllamaResponseError, match="invalid chat response"):
        OllamaClient(OllamaConfig(), transport).complete(
            system_prompt="s", user_prompt="u", response_schema={}
        )
