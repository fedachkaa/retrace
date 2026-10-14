import os
import sys
from dataclasses import dataclass
from math import isfinite
from urllib.parse import urlparse

import httpx
from pydantic import BaseModel, ConfigDict, ValidationError


class OllamaError(RuntimeError):
    """Base error for failures while using the local Ollama service."""


class OllamaConfigurationError(OllamaError):
    """Raised when local Ollama configuration is invalid."""


class OllamaUnavailableError(OllamaError):
    """Raised when the configured local Ollama service cannot be reached."""


class OllamaTimeoutError(OllamaError):
    """Raised when the configured local Ollama service times out."""


class OllamaResponseError(OllamaError):
    """Raised when Ollama returns an unusable response."""


@dataclass(frozen=True)
class OllamaConfig:
    model: str = "gemma3:4b"
    base_url: str = "http://127.0.0.1:11434"
    timeout_seconds: float = 60.0

    def __post_init__(self) -> None:
        parsed = urlparse(self.base_url)
        try:
            port = parsed.port
        except ValueError as error:
            raise OllamaConfigurationError(
                "Ollama base URL has an invalid port"
            ) from error
        if parsed.scheme not in {"http", "https"} or parsed.hostname not in {
            "127.0.0.1",
            "localhost",
            "::1",
        }:
            raise OllamaConfigurationError(
                "Ollama base URL must use HTTP or HTTPS on a loopback host"
            )
        if parsed.username is not None or parsed.password is not None:
            raise OllamaConfigurationError("Ollama base URL cannot contain user info")
        if (
            parsed.path not in {"", "/"}
            or parsed.params
            or parsed.query
            or parsed.fragment
        ):
            raise OllamaConfigurationError(
                "Ollama base URL cannot contain a path, parameters, query, or fragment"
            )
        if port is not None and port == 0:
            raise OllamaConfigurationError("Ollama base URL has an invalid port")
        if not self.model.strip():
            raise OllamaConfigurationError("Ollama model name must not be empty")
        if not isfinite(self.timeout_seconds) or self.timeout_seconds <= 0:
            raise OllamaConfigurationError(
                "Ollama timeout must be a finite number greater than zero"
            )

    @classmethod
    def from_env(cls) -> "OllamaConfig":
        timeout_value = os.getenv("RETRACE_OLLAMA_TIMEOUT_SECONDS", "60")
        try:
            timeout_seconds = float(timeout_value)
        except ValueError as error:
            raise OllamaConfigurationError(
                "RETRACE_OLLAMA_TIMEOUT_SECONDS must be a number"
            ) from error
        return cls(
            model=os.getenv("RETRACE_OLLAMA_MODEL", "gemma3:4b"),
            base_url=os.getenv("RETRACE_OLLAMA_BASE_URL", "http://127.0.0.1:11434"),
            timeout_seconds=timeout_seconds,
        )


class _Message(BaseModel):
    model_config = ConfigDict(extra="ignore")

    content: str


class _ChatResponse(BaseModel):
    model_config = ConfigDict(extra="ignore")

    message: _Message


class OllamaClient:
    def __init__(
        self,
        config: OllamaConfig,
        transport: httpx.BaseTransport | None = None,
        debug: bool = False,
    ) -> None:
        self.config = config
        self.transport = transport
        self.debug = debug

    def complete(
        self,
        *,
        system_prompt: str,
        user_prompt: str,
        response_schema: dict[str, object],
    ) -> str:
        request = {
            "model": self.config.model,
            "stream": False,
            "format": response_schema,
            "options": {"temperature": 0},
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
        }
        if self.debug:
            print(
                f"[RETRACE DEBUG] selected model: {self.config.model}",
                file=sys.stderr,
            )
        try:
            with httpx.Client(
                base_url=self.config.base_url,
                timeout=self.config.timeout_seconds,
                transport=self.transport,
                trust_env=False,
                follow_redirects=False,
            ) as client:
                response = client.post("/api/chat", json=request)
                response.raise_for_status()
        except httpx.TimeoutException as error:
            raise OllamaTimeoutError(
                "Local Ollama timed out; verify the model is available and try again"
            ) from error
        except httpx.HTTPStatusError as error:
            raise OllamaResponseError(
                f"Local Ollama returned HTTP {error.response.status_code}"
            ) from error
        except httpx.RequestError as error:
            raise OllamaUnavailableError(
                "Cannot reach local Ollama; start Ollama and verify the loopback URL"
            ) from error

        try:
            content = _ChatResponse.model_validate_json(
                response.content
            ).message.content
        except ValidationError as error:
            raise OllamaResponseError(
                "Local Ollama returned an invalid chat response"
            ) from error
        if self.debug:
            print(f"[RETRACE DEBUG] raw Ollama response: {content}", file=sys.stderr)
        return content
