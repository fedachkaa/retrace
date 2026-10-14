# Local AI memory extraction

RETRACE can turn one in-memory `Memory` into evidence-backed entities, claims, and
events by calling a locally running [Ollama](https://ollama.com/). This first version
does not persist data, call cloud services, download models, or expose an API endpoint.

## Setup

1. Install Ollama using its platform instructions and start it locally.
2. Install a model yourself, for example `ollama pull gemma3:4b`.
3. Configure RETRACE when using another installed model:

```shell
set RETRACE_OLLAMA_MODEL=gemma3:4b
set RETRACE_OLLAMA_BASE_URL=http://127.0.0.1:11434
set RETRACE_OLLAMA_TIMEOUT_SECONDS=60
```

The endpoint is deliberately restricted to `localhost`, `127.0.0.1`, or `::1`.
RETRACE will not send memory text to a remote Ollama host. The model name, loopback
URL, and positive timeout may also be supplied with `OllamaConfig` in Python.

Pass memory text on standard input to run the demo without placing personal text in
command-line arguments:

```shell
uv run python -m retrace_memory.demo_extract < my-memory.txt
```

The command prints structured JSON. It exits with a concise setup or model-response
error when Ollama is unavailable, times out, returns an HTTP error, or returns data
outside the extraction schema.

On Windows, the demo reads standard input as UTF-8 and accepts an optional UTF-8 BOM.
Other encodings are rejected instead of silently corrupting memory text.

### Synthetic-input debugging

Debugging is disabled by default. For synthetic test text only, enable it explicitly:

```shell
set RETRACE_EXTRACTION_DEBUG=1
uv run python -m retrace_memory.demo_extract < synthetic-memory.txt
```

Debug output goes to standard error and includes the selected model, raw structured
Ollama response, parsed extraction payload, accepted records, and rejected records
with their reasons. Raw model output can contain the supplied memory and extracted
personal information, so never enable this mode for private memories.

## Trust and evidence boundaries

Memory content is JSON-encoded into a user message and the system prompt explicitly
treats it as untrusted data rather than instructions. This reduces prompt-injection
risk but cannot guarantee that a local model follows the instruction.

The model can propose display names, local references, statements, event titles,
temporal expressions, and exact evidence quotes. It cannot supply domain UUIDs,
memory IDs, evidence offsets, whole-memory references, origin values, or exact
timestamps. Python creates UUIDs, resolves local references, assigns
`ai_inferred`, and calculates offsets from the original immutable memory.

A claim or event is accepted only when its evidence quote occurs exactly once as a
contiguous, case-sensitive substring. Missing, fabricated, or repeated quotes are
rejected with a diagnostic; they never fall back to whole-memory evidence. Repeated
quotes are rejected rather than resolved heuristically because the model schema has
no trustworthy occurrence selector.

## Limitations

- Output quality depends on the selected local model and its structured-output
  support. Extracted results remain AI inferences requiring review.
- This MVP deliberately does not extract relationships or exact instants. Approximate,
  relative, day, month, year, and unknown temporal contexts use the existing domain
  model without inventing timestamps.
- Entity identity is local to one extraction response. Equal names are not merged;
  different local references produce different entities.
- RETRACE does not log prompts or memory contents. Ollama is a separate local process,
  so its own logging and retention behavior should be reviewed by the operator.
- The synchronous adapter is intended for this small local pipeline. Persistence,
  retries, batching, evaluation infrastructure, and API integration are outside this
  phase.
