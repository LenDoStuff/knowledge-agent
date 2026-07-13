# LLM

This package is the shared PydanticAI provider boundary. It loads model
settings and owns the async model client behind a synchronous application API.

## Main entry points

- `load_llm_settings(profile)` reads the active provider settings.
- `open_agent_runtime(settings)` yields an `AgentRuntime` with one PydanticAI
  model, one `AsyncOpenAI` client, and one persistent event loop hosted on its
  own thread so synchronous callers remain safe inside hosts such as Streamlit.
- `AgentRuntime.run(agent, prompt, ...)` returns the native PydanticAI
  `AgentRunResult`.
- `AgentRuntime.run_coroutine(...)` runs embedded async resources such as
  LightRAG initialization, indexing, and shutdown on that same loop.

## Providers

- `api_key` uses `OpenAIChatModel` against NVIDIA's OpenAI-compatible endpoint
  with DeepSeek V4 Pro in deterministic non-thinking mode.
- `azure_project` uses `OpenAIResponsesModel` against the Foundry project
  endpoint with browser authentication and the configured reasoning effort.

HTTP retries and fallback models are disabled. PydanticAI agents may use one
explicit validation retry for malformed structured output or tool arguments.
LightRAG's indexing and keyword prompts are adapted to named PydanticAI agents;
it never creates a second raw text-generation client.

## Constraints

Keep this package provider-focused. Claim prompts, tools, citation validation,
and retrieval behavior belong in `agents/` or `claims/`. The runtime owns and
closes its async client on the same event loop used for every agent request.
