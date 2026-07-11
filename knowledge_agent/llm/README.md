# LLM

The LLM package is the shared provider boundary. It loads model settings,
constructs synchronous provider clients, and exposes one structured-output
interface for the agent packages.

## Main entry points

- `load_llm_settings(profile)` reads the active provider settings.
- `llm_provider(settings)` returns the concrete provider name.
- `parse_structured_output(settings, client, system, user, response_model)`
  parses one structured response into a validated Pydantic model.
- `open_structured_output_parser(settings)` opens the configured provider
  client and yields the parser callable used by agent functions.

## Providers

The `api_key` profile uses NVIDIA DeepSeek V4 Pro through NVIDIA's
OpenAI-compatible endpoint:

- `nvidia_base_url`
- `nvidia_api_key_ds4`
- model: `deepseek-ai/deepseek-v4-pro`

Both profiles use the OpenAI Responses structured-output parser with the
Pydantic response model passed as `text_format`. The SDK sends the model's JSON
Schema through `text.format`, and the provider must return schema-conforming
output. The `api_key` profile calls NVIDIA's OpenAI-compatible Responses
endpoint; the `azure_project` profile uses the Azure AI Project OpenAI client.

## Constraints

Keep this package provider-focused. Claim-specific prompts, research strategy,
document metadata rules, and retrieval behavior belong in `agents/` or
`claims/`. Provider failures should surface as `LlmError`; do not add hidden
retries or fallback providers.
