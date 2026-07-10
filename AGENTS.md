# Agent Guidelines

## Mantra

Keep the code clean, direct, and easy to reason about. This is a proof of concept, so favor simplicity over production readiness.

## Principles

- Keep implementations simple when possible.
- Avoid fallbacks unless they are explicitly required and visible in the code path.
- Do not add backwards-compatibility shims, legacy aliases, old import re-exports, or migration fallbacks unless the user explicitly asks for them.
- Do not add hidden behavior, implicit magic, or surprising side effects.
- Classes and abstractions need to earn their place. Prefer plain functions and straightforward data flow until structure is clearly needed.
- Optimize for readability and fast iteration over broad extensibility.
- Make behavior explicit. If something can fail, surface that failure clearly instead of silently recovering.
- Keep changes scoped to the task at hand.
- Do not introduce production-grade infrastructure, configuration layers, or defensive complexity unless the POC genuinely needs it.

## Code Style

- Prefer clear names over clever names.
- Prefer small, focused functions over large generic systems.
- Prefer function-focused design: put behavior in plain functions with explicit inputs and outputs before introducing classes or stateful wrappers.
- Use classes mainly when they own real state, manage an external resource, implement a clear protocol, or make dependency injection materially simpler.
- Avoid premature abstractions and speculative extension points.
- Keep dependencies minimal.
- Delete unused code instead of keeping it around for possible future use.
- When refactoring, update callers, tests, and docs to the new canonical path instead of preserving old paths.

## Module Documentation

- Each first-party application package under `knowledge_agent/` should have a local `README.md`.
- Keep module READMEs short and practical: purpose, main entry points, owned side effects or persisted data, and important constraints.
- Update the relevant module README when changing that module's responsibilities or externally visible behavior.

## Testing And Verification

- Add focused tests when behavior is non-trivial or easy to regress.
- For simple POC flows, lightweight manual verification is acceptable.
- Do not hide test failures behind retries or alternate execution paths.
