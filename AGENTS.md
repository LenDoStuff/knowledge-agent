# Agent Guidelines

## Mantra

Keep the code clean, direct, and easy to reason about. This is a proof of concept, so favor simplicity over production readiness.

## Principles

- Keep implementations simple when possible.
- Avoid fallbacks unless they are explicitly required and visible in the code path.
- Do not add hidden behavior, implicit magic, or surprising side effects.
- Classes and abstractions need to earn their place. Prefer plain functions and straightforward data flow until structure is clearly needed.
- Optimize for readability and fast iteration over broad extensibility.
- Make behavior explicit. If something can fail, surface that failure clearly instead of silently recovering.
- Keep changes scoped to the task at hand.
- Do not introduce production-grade infrastructure, configuration layers, or defensive complexity unless the POC genuinely needs it.

## Code Style

- Prefer clear names over clever names.
- Prefer small, focused functions over large generic systems.
- Avoid premature abstractions and speculative extension points.
- Keep dependencies minimal.
- Delete unused code instead of keeping it around for possible future use.

## Testing And Verification

- Add focused tests when behavior is non-trivial or easy to regress.
- For simple POC flows, lightweight manual verification is acceptable.
- Do not hide test failures behind retries or alternate execution paths.
