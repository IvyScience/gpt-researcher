---
name: unit-tests-docker
description: Enforce the unit test workflow for this repo. Use when the user asks to run unit tests, pytest, tests/unit, or when adding/modifying unit tests. Always run unit tests inside the existing Docker container via docker exec using the exact command provided. If unit tests were changed, rebuild the Docker image first, then run the command.
---

# Unit Tests (Docker)

## Rules

### Always run unit tests with this exact command

When unit tests need to be executed, run:

```bash
cd /Users/wuyanhong/workspace/gpt-researcher-1 && docker exec gpt-researcher-1-gpt-researcher-1 python3 -m pytest -q tests/unit/
```

Do not run `pytest` locally for unit tests unless the user explicitly asks.

### If unit tests changed, rebuild Docker first

If the current change set includes any added/modified/deleted files under `tests/unit/`, rebuild the Docker image first:

```bash
cd /Users/wuyanhong/workspace/gpt-researcher-1 && docker compose build gpt-researcher
```

Then run the unit test command above.

## Examples (trigger phrases)

- "跑一下单元测试"
- "run unit tests"
- "pytest"
- "tests/unit"
