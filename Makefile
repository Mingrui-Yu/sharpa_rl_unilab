.PHONY: check test test-slow test-all build

check:
	uv run ruff check src tests
	uv run ruff format --check src tests
	uv run mypy src/sharpa_rl_unilab
	uv run pyright

test:
	uv run pytest

test-slow:
	uv run pytest -m slow

test-all: check test test-slow

build:
	uv build
