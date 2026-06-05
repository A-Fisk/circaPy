# makefile 

.PHONY: all
all: format test


.PHONY: format
format:
	@echo "Formatting code"
	uv run ruff format **/*.py


.PHONY: test
test:
	@echo "running tests"
	uv run pytest


.PHONY: docs
docs:
	@echo "Building documentation"
	uv run --group docs sphinx-build docs/source docs/build/html


