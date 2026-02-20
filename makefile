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
	uv run python -m unittest tests/preprocessing_tests.py
	uv run python -m unittest tests/activity_tests.py
	uv run python -m unittest tests/periodogram_tests.py
	uv run python -m unittest tests/episode_finder_tests.py
	uv run python -m unittest tests/plots_tests.py


