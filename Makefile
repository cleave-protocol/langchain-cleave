.PHONY: all format lint type test tests integration_test integration_tests check_imports build help

all: help

TEST_FILE ?= tests/unit_tests/

integration_test integration_tests: TEST_FILE=tests/integration_tests/

test tests:
	uv run --group test pytest --disable-socket --allow-unix-socket $(TEST_FILE)

integration_test integration_tests:
	uv run --group test --group test_integration pytest -v --tb=short $(TEST_FILE)

PYTHON_FILES=.

lint:
	./scripts/lint_imports.sh
	uv run --group lint ruff check $(PYTHON_FILES)
	uv run --group lint ruff format $(PYTHON_FILES) --diff
	uv run --group typing --group test mypy $(PYTHON_FILES)

type:
	uv run --group typing --group test mypy $(PYTHON_FILES)

format:
	uv run --group lint ruff format $(PYTHON_FILES)
	uv run --group lint ruff check --fix $(PYTHON_FILES)

check_imports: $(shell find langchain_cleave -name '*.py')
	uv run python ./scripts/check_imports.py $^

build:
	uv build --build-constraint build-constraints.txt --require-hashes

help:
	@echo 'check_imports                - check imports'
	@echo 'format                       - run code formatters'
	@echo 'lint                         - run linters'
	@echo 'type                         - run type checking'
	@echo 'test                         - run unit tests'
	@echo 'integration_tests            - run integration tests against the live API'
	@echo 'build                        - build the sdist and wheel into dist/'
