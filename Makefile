.PHONY: help install install-dev test test-cov lint format type-check clean

help:  ## Show this help message
	@echo "Available commands:"
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-20s\033[0m %s\n", $$1, $$2}'

install:  ## Install package
	pip install -e .

install-dev:  ## Install package with development dependencies
	pip install -e ".[dev]"

test:  ## Run tests
	pytest

test-cov:  ## Run tests with coverage report
	pytest --cov=benchmark_capture --cov-report=html --cov-report=term-missing

lint:  ## Run linter (flake8)
	flake8 benchmark_capture/ tests/

format:  ## Format code (black + isort)
	black benchmark_capture/ tests/
	isort benchmark_capture/ tests/

format-check:  ## Check code formatting
	black --check benchmark_capture/ tests/
	isort --check benchmark_capture/ tests/

type-check:  ## Run type checker (mypy)
	mypy benchmark_capture/

check-all:  ## Run all checks (format, lint, type-check, test)
	@echo "Running format check..."
	@$(MAKE) format-check
	@echo "\nRunning linter..."
	@$(MAKE) lint
	@echo "\nRunning type checker..."
	@$(MAKE) type-check
	@echo "\nRunning tests..."
	@$(MAKE) test-cov

clean:  ## Clean build artifacts and cache
	rm -rf build/ dist/ *.egg-info
	rm -rf .pytest_cache .mypy_cache .coverage htmlcov/
	find . -type d -name __pycache__ -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete

build:  ## Build distribution packages
	python -m build

.DEFAULT_GOAL := help
