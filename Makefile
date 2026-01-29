.PHONY: lint format typecheck typecheck-strict test coverage ci install-hooks clean

lint:
	ruff check src tests
	ruff format --check src tests

format:
	ruff check --fix src tests
	ruff format src tests

typecheck:
	pyright

typecheck-strict:
	pyright --typeCheckingMode strict src

test:
	pytest

coverage:
	pytest --cov --cov-report=html --cov-report=term-missing

ci: lint typecheck coverage

install-hooks:
	pre-commit install

clean:
	rm -rf .pytest_cache .coverage htmlcov .ruff_cache
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
