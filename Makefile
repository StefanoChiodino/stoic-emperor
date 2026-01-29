.PHONY: lint format typecheck typecheck-strict test coverage ci install-hooks clean

dev:
	uvicorn src.web.api:app --reload

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

clean:
	rm -rf .pytest_cache .coverage htmlcov .ruff_cache
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
