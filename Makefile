.PHONY: install test lint format clean help venv setup

help:
	@echo "Available commands:"
	@echo "  make setup    - Create venv and install dependencies"
	@echo "  make install  - Install package in development mode"
	@echo "  make test     - Run test suite"
	@echo "  make lint     - Run linter (ruff)"
	@echo "  make format   - Format code (black)"
	@echo "  make clean    - Clean build artifacts"

setup:
	@echo "🚀 Setting up virtual environment..."
	@python3 -m venv venv || python -m venv venv
	@echo "✅ Virtual environment created!"
	@echo "📥 Activate it and run: pip install -r requirements-dev.txt"
	@echo "   On macOS/Linux: source venv/bin/activate"
	@echo "   On Windows: venv\\Scripts\\activate"

install:
	pip install -e ".[dev]"

test:
	pytest -v

lint:
	ruff check minime tests

format:
	black minime tests

clean:
	rm -rf build/
	rm -rf dist/
	rm -rf *.egg-info
	find . -type d -name __pycache__ -exec rm -r {} +
	find . -type f -name "*.pyc" -delete

