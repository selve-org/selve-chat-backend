.PHONY: help install dev prod test lint format clean docker-up docker-down migrate

# Default target
help:
	@echo "SELVE Chatbot Backend - Available Commands"
	@echo "==========================================="
	@echo "  make install     - Install dependencies"
	@echo "  make dev         - Run development server (port 9000)"
	@echo "  make prod        - Run production server"
	@echo "  make test        - Run tests"
	@echo "  make lint        - Run linter"
	@echo "  make format      - Format code"
	@echo "  make clean       - Clean cache files"
	@echo "  make docker-up   - Start Docker services"
	@echo "  make docker-down - Stop Docker services"
	@echo "  make migrate     - Run database migrations"
	@echo "  make shell       - Open Python shell"
	@echo "  make health      - Check service health"

# Python settings
PYTHON := python3
VENV := venv
PIP := $(VENV)/bin/pip
PYTEST := $(VENV)/bin/pytest
UVICORN := $(VENV)/bin/uvicorn

# Install dependencies
install:
	$(PYTHON) -m venv $(VENV)
	$(PIP) install --upgrade pip
	$(PIP) install -r requirements.txt
	@echo "✅ Dependencies installed"

# Development server
dev:
	@echo "🚀 Starting development server on port 9000..."
	$(UVICORN) app.main:app --reload --host 0.0.0.0 --port 9000

# Run with auto-reload (alias for dev)
run: dev

# Production server
prod:
	@echo "🚀 Starting production server..."
	$(UVICORN) app.main:app --host 0.0.0.0 --port 9000 --workers 4

# Run tests
test:
	$(PYTEST) tests/ -v --cov=app --cov-report=term-missing

# Run specific test by name
test-one:
	$(PYTEST) tests/ -v -k "$(TEST)"

# Run tests without coverage (faster)
test-fast:
	$(PYTEST) tests/ -v

# Lint code
lint:
	$(VENV)/bin/ruff check app/ tests/
	$(VENV)/bin/mypy app/ --ignore-missing-imports

# Format code
format:
	$(VENV)/bin/ruff format app/ tests/
	$(VENV)/bin/isort app/ tests/

# Clean cache
clean:
	find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete 2>/dev/null || true
	find . -type d -name ".pytest_cache" -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name ".mypy_cache" -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name ".ruff_cache" -exec rm -rf {} + 2>/dev/null || true
	@echo "✅ Cache cleaned"

# Docker services
docker-up:
	cd .. && docker-compose up -d redis qdrant
	@echo "✅ Redis and Qdrant started"

docker-down:
	cd .. && docker-compose down
	@echo "✅ Docker services stopped"

# Database migrations
migrate:
	$(VENV)/bin/prisma migrate dev
	@echo "✅ Migrations applied"

migrate-prod:
	$(VENV)/bin/prisma migrate deploy
	@echo "✅ Production migrations applied"

# Generate Prisma client
prisma-generate:
	$(VENV)/bin/prisma generate
	@echo "✅ Prisma client generated"

# Open Python shell with app loaded
shell:
	$(VENV)/bin/python -i -c "from app.main import app; print('🐍 SELVE Chatbot Shell'); print('App loaded. Use: app, from app.services import *')"

# Interactive Python REPL
repl:
	$(VENV)/bin/python

# Check all services health
health:
	@echo "🔍 Checking services..."
	@curl -sf http://localhost:9000/health > /dev/null && echo "✅ Backend: Running" || echo "❌ Backend: Not running"
	@curl -sf http://localhost:6333/health > /dev/null && echo "✅ Qdrant: Running" || echo "❌ Qdrant: Not running"
	@redis-cli ping > /dev/null 2>&1 && echo "✅ Redis: Running" || echo "❌ Redis: Not running"

# Index content into Qdrant
index:
	$(VENV)/bin/python -m scripts.index_content
	@echo "✅ Content indexed"

# Quick start (install + docker + dev)
start: install docker-up dev
