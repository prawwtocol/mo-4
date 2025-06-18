.PHONY: install lint format check run run-tsp clean

# ====================================================================================
#  Development
# ====================================================================================

install:
	@echo ">>> Installing dependencies..."
	@poetry install

lint:
	@echo ">>> Linting code..."
	@poetry run ruff check .

format:
	@echo ">>> Formatting code..."
	@poetry run ruff format .

check: lint

# ====================================================================================
#  Running
# ====================================================================================

run:
	@echo ">>> Running main experiments..."
	@poetry run python run_experiments.py

run-tsp:
	@echo ">>> Running TSP example..."
	@poetry run python tsp_example/tsp.py

# ====================================================================================
#  Cleaning
# ====================================================================================

clean:
	@echo ">>> Cleaning up..."
	@rm -rf results
	@rm -rf .venv
	@find . -type f -name "*.pyc" -delete
	@find . -type d -name "__pycache__" -delete

help:
	@echo "Makefile for project"
	@echo ""
	@echo "Usage:"
	@echo "    make install        - Install dependencies"
	@echo "    make lint           - Lint code"
	@echo "    make format         - Format code"
	@echo "    make check          - Alias for lint"
	@echo "    make run            - Run main experiments"
	@echo "    make run-tsp        - Run TSP example"
	@echo "    make clean          - Clean up generated files" 