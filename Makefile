.PHONY: help dev server ui-dev ui-build install test restart clean logs smoke

help: ## Show this help
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-15s\033[0m %s\n", $$1, $$2}'

install: ## Install Python package in dev mode + UI deps
	pip install -e ".[dev]"
	cd ui && npm install

server: ## Start the backend server (auto-reloads on Python changes)
	python -m ormah.main

ui-dev: ## Start the Vite dev server (hot-reload for UI work)
	cd ui && npm run dev

ui-build: ## Build the UI into ui/dist/ for production
	cd ui && npx vite build

dev: ## Start backend + UI dev server together (requires ctrl-c to stop both)
	@trap 'kill 0' EXIT; \
	python -m ormah.main & \
	cd ui && npm run dev

restart: ## Rebuild UI and restart backend (kills existing ormah.main process)
	@echo "==> Building UI..."
	cd ui && npx vite build
	@echo "==> Stopping existing server..."
	-pkill -f "ormah.main" 2>/dev/null || true
	@sleep 1
	@echo "==> Starting server..."
	python -m ormah.main &
	@echo "==> Server restarted. Open http://localhost:8787"

test: ## Run the test suite
	python -m pytest tests/ -v

lint: ## Run ruff linter
	ruff check src/ tests/

clean: ## Remove build artifacts
	rm -rf src/ormah/ui_dist ui/node_modules/.vite
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	rm -rf .pytest_cache

logs: ## Tail the server logs (if running in background)
	@echo "Server runs with stdout logging. Use 'make server' in foreground to see logs."

smoke: ## Run fresh-install smoke test in Docker
	docker build -f tests/smoke/Dockerfile -t ormah-smoke .
	docker run --rm \
		-v ormah-model-cache:/tmp/fastembed_cache \
		ormah-smoke
