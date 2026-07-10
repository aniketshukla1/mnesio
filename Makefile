# mneme — one-command developer entrypoints.
# Run `make` (or `make help`) to see everything.

.DEFAULT_GOAL := help
CARGO ?= cargo
PORT  ?= 7777

.PHONY: help demo run test lint mcp install docker clean

help: ## Show this help
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) \
	  | awk 'BEGIN{FS=":.*?## "}{printf "  \033[36m%-10s\033[0m %s\n", $$1, $$2}'

demo: ## Instant demo — live dashboard, ZERO downloads (mock embedder + synthetic writer + learning curve)
	MNEME_DEMO=1 MNEME_EMBEDDER=mock MNEME_PROCEDURAL=on MNEME_PORT=$(PORT) $(CARGO) run --release -p mneme-server

run: ## Real server — persistent log + fastembed embeddings (downloads bge-small on first run)
	MNEME_EMBEDDER=fastembed MNEME_PORT=$(PORT) $(CARGO) run --release -p mneme-server

test: ## Run the workspace test suite
	$(CARGO) test --workspace

lint: ## Exactly what CI enforces (fmt + clippy -D warnings)
	$(CARGO) fmt --check
	$(CARGO) clippy --workspace -- -D warnings

mcp: ## Install the MCP server binary (for Claude Desktop / Cursor / any MCP client)
	$(CARGO) install --path crates/mneme-mcp

install: ## Install the `mneme` server binary onto your PATH
	$(CARGO) install --path crates/mneme-server

docker: ## Build + run the server in Docker — no Rust toolchain needed (http://localhost:7777)
	docker compose up --build

clean: ## Remove build artifacts + local demo data
	$(CARGO) clean
	rm -rf ./mneme-data
