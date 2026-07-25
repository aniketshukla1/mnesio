# mnesio — one-command developer entrypoints.
# Run `make` (or `make help`) to see everything.

.DEFAULT_GOAL := help
CARGO ?= cargo
PORT  ?= 7777

.PHONY: help demo run test lint mcp install docker clean

help: ## Show this help
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) \
	  | awk 'BEGIN{FS=":.*?## "}{printf "  \033[36m%-10s\033[0m %s\n", $$1, $$2}'

demo: ## Instant demo — live dashboard, ZERO downloads (mock embedder + synthetic writer + learning curve)
	MNESIO_DEMO=1 MNESIO_EMBEDDER=mock MNESIO_PROCEDURAL=on MNESIO_PORT=$(PORT) $(CARGO) run --release -p mnesio-server

run: ## Real server — persistent log + fastembed embeddings (downloads bge-small on first run)
	MNESIO_EMBEDDER=fastembed MNESIO_PORT=$(PORT) $(CARGO) run --release -p mnesio-server

test: ## Run the workspace test suite
	$(CARGO) test --workspace

lint: ## Exactly what CI enforces (fmt + clippy -D warnings)
	$(CARGO) fmt --check
	$(CARGO) clippy --workspace -- -D warnings

mcp: ## Install the MCP server binary (for Claude Desktop / Cursor / any MCP client)
	$(CARGO) install --path crates/mnesio-mcp

install: ## Install the `mnesio` server binary onto your PATH
	$(CARGO) install --path crates/mnesio-server

docker: ## Build + run the server in Docker — no Rust toolchain needed (http://localhost:7777)
	docker compose up --build

clean: ## Remove build artifacts + local demo data
	$(CARGO) clean
	rm -rf ./mnesio-data
