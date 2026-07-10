#!/usr/bin/env bash
# mneme quickstart — clone (if needed), build, and launch the live server.
#
#   ./scripts/quickstart.sh          # build + run the zero-download demo
#   REAL=1 ./scripts/quickstart.sh   # persistent log + real fastembed embeddings
#
# Or straight from the web (reviews the script first, please):
#   curl -fsSL https://raw.githubusercontent.com/mneme/mneme/main/scripts/quickstart.sh | bash
set -euo pipefail

REPO_URL="https://github.com/mneme/mneme.git"
PORT="${MNEME_PORT:-7777}"

info() { printf '\033[36m▸ %s\033[0m\n' "$*"; }
err()  { printf '\033[31m✗ %s\033[0m\n' "$*" >&2; }

if ! command -v cargo >/dev/null 2>&1; then
  err "Rust/cargo not found."
  echo "  Install Rust:                 https://rustup.rs"
  echo "  Or run with no toolchain:     docker compose up --build"
  exit 1
fi

# If we're not already inside the repo, clone it.
if ! grep -q 'name = "mneme-server"' crates/mneme-server/Cargo.toml 2>/dev/null; then
  info "Cloning mneme into ./mneme"
  git clone --depth 1 "$REPO_URL" mneme
  cd mneme
fi

info "Building mneme-server (the first build compiles the workspace — a few minutes)"
cargo build --release -p mneme-server

if [ "${REAL:-0}" = "1" ]; then
  info "Starting real server on http://127.0.0.1:${PORT}  (fastembed, persistent ./mneme-data)"
  MNEME_EMBEDDER=fastembed MNEME_PORT="$PORT" exec ./target/release/mneme
else
  info "Starting demo on http://127.0.0.1:${PORT}  (mock embedder, live learning curve, zero downloads)"
  MNEME_DEMO=1 MNEME_EMBEDDER=mock MNEME_PROCEDURAL=on MNEME_PORT="$PORT" exec ./target/release/mneme
fi
