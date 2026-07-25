#!/usr/bin/env bash
# mnesio quickstart — clone (if needed), build, and launch the live server.
#
#   ./scripts/quickstart.sh          # build + run the zero-download demo
#   REAL=1 ./scripts/quickstart.sh   # persistent log + real fastembed embeddings
#
# Or straight from the web (reviews the script first, please):
#   curl -fsSL https://raw.githubusercontent.com/mnesio/mnesio/main/scripts/quickstart.sh | bash
set -euo pipefail

REPO_URL="https://github.com/mnesio/mnesio.git"
PORT="${MNESIO_PORT:-7777}"

info() { printf '\033[36m▸ %s\033[0m\n' "$*"; }
err()  { printf '\033[31m✗ %s\033[0m\n' "$*" >&2; }

if ! command -v cargo >/dev/null 2>&1; then
  err "Rust/cargo not found."
  echo "  Install Rust:                 https://rustup.rs"
  echo "  Or run with no toolchain:     docker compose up --build"
  exit 1
fi

# If we're not already inside the repo, clone it.
if ! grep -q 'name = "mnesio-server"' crates/mnesio-server/Cargo.toml 2>/dev/null; then
  info "Cloning mnesio into ./mnesio"
  git clone --depth 1 "$REPO_URL" mnesio
  cd mnesio
fi

info "Building mnesio-server (the first build compiles the workspace — a few minutes)"
cargo build --release -p mnesio-server

if [ "${REAL:-0}" = "1" ]; then
  info "Starting real server on http://127.0.0.1:${PORT}  (fastembed, persistent ./mnesio-data)"
  MNESIO_EMBEDDER=fastembed MNESIO_PORT="$PORT" exec ./target/release/mnesio
else
  info "Starting demo on http://127.0.0.1:${PORT}  (mock embedder, live learning curve, zero downloads)"
  MNESIO_DEMO=1 MNESIO_EMBEDDER=mock MNESIO_PROCEDURAL=on MNESIO_PORT="$PORT" exec ./target/release/mnesio
fi
