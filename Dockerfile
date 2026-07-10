# mneme — self-contained server image.
#
# This intentionally prioritises "works on the first `docker compose up`" over
# minimal image size: it builds and runs in the same Rust image, so every
# native dependency (onnxruntime for fastembed, tantivy, hnsw_rs) is present at
# runtime with no shared-library hunting in a slim base. Shrinking to a
# multi-stage distroless image is a follow-up, not a correctness requirement.
FROM rust:1-slim-bookworm

# Native build + runtime deps: C toolchain (tantivy/hnsw_rs), cmake, TLS.
RUN apt-get update && apt-get install -y --no-install-recommends \
      build-essential cmake pkg-config libssl-dev ca-certificates \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /mneme
COPY . .

# Build the server (its bin name is `mneme`). Default features only — the heavy
# GPU/LLM features stay off so the image builds on any machine.
RUN cargo build --release -p mneme-server

# The append-only log lives on a mounted volume, not in the image layer.
ENV MNEME_DATA=/data
VOLUME /data

# Container defaults:
#  - MNEME_HOST=0.0.0.0 so the published port is actually reachable.
#  - demo + mock embedder means the dashboard comes up with ZERO external
#    downloads on first boot. Flip MNEME_DEMO=0 + MNEME_EMBEDDER=fastembed
#    (see docker-compose.yml) for real, persistent use.
ENV MNEME_HOST=0.0.0.0 \
    MNEME_PORT=7777 \
    MNEME_DEMO=1 \
    MNEME_EMBEDDER=mock \
    MNEME_PROCEDURAL=on \
    RUST_LOG=info

EXPOSE 7777
CMD ["./target/release/mneme"]
