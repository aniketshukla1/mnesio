# Security Policy

## Reporting a vulnerability

**Please do not open a public issue for a security problem.**

Report it privately through GitHub's private vulnerability reporting:
**Security → Report a vulnerability** on the repository. That opens a private
advisory visible only to the maintainers.

Please include:

- what the issue is and roughly how severe you think it is,
- the smallest reproduction you can manage (a failing test is ideal),
- the commit or version, your OS, and any feature flags in play — several
  subsystems are feature-gated and behave differently.

You can expect an acknowledgement within a few days. Since mnesio is a
pre-1.0 project maintained in the open, please allow reasonable time for a fix
before disclosing publicly, and tell us if you have a disclosure deadline.

## Supported versions

mnesio is pre-1.0. Only the latest release on `main` receives security fixes;
there are no maintained backport branches yet. That will change when 1.0 ships.

## What we consider a security issue

mnesio makes a small number of security-relevant promises. A break in any of them
is a vulnerability, not a bug:

- **Scope isolation.** A `Scope` is a security boundary. Any path that reads or
  learns across a scope without an explicit aggregation/anonymization step — in
  retrieval, procedural learning, evolution, or a materialized view — is a
  vulnerability.
- **Crypto-shred erasure.** Forgetting a subject drops its key, after which the
  content must be unrecoverable — including from rebuilt materialized views and
  from historical/time-travel reads. Any route that reconstructs erased plaintext
  (a cached view, a KV cartridge blob, a replay path) is a vulnerability.
- **The procedural safety gate.** Nothing procedural may activate without passing
  `EvalReport::is_committable()`. Any bypass — including via configuration —
  is a vulnerability. Likewise for skill import: a certificate whose signature
  fails, or which is activated without re-running the importer's own gate.
- **Agent ACLs and actor attribution.** A path that lets one agent read another's
  memories against the configured ACL.
- **Append-only integrity.** Anything that mutates or drops existing log entries
  rather than appending a superseding one.

## Credentials

mnesio reads all provider API keys **from the environment only** — never from a
constructor argument, a command-line flag, or a config file, and they are never
written to the log or to any materialized view. If you find a path where a key is
persisted, logged, or echoed, please report it.

Do not send us credentials, tokens, or production data in a report. If a
reproduction seems to need them, say so and we'll work out a safe alternative.

## Scope notes

Things that are **not** vulnerabilities on their own:

- Running the demo/dashboard bound to `0.0.0.0`. That is opt-in via `MNESIO_HOST`
  and documented; mnesio ships no authentication layer and is intended to run
  behind your own trust boundary. If you expose it publicly, put an
  authenticating proxy in front.
- Resource exhaustion from deliberately unbounded configuration (for example,
  raising the evolution caps far past their defaults).
- Findings that require an attacker who already has write access to the event
  log or the host filesystem.
