# Task Plan: Hermes-Friendly Expense Interface

- [x] Inspect existing CLI and Flask workflows.
- [x] Add an agent-friendly API surface that previews receipts without changing files.
- [x] Add an explicit approval path for writing `expenses.csv` and renaming files.
- [x] Add a small usage document for the Hermes agent.
- [x] Verify behavior with a local temporary receipt folder.
- [x] Add Render upload/session endpoints so Hermes can use the deployed app.
- [x] Verify remote-style upload, preview, process, and download flow locally.

## Review

- Added guarded Flask endpoints for Hermes-style callers.
- Added `--json` mode to the CLI for machine-readable preview/apply flows.
- Verified syntax with AST compilation.
- Verified `/api/expenses/manifest`, `/api/expenses/preview`, and dry-run `/api/expenses/process` through Flask's test client.
- Verified an isolated apply run renamed a sample receipt and wrote `expenses.csv` with the required columns.
- Verified remote-style multipart upload, session preview, approved process, CSV download, and ZIP download through Flask's test client.
