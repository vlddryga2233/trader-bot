# Repository Guidelines

## Project Structure & Module Organization

Application code lives in `src/trader_bot/`, organized by responsibility: `data/` handles ingestion and sources, `features/` and `labeling/` prepare model inputs, `models/` trains and registers models, `backtest/` evaluates strategies, `risk/` enforces limits, and `execution/` integrates with Binance. `scheduler/`, `db/`, and `config/` provide orchestration, persistence, and settings. Tests are in `tests/`; operational and design documentation is in `docs/`; strategy defaults are in `config/strategy.yaml`. Generated data, models, and runtime state belong under `artifacts/` and `state/`, not in source modules.

## Build, Test, and Development Commands

- `uv sync` installs project and development dependencies for Python 3.12.
- `uv run pytest` runs the test suite.
- `uv run ruff check src tests` checks code style and common errors.
- `uv run trader-bot --help` lists CLI commands. For local setup and testnet workflows, follow `README.md` and `docs/operations.md`.
- `docker compose up -d timescaledb` starts the local database service used by the application.

## Coding Style & Naming Conventions

Use Python 3.12, four spaces for indentation, and Ruff's configured 100-character line limit. Keep modules grouped by domain under `src/trader_bot/`. Use `snake_case` for modules, functions, and variables; use `PascalCase` for classes. Keep configuration in YAML and load it through the existing settings/configuration code rather than duplicating defaults.

## Testing Guidelines

Pytest is configured to discover tests under `tests/`, with automatic asyncio support. Name files `test_*.py` and test functions `test_*`; organize cases around behavior such as feature construction, risk controls, and order management. Run `uv run pytest` before submitting changes, and add or update focused tests when changing behavior. Tests and development runs should not require live trading credentials.

## Commit & Pull Request Guidelines

This checkout has no accessible Git history to establish a commit convention. Use short imperative commit subjects (for example, `Fix stop order reconciliation`). Pull requests should explain the behavior changed, note risk or configuration impacts, link related issues when available, and include test results. Add screenshots only when a change has a visual component.

## Security & Configuration

Keep secrets in an untracked `.env` based on `.env.example`; never commit API keys, credentials, or generated model/data artifacts. Use Binance testnet credentials for execution work. Preserve the hard risk limits and kill-switch behavior, and review `docs/risk-and-execution.md` before changing order handling or safety controls.
