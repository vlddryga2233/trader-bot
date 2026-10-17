from __future__ import annotations

import asyncio
import json
import logging
from datetime import UTC, date, datetime
from pathlib import Path

import structlog
import typer

from trader_bot.config.settings import Settings, load_strategy

app = typer.Typer(no_args_is_help=True, add_completion=False)


def _ctx():
    settings = Settings()
    cfg = load_strategy(settings.strategy_config)
    structlog.configure(wrapper_class=structlog.make_filtering_bound_logger(logging.INFO))
    from trader_bot.db.store import make_engine

    return settings, cfg, make_engine(settings.database_url)


@app.command("init-db")
def init_db_cmd() -> None:
    """Create tables and Timescale hypertables."""
    from trader_bot.db.schema import init_db

    _, _, engine = _ctx()
    init_db(engine)
    typer.echo("database initialised")


@app.command()
def backfill() -> None:
    """Download history (klines, funding, derivatives metrics) for all symbols."""
    from trader_bot.data.ingest.backfill import backfill_symbol, report_gaps

    _, cfg, engine = _ctx()
    for s in cfg.symbols:
        backfill_symbol(engine, s, cfg.history_start)
        gaps = {k: len(v) for k, v in report_gaps(engine, s).items()}
        typer.echo(f"{s} gaps: {gaps}")


@app.command("backfill-liquidations")
def backfill_liquidations_cmd(
    start: str = typer.Option(..., help="First event date to load (YYYY-MM-DD)"),
    end: str = typer.Option(..., help="Last event date to load (YYYY-MM-DD)"),
) -> None:
    """Download historical Binance liquidation aggregates from API BRICKS."""
    from trader_bot.data.ingest.backfill import backfill_liquidations

    settings, cfg, engine = _ctx()
    try:
        start_date = date.fromisoformat(start)
        end_date = date.fromisoformat(end)
    except ValueError as exc:
        raise typer.BadParameter("dates must use YYYY-MM-DD format") from exc
    if start_date > end_date:
        raise typer.BadParameter("start must not be after end")
    key = settings.api_briks_api_key.get_secret_value()
    try:
        count = backfill_liquidations(
            engine,
            cfg.symbols,
            start_date,
            end_date,
            key,
        )
    except ValueError as exc:
        raise typer.BadParameter(str(exc)) from exc
    typer.echo(f"loaded {count} aggregate liquidation rows from API BRICKS")


@app.command()
def collect() -> None:
    """Run live collectors (closed klines, liquidations, metrics polling)."""
    from trader_bot.data.ingest.live import run_collectors

    _, cfg, engine = _ctx()
    asyncio.run(run_collectors(engine, cfg.symbols))


@app.command()
def train(
    tune_params: bool = typer.Option(False, "--tune", help="Run Optuna first"),
    no_promote: bool = typer.Option(
        False, "--no-promote", help="Save trained candidates without changing active models"
    ),
) -> None:
    """Train + cross-validate models and promote them if not worse than the current ones."""
    from trader_bot.models.registry import ModelRegistry
    from trader_bot.models.retrain import retrain_symbol

    settings, cfg, engine = _ctx()
    reg = ModelRegistry(settings.artifacts_dir)
    for s in cfg.symbols:
        result = retrain_symbol(
            engine, cfg, reg, s, tune_params, promote_model=not no_promote
        )
        typer.echo(json.dumps({s: result}, indent=2))


@app.command()
def experiment(
    flat_atr_mults: str = typer.Option(
        "0.5,1.0,1.5", help="Comma-separated flat-label ATR multipliers to compare"
    ),
) -> None:
    """Compare flat-label bands and balanced classifiers with purged walk-forward CV."""
    from trader_bot.models.experiments import compare_label_and_weight_variants

    _, cfg, engine = _ctx()
    try:
        multipliers = [float(value.strip()) for value in flat_atr_mults.split(",")]
    except ValueError as exc:
        raise typer.BadParameter("use comma-separated positive numbers") from exc
    out = Path("artifacts/experiments") / datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    report = compare_label_and_weight_variants(engine, cfg, out, multipliers)
    summary = {
        symbol: [
            {
                "flat_atr_mult": item["flat_atr_mult"],
                "class_balance": item["class_balance"],
                "balanced_accuracy": item["metrics"]["balanced_accuracy"],
                "directional_hc_coverage": item["metrics"]["directional_hc_coverage"],
                "directional_hc_precision": item["metrics"]["directional_hc_precision"],
                "oos_class_counts": item["metrics"]["oos_class_counts"],
                "oos_predicted_class_counts": item["metrics"]["oos_predicted_class_counts"],
            }
            for item in results
        ]
        for symbol, results in report["symbols"].items()
    }
    typer.echo(json.dumps({"report": str(out / "report.json"), "summary": summary}, indent=2))


@app.command()
def backtest(
    out: Path | None = None,
    flat_atr_mult: float | None = typer.Option(None, help="Override the flat-label ATR multiplier"),
    min_reward_risk: float | None = typer.Option(None, help="Override the risk engine reward/risk floor"),
    balance_classes: bool = typer.Option(False, help="Use balanced classifier weights for this run"),
    registry_params: bool = typer.Option(
        False, "--registry-params", help="Use hyperparameters saved with promoted models"
    ),
) -> None:
    """Walk-forward out-of-sample backtest; prints metrics and the Sharpe/MaxDD gate result."""
    from trader_bot.backtest.run import run_backtest

    settings, cfg, engine = _ctx()
    if flat_atr_mult is not None:
        if flat_atr_mult <= 0:
            raise typer.BadParameter("must be greater than zero")
        cfg.labeling = cfg.labeling.model_copy(update={"flat_atr_mult": flat_atr_mult})
    if min_reward_risk is not None:
        if min_reward_risk <= 0:
            raise typer.BadParameter("must be greater than zero")
        cfg.risk = cfg.risk.model_copy(update={"min_reward_risk": min_reward_risk})
    if balance_classes:
        cfg.model = cfg.model.model_copy(update={"class_balance": True})
    if out is None:
        out = Path("artifacts/backtest")
        if flat_atr_mult is not None or min_reward_risk is not None:
            band = f"_band{flat_atr_mult:g}" if flat_atr_mult is not None else ""
            rr = f"_rr{min_reward_risk:g}" if min_reward_risk is not None else ""
            weighted = "_balanced" if balance_classes else ""
            out = Path(f"artifacts/backtest{band}{rr}{weighted}")
    params_by_symbol = None
    if registry_params:
        from trader_bot.models.registry import ModelRegistry

        reg = ModelRegistry(settings.artifacts_dir)
        params_by_symbol = {symbol: reg.current_params(symbol) for symbol in cfg.symbols}
    typer.echo(json.dumps(run_backtest(engine, cfg, out, params_by_symbol), indent=2, default=str))


@app.command("signal-study")
def signal_study(
    holdout_start: str = typer.Option("2025-10-01", help="First date reserved for final holdout"),
    out: Path | None = typer.Option(None, help="Output directory for the study report"),
) -> None:
    """Inspect class/probability balance, tune thresholds on validation, score final holdout."""
    from trader_bot.models.signal_study import run_signal_study

    _, cfg, engine = _ctx()
    try:
        holdout = datetime.fromisoformat(holdout_start)
        if holdout.tzinfo is None:
            holdout = holdout.replace(tzinfo=UTC)
    except ValueError as exc:
        raise typer.BadParameter("holdout_start must be a date or timestamp") from exc
    if out is None:
        out = Path("artifacts/experiments") / datetime.now(UTC).strftime(
            "signal-study-%Y%m%dT%H%M%SZ"
        )
    report = run_signal_study(engine, cfg, holdout, out)
    summary = {
        "report": str(out / "report.json"),
        "selected_parameters": report["selected_parameters"],
        "final_holdout": report["final_holdout"],
    }
    typer.echo(json.dumps(summary, indent=2, default=str))


def _parse_utc(value: str, name: str) -> datetime:
    try:
        parsed = datetime.fromisoformat(value)
    except ValueError as exc:
        raise typer.BadParameter(f"{name} must be a date or timestamp") from exc
    return parsed if parsed.tzinfo else parsed.replace(tzinfo=UTC)


@app.command("barrier-experiment")
def barrier_experiment(
    holdout_start: str = typer.Option("2025-10-01", help="Rows from this date are excluded"),
    tp_atr_mults: str = typer.Option("2.0,3.0,4.0", help="Comma-separated take-profit ATR mults"),
    horizons: str = typer.Option("3,6,12", help="Comma-separated horizons in candles"),
    pooled: bool = typer.Option(False, "--pooled", help="Train one model on all symbols"),
    feature_groups: str = typer.Option(
        "", help="Comma-separated extra feature groups to enable: cross_asset,calendar,derivs_extra"
    ),
    model_size: str = typer.Option("default", help="default or small"),
    t_min: float = typer.Option(3.0, help="Minimum non-overlapping t-statistic"),
    min_trades: int = typer.Option(40, help="Minimum non-overlapping trades per symbol"),
    interval: str = typer.Option("4h", help="4h or 1d (1d is research-only)"),
    min_train_candles: int | None = typer.Option(None, help="Override model.min_train_candles"),
    limit_offset_atr: float = typer.Option(0.0, help="Limit entry offset in ATR (0 = market)"),
    taker_fee: float | None = typer.Option(
        None, help="What-if fee for cost sensitivity (report only; never used for selection)"
    ),
) -> None:
    """Purged-CV grid for barrier labels + EV rule, using only pre-holdout data."""
    import pandas as pd

    from trader_bot.models.barrier_experiment import SUMMARY_KEYS, run_barrier_experiment

    _, cfg, engine = _ctx()
    try:
        tps = [float(value) for value in tp_atr_mults.split(",")]
        hs = [int(value) for value in horizons.split(",")]
    except ValueError as exc:
        raise typer.BadParameter("use comma-separated numbers") from exc
    if any(value <= 0 for value in tps) or any(value <= 0 for value in hs):
        raise typer.BadParameter("values must be positive")
    allowed = {"cross_asset", "calendar", "derivs_extra"}
    extra = {group.strip() for group in feature_groups.split(",") if group.strip()}
    if not extra <= allowed:
        raise typer.BadParameter(f"feature groups must be in {sorted(allowed)}")
    if model_size not in ("default", "small"):
        raise typer.BadParameter("model_size must be default or small")
    if interval not in ("4h", "1d"):
        raise typer.BadParameter("interval must be 4h or 1d")
    if not 0 <= limit_offset_atr <= 1:
        raise typer.BadParameter("limit_offset_atr must be between 0 and 1")
    if pooled and limit_offset_atr > 0:
        raise typer.BadParameter("limit entry is not supported with --pooled")
    cfg.decision_interval = interval
    if min_train_candles is not None:
        cfg.model = cfg.model.model_copy(update={"min_train_candles": min_train_candles})
    groups = cfg.features.groups.model_copy(update={group: True for group in extra})
    cfg.features = cfg.features.model_copy(update={"groups": groups})
    cfg.barrier = cfg.barrier.model_copy(update={"model_size": model_size,
                                                "limit_offset_atr": limit_offset_atr})
    if taker_fee is not None:
        cfg.backtest = cfg.backtest.model_copy(update={"taker_fee": taker_fee})
    holdout = pd.Timestamp(_parse_utc(holdout_start, "holdout_start"))
    out = Path("artifacts/experiments") / datetime.now(UTC).strftime("barrier-%Y%m%dT%H%M%SZ")
    report = run_barrier_experiment(engine, cfg, out, holdout, tps, hs, pooled=pooled,
                                    t_min=t_min, min_trades=min_trades)
    summary = {
        symbol: [{"horizon_candles": item["horizon_candles"],
                  "tp_atr_mult": item["tp_atr_mult"],
                  **{key: item["metrics"][key] for key in SUMMARY_KEYS}}
                 for item in results]
        for symbol, results in report["symbols"].items()
    }
    typer.echo(json.dumps({"report": str(out / "report.json"), "summary": summary,
                           "selection": report["selection"]},
                          indent=2, default=str))


@app.command("meta-experiment")
def meta_experiment(
    holdout_start: str = typer.Option("2025-10-01", help="Rows from this date are excluded"),
    primaries: str = typer.Option("breakout_20,momentum_42", help="Comma-separated rules"),
    horizons: str = typer.Option("6,12", help="Comma-separated horizons in candles"),
    tp_atr_mults: str = typer.Option("3.0", help="Comma-separated take-profit ATR mults"),
    t_min: float = typer.Option(3.0, help="Minimum non-overlapping t-statistic"),
    min_trades: int = typer.Option(40, help="Minimum non-overlapping trades per symbol"),
) -> None:
    """Purged-CV grid for a rule + model filter, using pre-holdout data only."""
    import pandas as pd

    from trader_bot.models.meta import PRIMARIES, run_meta_experiment

    _, cfg, engine = _ctx()
    rules = [value.strip() for value in primaries.split(",") if value.strip()]
    if not rules or not set(rules) <= set(PRIMARIES):
        raise typer.BadParameter(f"primaries must be in {list(PRIMARIES)}")
    try:
        tps = [float(value) for value in tp_atr_mults.split(",")]
        hs = [int(value) for value in horizons.split(",")]
    except ValueError as exc:
        raise typer.BadParameter("use comma-separated numbers") from exc
    if any(value <= 0 for value in tps) or any(value <= 0 for value in hs):
        raise typer.BadParameter("values must be positive")
    holdout = pd.Timestamp(_parse_utc(holdout_start, "holdout_start"))
    out = Path("artifacts/experiments") / datetime.now(UTC).strftime("meta-%Y%m%dT%H%M%SZ")
    report = run_meta_experiment(engine, cfg, out, holdout, rules, hs, tps, t_min, min_trades)
    typer.echo(json.dumps({"report": str(out / "report.json"),
                           "selection": report["selection"]}, indent=2, default=str))


@app.command("barrier-backtest")
def barrier_backtest(
    holdout_start: str | None = typer.Option(None, help="Split report into validation/holdout"),
    out: Path | None = typer.Option(None, help="Output directory"),  # noqa: B008
) -> None:
    """Walk-forward backtest with barrier labels and the expected-value signal rule."""
    import pandas as pd

    from trader_bot.backtest.run_barrier import run_barrier_backtest

    _, cfg, engine = _ctx()
    if cfg.barrier.tp_atr_mult / cfg.risk.atr_stop_mult < cfg.risk.min_reward_risk:
        raise typer.BadParameter("barrier.tp_atr_mult / atr_stop_mult is below min_reward_risk")
    holdout = pd.Timestamp(_parse_utc(holdout_start, "holdout_start")) if holdout_start else None
    out = out or Path("artifacts/backtest_barrier")
    report = run_barrier_backtest(engine, cfg, out, holdout)
    typer.echo(json.dumps({"report": str(out / "report.json"),
                           "segments": {key: value["portfolio"]
                                        for key, value in report["segments"].items()}},
                          indent=2, default=str))


@app.command()
def baselines(
    holdout_start: str | None = typer.Option(None, help="Split report into validation/holdout"),
    out: Path = typer.Option(Path("artifacts/baselines"), help="Output directory"),  # noqa: B008
) -> None:
    """Rule-based baselines using the same simulator, sizing, and costs as the ML backtest."""
    import pandas as pd

    from trader_bot.backtest.baselines import run_baselines

    _, cfg, engine = _ctx()
    holdout = pd.Timestamp(_parse_utc(holdout_start, "holdout_start")) if holdout_start else None
    report = run_baselines(engine, cfg, out, holdout)
    summary = {name: {segment: result["portfolio"] for segment, result in segments.items()}
               for name, segments in report["baselines"].items()}
    typer.echo(json.dumps(summary, indent=2, default=str))


@app.command()
def run(testnet: bool = typer.Option(True, "--testnet/--live")) -> None:
    """Start the trading loop. Only --testnet is supported in this phase."""
    from trader_bot.execution.client import BinanceSdkClient
    from trader_bot.scheduler.runner import run_live

    settings, cfg, engine = _ctx()
    if not testnet or not settings.binance_testnet:
        typer.echo("Live trading is out of scope (Stage 5): set BINANCE_TESTNET=true and use --testnet", err=True)
        raise typer.Exit(2)
    if not settings.binance_api_key.get_secret_value():
        typer.echo("BINANCE_API_KEY / BINANCE_API_SECRET missing in .env", err=True)
        raise typer.Exit(2)
    client = BinanceSdkClient(settings.binance_api_key.get_secret_value(),
                              settings.binance_api_secret.get_secret_value(), testnet=True)
    asyncio.run(run_live(cfg, settings, engine, client))


@app.command("killswitch-reset")
def killswitch_reset() -> None:
    """Manually clear the kill-switch lock after reviewing what happened."""
    from trader_bot.risk.kill_switch import KillSwitch

    settings, cfg, _ = _ctx()
    ks = KillSwitch(settings.state_dir, cfg.risk.daily_loss_limit, cfg.risk.weekly_loss_limit)
    info = ks.lock_info()
    ks.reset()
    typer.echo(f"kill switch cleared (was: {info})")


if __name__ == "__main__":
    app()
