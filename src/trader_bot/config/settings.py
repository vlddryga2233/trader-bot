from __future__ import annotations

from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel, Field, SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict

from trader_bot.risk import limits


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    binance_api_key: SecretStr = SecretStr("")
    binance_api_secret: SecretStr = SecretStr("")
    binance_testnet: bool = True
    database_url: str = "postgresql+psycopg://trader:trader@localhost:5432/trader"
    glassnode_api_key: SecretStr = SecretStr("")
    cryptoquant_api_key: SecretStr = SecretStr("")
    coinglass_api_key: SecretStr = SecretStr("")
    api_briks_api_key: SecretStr = SecretStr("")
    strategy_config: Path = Path("config/strategy.yaml")
    state_dir: Path = Path("state")
    artifacts_dir: Path = Path("artifacts")


class FeatureGroups(BaseModel):
    price: bool = True
    derivatives: bool = True
    liquidations: bool = False
    onchain: bool = False
    cross_asset: bool = False
    calendar: bool = False
    derivs_extra: bool = False


class FeaturesCfg(BaseModel):
    groups: FeatureGroups = FeatureGroups()
    onchain_publication_lag_hours: int = 24


class LabelingCfg(BaseModel):
    horizon_candles: int = 6
    flat_atr_mult: float = 1.0


class ModelCfg(BaseModel):
    confidence_threshold: float = Field(default=0.75, ge=0.0, le=1.0)
    class_balance: bool = False
    promotion_min_balanced_accuracy: float = Field(default=0.38, ge=0.0, le=1.0)
    promotion_min_signal_precision: float = Field(default=0.40, ge=0.0, le=1.0)
    promotion_min_signal_coverage: float = Field(default=0.005, ge=0.0, le=1.0)
    embargo_candles: int = 6
    cv_splits: int = 5
    min_train_candles: int = 1500
    retrain_every_candles: int = 540
    optuna_trials: int = 30


class RiskCfg(BaseModel):
    risk_per_trade: float = 0.01
    leverage: int = 3
    atr_stop_mult: float = 2.0
    min_reward_risk: float = 1.5
    max_open_positions: int = 2
    daily_loss_limit: float = 0.03
    weekly_loss_limit: float = 0.07

    def clamped(self) -> RiskCfg:
        return RiskCfg(
            risk_per_trade=min(self.risk_per_trade, limits.MAX_RISK_PER_TRADE),
            leverage=min(self.leverage, limits.MAX_LEVERAGE),
            atr_stop_mult=max(self.atr_stop_mult, limits.MIN_ATR_STOP_MULT),
            min_reward_risk=self.min_reward_risk,
            max_open_positions=min(self.max_open_positions, limits.MAX_OPEN_POSITIONS),
            daily_loss_limit=min(self.daily_loss_limit, limits.MAX_DAILY_LOSS_LIMIT),
            weekly_loss_limit=min(self.weekly_loss_limit, limits.MAX_WEEKLY_LOSS_LIMIT),
        )


class BarrierCfg(BaseModel):
    tp_atr_mult: float = Field(default=3.0, gt=0)
    min_ev_r: float = Field(default=0.05, ge=0)
    calib_frac: float = Field(default=0.2, gt=0, lt=0.5)
    promotion_min_signals: int = Field(default=30, ge=1)
    promotion_min_mean_r: float = 0.05
    model_size: Literal["default", "small"] = "default"


class BacktestCfg(BaseModel):
    taker_fee: float = 0.0005
    slippage_bps: float = 2.0
    initial_equity: float = 10000
    start: str = "2023-10-01"


class StrategyConfig(BaseModel):
    symbols: list[str] = Field(default_factory=lambda: ["BTCUSDT", "ETHUSDT"])
    decision_interval: str = "4h"
    history_start: str = "2022-06-01"
    features: FeaturesCfg = FeaturesCfg()
    labeling: LabelingCfg = LabelingCfg()
    model: ModelCfg = ModelCfg()
    risk: RiskCfg = RiskCfg()
    backtest: BacktestCfg = BacktestCfg()
    signal_mode: Literal["classic", "barrier"] = "classic"
    barrier: BarrierCfg = BarrierCfg()


def load_strategy(path: Path | str) -> StrategyConfig:
    data = yaml.safe_load(Path(path).read_text()) or {}
    cfg = StrategyConfig.model_validate(data)
    cfg.risk = cfg.risk.clamped()
    if cfg.signal_mode == "barrier" and (
        cfg.barrier.tp_atr_mult / cfg.risk.atr_stop_mult < cfg.risk.min_reward_risk
    ):
        raise ValueError("barrier.tp_atr_mult / risk.atr_stop_mult is below risk.min_reward_risk")
    return cfg
