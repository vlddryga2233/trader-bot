"""Two calibrated binary classifiers: P(long TP first) and P(short TP first)."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from catboost import CatBoostClassifier
from lightgbm import LGBMClassifier
from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import brier_score_loss, roc_auc_score

from trader_bot.config.settings import StrategyConfig
from trader_bot.features.builder import model_columns
from trader_bot.labeling.barriers import make_barrier_labels
from trader_bot.models.cv import purged_walk_forward
from trader_bot.models.ensemble import DEFAULT_PARAMS
from trader_bot.models.train import CORE_FEATURES
from trader_bot.risk.engine import round_trip_cost

BARRIER_PRED_COLS = ["p_long_win", "p_short_win", "p_long_raw", "p_short_raw"]
MIN_CALIBRATION_ROWS = 200
SMALL_PARAMS = {
    "lgbm": {"num_leaves": 15, "min_child_samples": 100, "n_estimators": 300},
    "catboost": {"depth": 4, "l2_leaf_reg": 5.0},
}


@dataclass
class BarrierDataset:
    X: pd.DataFrame
    y_long: pd.Series
    y_short: pd.Series
    r_long: pd.Series
    r_short: pd.Series
    has_label: pd.Series
    columns: list[str]

    @property
    def labeled(self) -> np.ndarray:
        return np.flatnonzero(self.has_label.to_numpy())


def build_barrier_dataset(feats: pd.DataFrame, cfg: StrategyConfig) -> BarrierDataset:
    valid = feats.dropna(subset=[*CORE_FEATURES, "ctx_atr", "ctx_high", "ctx_low"])
    lab = make_barrier_labels(valid, cfg.labeling.horizon_candles, cfg.risk.atr_stop_mult,
                              cfg.barrier.tp_atr_mult)
    has = lab["r_long"].notna() & lab["r_short"].notna()
    return BarrierDataset(valid, lab["y_long"], lab["y_short"], lab["r_long"], lab["r_short"],
                          has, model_columns(valid))


def _binary_models(seed: int, size: str = "default") -> list:
    params = {key: {**value, **(SMALL_PARAMS[key] if size == "small" else {})}
              for key, value in DEFAULT_PARAMS.items()}
    return [
        LGBMClassifier(**params["lgbm"], random_state=seed, verbose=-1),
        CatBoostClassifier(**params["catboost"], loss_function="Logloss",
                           random_seed=seed, verbose=0),
    ]


class BarrierModel:
    """Per side: LightGBM + CatBoost averaged, then isotonic calibration on a purged tail."""

    def __init__(self, columns: list[str], purge: int, calib_frac: float = 0.2, seed: int = 42,
                 size: str = "default"):
        self.columns = columns
        self.purge = purge
        self.calib_frac = calib_frac
        self.seed = seed
        self.size = size
        self.models: dict[str, list] = {}
        self.calibrators: dict[str, IsotonicRegression | None] = {}

    def _raw(self, side: str, Xc: pd.DataFrame) -> np.ndarray:
        return np.mean([m.predict_proba(Xc)[:, 1] for m in self.models[side]], axis=0)

    def fit(self, X: pd.DataFrame, y_long: pd.Series, y_short: pd.Series) -> BarrierModel:
        """X rows must be time-ordered and fully labeled."""
        Xc = X[self.columns]
        n = len(Xc)
        cal_start = int(n * (1 - self.calib_frac))
        fit_end = cal_start - self.purge
        use_calibration = n - cal_start >= MIN_CALIBRATION_ROWS and fit_end > 0
        if not use_calibration:
            fit_end = n
        for side, y in (("long", y_long), ("short", y_short)):
            y_int = y.astype(int)
            if y_int.iloc[:fit_end].nunique() < 2:
                raise ValueError(f"{side} labels contain a single class")
            self.models[side] = _binary_models(self.seed, self.size)
            for model in self.models[side]:
                model.fit(Xc.iloc[:fit_end], y_int.iloc[:fit_end])
            if use_calibration:
                iso = IsotonicRegression(y_min=0.0, y_max=1.0, out_of_bounds="clip")
                iso.fit(self._raw(side, Xc.iloc[cal_start:]), y_int.iloc[cal_start:])
                self.calibrators[side] = iso
            else:
                self.calibrators[side] = None
        return self

    def predict(self, X: pd.DataFrame) -> pd.DataFrame:
        Xc = X[self.columns]
        out = pd.DataFrame(index=X.index)
        for side in ("long", "short"):
            raw = self._raw(side, Xc)
            iso = self.calibrators[side]
            out[f"p_{side}_raw"] = raw
            out[f"p_{side}_win"] = iso.predict(raw) if iso is not None else raw
        return out[BARRIER_PRED_COLS]


def _auc(y: np.ndarray, p: np.ndarray) -> float | None:
    return float(roc_auc_score(y, p)) if len(np.unique(y)) == 2 else None


def _calibration_table(y: np.ndarray, p: np.ndarray, bins: int = 10) -> list[dict]:
    df = pd.DataFrame({"y": y, "p": p})
    df["bin"] = pd.qcut(df["p"], bins, labels=False, duplicates="drop")
    table = df.groupby("bin").agg(mean_pred=("p", "mean"), hit_rate=("y", "mean"),
                                  n=("y", "size"))
    return table.reset_index(drop=True).to_dict(orient="records")


def non_overlapping(signal: np.ndarray, horizon: int) -> np.ndarray:
    """Greedy mask: take a signal, then skip the next horizon-1 rows (one position at a time)."""
    taken = np.zeros(len(signal), dtype=bool)
    next_free = 0
    for i in np.flatnonzero(signal):
        if i >= next_free:
            taken[i] = True
            next_free = i + horizon
    return taken


def _t_stat(values: np.ndarray) -> float | None:
    if len(values) < 2:
        return None
    sd = values.std(ddof=1)
    return float(values.mean() / (sd / np.sqrt(len(values)))) if sd > 0 else None


def barrier_signal_metrics(P: pd.DataFrame, X: pd.DataFrame, y_long: pd.Series,
                           y_short: pd.Series, r_long: pd.Series, r_short: pd.Series,
                           cfg: StrategyConfig) -> dict:
    """Score probabilities and the EV trading rule on out-of-sample rows (all aligned)."""
    stop_mult, tp_mult = cfg.risk.atr_stop_mult, cfg.barrier.tp_atr_mult
    rr = tp_mult / stop_mult
    price = X["ctx_close"].to_numpy(dtype=float)
    atr = X["ctx_atr"].to_numpy(dtype=float)
    cost_r = round_trip_cost(cfg.backtest.taker_fee, cfg.backtest.slippage_bps) * price / (
        stop_mult * atr
    )
    pl, ps = P["p_long_win"].to_numpy(), P["p_short_win"].to_numpy()
    yl, ys = y_long.to_numpy(dtype=int), y_short.to_numpy(dtype=int)
    rl, rs = r_long.to_numpy(dtype=float), r_short.to_numpy(dtype=float)
    ev_long = pl * rr - (1 - pl) - cost_r
    ev_short = ps * rr - (1 - ps) - cost_r
    direction = np.where(ev_long >= ev_short, 1, -1)
    signal = np.maximum(ev_long, ev_short) >= cfg.barrier.min_ev_r
    realized = np.where(direction == 1, rl, rs) - cost_r
    hits = np.where(direction == 1, yl, ys)
    n_signals = int(signal.sum())
    trades = non_overlapping(signal, cfg.labeling.horizon_candles)
    trade_r = realized[trades]
    return {
        "n_oos": len(P),
        "reward_risk": rr,
        "rr_meets_risk_floor": rr >= cfg.risk.min_reward_risk,
        "base_rate_long_win": float(yl.mean()),
        "base_rate_short_win": float(ys.mean()),
        "auc_long": _auc(yl, pl),
        "auc_short": _auc(ys, ps),
        "brier_long": float(brier_score_loss(yl, pl)),
        "brier_short": float(brier_score_loss(ys, ps)),
        "calibration_long": _calibration_table(yl, pl),
        "calibration_short": _calibration_table(ys, ps),
        "n_signals": n_signals,
        "n_long_signals": int((signal & (direction == 1)).sum()),
        "n_short_signals": int((signal & (direction == -1)).sum()),
        "signal_coverage": n_signals / len(P) if len(P) else 0.0,
        "hit_rate_on_signals": float(hits[signal].mean()) if n_signals else None,
        "mean_net_r_on_signals": float(realized[signal].mean()) if n_signals else None,
        "n_trades_nonoverlap": int(trades.sum()),
        "mean_net_r_nonoverlap": float(trade_r.mean()) if len(trade_r) else None,
        "t_stat_nonoverlap": _t_stat(trade_r),
        "always_long_mean_net_r": float(np.mean(rl - cost_r)),
        "always_short_mean_net_r": float(np.mean(rs - cost_r)),
    }


def cv_evaluate_barrier(ds: BarrierDataset, cfg: StrategyConfig) -> dict:
    """Purged walk-forward CV of the barrier model and the EV signal rule."""
    lab = ds.labeled
    X = ds.X.iloc[lab]
    yl, ys = ds.y_long.iloc[lab], ds.y_short.iloc[lab]
    rl, rs = ds.r_long.iloc[lab], ds.r_short.iloc[lab]
    m, h = cfg.model, cfg.labeling.horizon_candles
    preds, folds = [], []
    for fold, (tr, te) in enumerate(
        purged_walk_forward(len(X), m.cv_splits, h, m.embargo_candles, m.min_train_candles),
        start=1,
    ):
        model = BarrierModel(ds.columns, purge=h + m.embargo_candles,
                             calib_frac=cfg.barrier.calib_frac,
                             size=cfg.barrier.model_size).fit(
            X.iloc[tr], yl.iloc[tr], ys.iloc[tr]
        )
        prediction = model.predict(X.iloc[te])
        preds.append(prediction)
        fold_metrics = barrier_signal_metrics(prediction, X.iloc[te], yl.iloc[te], ys.iloc[te],
                                              rl.iloc[te], rs.iloc[te], cfg)
        folds.append({
            "fold": fold, "n_train": len(tr), "n_test": len(te),
            **{key: fold_metrics[key] for key in FOLD_KEYS},
        })
    P = pd.concat(preds)
    idx = P.index
    metrics = barrier_signal_metrics(P, X.loc[idx], yl.loc[idx], ys.loc[idx], rl.loc[idx],
                                     rs.loc[idx], cfg)
    return summarize_folds(metrics, folds)


FOLD_KEYS = ("auc_long", "auc_short", "n_signals", "hit_rate_on_signals",
             "mean_net_r_on_signals", "n_trades_nonoverlap", "mean_net_r_nonoverlap")


def summarize_folds(metrics: dict, folds: list[dict]) -> dict:
    def positive(key: str) -> int:
        return sum(1 for fold in folds if fold[key] is not None and fold[key] > 0)

    metrics["cv_folds"] = folds
    metrics["folds_with_positive_mean_r"] = positive("mean_net_r_on_signals")
    metrics["folds_with_positive_nonoverlap_r"] = positive("mean_net_r_nonoverlap")
    metrics["folds_with_signals"] = sum(1 for fold in folds if fold["n_signals"] > 0)
    return metrics


def walk_forward_barrier_predictions(ds: BarrierDataset, cfg: StrategyConfig,
                                     start: pd.Timestamp) -> pd.DataFrame:
    """Out-of-sample predictions; retrain every N candles using only purged past data."""
    m, h = cfg.model, cfg.labeling.horizon_candles
    first = max(int(ds.X.index.searchsorted(start)), m.min_train_candles)
    has = ds.has_label.to_numpy()
    out = []
    for s in range(first, len(ds.X), m.retrain_every_candles):
        e = min(s + m.retrain_every_candles, len(ds.X))
        idx = np.arange(0, max(s - h - m.embargo_candles, 0))
        idx = idx[has[idx]]
        if len(idx) < m.min_train_candles // 2:
            continue
        model = BarrierModel(ds.columns, purge=h + m.embargo_candles,
                             calib_frac=cfg.barrier.calib_frac,
                             size=cfg.barrier.model_size).fit(
            ds.X.iloc[idx], ds.y_long.iloc[idx], ds.y_short.iloc[idx]
        )
        out.append(model.predict(ds.X.iloc[s:e]))
    return pd.concat(out) if out else pd.DataFrame(columns=BARRIER_PRED_COLS)
