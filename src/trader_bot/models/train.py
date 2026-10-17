from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import optuna
import pandas as pd
from sklearn.metrics import balanced_accuracy_score, log_loss

from trader_bot.config.settings import StrategyConfig
from trader_bot.features.builder import model_columns
from trader_bot.labeling.labels import make_labels, to_class
from trader_bot.models.cv import purged_walk_forward
from trader_bot.models.ensemble import PROB_COLS, Ensemble, _cat, _lgbm

CORE_FEATURES = ("ret_42", "vol_42", "atr_pct")


@dataclass
class Dataset:
    X: pd.DataFrame          # all rows with usable features (incl. latest rows w/o labels)
    y_class: pd.Series       # 0/1/2, -1 where unlabeled
    y_ret: pd.Series
    has_label: pd.Series     # bool
    columns: list[str]

    @property
    def labeled(self) -> np.ndarray:
        return np.flatnonzero(self.has_label.to_numpy())


def build_dataset(feats: pd.DataFrame, cfg: StrategyConfig) -> Dataset:
    valid = feats.dropna(subset=list(CORE_FEATURES))
    lab = make_labels(valid, cfg.labeling.horizon_candles, cfg.labeling.flat_atr_mult)
    has = lab["label"].notna()
    y_class = pd.Series(-1, index=valid.index)
    y_class[has] = to_class(lab.loc[has, "label"])
    return Dataset(valid, y_class, lab["fwd_ret"], has, model_columns(valid))


def _train_idx(ds: Dataset, upto: int) -> np.ndarray:
    idx = np.arange(0, upto)
    return idx[ds.has_label.to_numpy()[idx]]


def cv_evaluate(ds: Dataset, cfg: StrategyConfig, params: dict | None = None) -> dict:
    """Purged walk-forward evaluation of the full ensemble on labeled rows."""
    lab = ds.labeled
    X, yc, yr = ds.X.iloc[lab], ds.y_class.iloc[lab], ds.y_ret.iloc[lab]
    m = cfg.model
    probs, y_true, rets, y_ret_true, fold_metrics = [], [], [], [], []
    for fold, (tr, te) in enumerate(
        purged_walk_forward(len(X), m.cv_splits, cfg.labeling.horizon_candles,
                            m.embargo_candles, m.min_train_candles), start=1
    ):
        ens = Ensemble(ds.columns, params, class_balance=m.class_balance).fit(
            X.iloc[tr], yc.iloc[tr], yr.iloc[tr]
        )
        pred = ens.predict(X.iloc[te])
        fold_probs = pred[PROB_COLS].to_numpy()
        fold_pred = fold_probs.argmax(axis=1)
        probs.append(fold_probs)
        rets.append(pred["pred_ret"].to_numpy())
        fold_y = yc.iloc[te].to_numpy()
        y_true.append(fold_y)
        y_ret_true.append(yr.iloc[te].to_numpy())
        fold_metrics.append({
            "fold": fold,
            "n_train": len(tr),
            "n_test": len(te),
            "train_class_counts": {
                name: int((yc.iloc[tr].to_numpy() == cls).sum())
                for cls, name in enumerate(("short", "flat", "long"))
            },
            "test_class_counts": {
                name: int((fold_y == cls).sum())
                for cls, name in enumerate(("short", "flat", "long"))
            },
            "predicted_test_class_counts": {
                name: int((fold_pred == cls).sum())
                for cls, name in enumerate(("short", "flat", "long"))
            },
        })
    P, y, r, yr_t = map(np.concatenate, (probs, y_true, rets, y_ret_true))
    pred = P.argmax(axis=1)
    conf = P.max(axis=1)
    hc = conf >= m.confidence_threshold
    directional = pred != 1
    directional_hc = directional & (conf >= m.confidence_threshold)
    return_agreement = ((pred == 0) & (r < 0)) | ((pred == 2) & (r > 0))
    directional_signal = directional_hc & return_agreement
    class_names = ("short", "flat", "long")
    class_metrics = {}
    true_counts, pred_counts = {}, {}
    recalls = []
    for cls, name in enumerate(class_names):
        actual = y == cls
        predicted = pred == cls
        support = int(actual.sum())
        predicted_count = int(predicted.sum())
        correct = int((actual & predicted).sum())
        precision = correct / predicted_count if predicted_count else None
        recall = correct / support if support else None
        class_metrics[name] = {"precision": precision, "recall": recall, "support": support}
        true_counts[name] = support
        pred_counts[name] = predicted_count
        if recall is not None:
            recalls.append(recall)
    return {
        "logloss": float(log_loss(y, P, labels=[0, 1, 2])),
        "accuracy": float((pred == y).mean()),
        "balanced_accuracy": float(np.mean(recalls)) if recalls else None,
        "oos_class_counts": true_counts,
        "oos_predicted_class_counts": pred_counts,
        "cv_folds": fold_metrics,
        "per_class": class_metrics,
        "hc_coverage": float(hc.mean()),
        "hc_precision": float((pred[hc] == y[hc]).mean()) if hc.any() else None,
        "directional_prediction_rate": float(directional.mean()),
        "directional_hc_coverage": float(directional_hc.mean()),
        "directional_hc_precision": (
            float((pred[directional_hc] == y[directional_hc]).mean())
            if directional_hc.any() else None
        ),
        "directional_signal_coverage": float(directional_signal.mean()),
        "directional_signal_precision": (
            float((pred[directional_signal] == y[directional_signal]).mean())
            if directional_signal.any() else None
        ),
        "ret_corr": float(np.corrcoef(r, yr_t)[0, 1]),
        "n_oos": len(y),
    }


def tune(ds: Dataset, cfg: StrategyConfig, trials: int | None = None) -> dict:
    """Tune classifier parameters for balanced directional classification."""
    lab = ds.labeled
    X, yc = ds.X.iloc[lab][ds.columns], ds.y_class.iloc[lab]
    m = cfg.model
    splits = list(purged_walk_forward(len(X), m.cv_splits, cfg.labeling.horizon_candles,
                                      m.embargo_candles, m.min_train_candles))

    def score(build) -> float:
        scores = []
        for tr, te in splits:
            model = build().fit(X.iloc[tr], yc.iloc[tr])
            predicted = model.predict(X.iloc[te]).astype(int).ravel()
            scores.append(balanced_accuracy_score(yc.iloc[te], predicted))
        return float(1.0 - np.mean(scores))

    def lgbm_obj(t: optuna.Trial) -> float:
        p = {"learning_rate": t.suggest_float("learning_rate", 0.01, 0.1, log=True),
             "num_leaves": t.suggest_int("num_leaves", 8, 64),
             "min_child_samples": t.suggest_int("min_child_samples", 20, 200),
             "colsample_bytree": t.suggest_float("colsample_bytree", 0.4, 1.0),
             "n_estimators": t.suggest_int("n_estimators", 150, 600)}
        return score(lambda: _lgbm("clf", p, 42, m.class_balance))

    def cat_obj(t: optuna.Trial) -> float:
        p = {"learning_rate": t.suggest_float("learning_rate", 0.01, 0.15, log=True),
             "depth": t.suggest_int("depth", 3, 8),
             "l2_leaf_reg": t.suggest_float("l2_leaf_reg", 1.0, 10.0),
             "iterations": t.suggest_int("iterations", 150, 600)}
        return score(lambda: _cat("clf", p, 42, m.class_balance))

    n = trials or m.optuna_trials
    best = {}
    for name, obj in (("lgbm", lgbm_obj), ("catboost", cat_obj)):
        study = optuna.create_study(direction="minimize",
                                    sampler=optuna.samplers.TPESampler(seed=42))
        study.optimize(obj, n_trials=n)
        best[name] = study.best_params
    return best


def fit_final(ds: Dataset, params: dict | None = None, class_balance: bool = False) -> Ensemble:
    lab = ds.labeled
    return Ensemble(ds.columns, params, class_balance=class_balance).fit(
        ds.X.iloc[lab], ds.y_class.iloc[lab], ds.y_ret.iloc[lab]
    )


def walk_forward_predictions(
    ds: Dataset, cfg: StrategyConfig, params: dict | None, start: pd.Timestamp
) -> pd.DataFrame:
    """Out-of-sample predictions: models are retrained every N candles on purged past data only."""
    m, h = cfg.model, cfg.labeling.horizon_candles
    first = max(int(ds.X.index.searchsorted(start)), m.min_train_candles)
    out = []
    for s in range(first, len(ds.X), m.retrain_every_candles):
        e = min(s + m.retrain_every_candles, len(ds.X))
        tr = _train_idx(ds, s - h - m.embargo_candles)
        if len(tr) < m.min_train_candles // 2:
            continue
        ens = Ensemble(ds.columns, params, class_balance=m.class_balance).fit(
            ds.X.iloc[tr], ds.y_class.iloc[tr], ds.y_ret.iloc[tr]
        )
        out.append(ens.predict(ds.X.iloc[s:e]))
    return pd.concat(out) if out else pd.DataFrame(columns=[*PROB_COLS, "pred_ret"])
