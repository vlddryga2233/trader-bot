from __future__ import annotations

import numpy as np
import pandas as pd
from catboost import CatBoostClassifier, CatBoostRegressor
from lightgbm import LGBMClassifier, LGBMRegressor

N_CLASSES = 3  # 0 short, 1 flat, 2 long
PROB_COLS = ["p_short", "p_flat", "p_long"]

DEFAULT_PARAMS = {
    "lgbm": {"n_estimators": 400, "learning_rate": 0.03, "num_leaves": 31,
             "subsample": 0.8, "subsample_freq": 1, "colsample_bytree": 0.7,
             "min_child_samples": 40},
    "catboost": {"iterations": 400, "learning_rate": 0.05, "depth": 6},
}


def _lgbm(kind: str, params: dict, seed: int, class_balance: bool = False):
    cls = LGBMClassifier if kind == "clf" else LGBMRegressor
    options = {**DEFAULT_PARAMS["lgbm"], **params}
    if kind == "clf" and class_balance:
        options["class_weight"] = "balanced"
    return cls(**options, random_state=seed, verbose=-1)


def _cat(kind: str, params: dict, seed: int, class_balance: bool = False):
    p = {**DEFAULT_PARAMS["catboost"], **params}
    if kind == "clf":
        if class_balance:
            p["auto_class_weights"] = "Balanced"
        return CatBoostClassifier(**p, loss_function="MultiClass", random_seed=seed, verbose=0)
    return CatBoostRegressor(**p, loss_function="RMSE", random_seed=seed, verbose=0)


def _align_proba(model, proba: np.ndarray) -> np.ndarray:
    """Place predict_proba columns into fixed (short, flat, long) order."""
    out = np.zeros((proba.shape[0], N_CLASSES))
    for j, c in enumerate(np.asarray(model.classes_).astype(int).ravel()):
        out[:, c] = proba[:, j]
    return out


class Ensemble:
    """Average of LightGBM + CatBoost, for direction probabilities and expected return."""

    def __init__(self, columns: list[str], params: dict | None = None, seed: int = 42,
                 class_balance: bool = False):
        self.columns = columns
        self.params = params or {}
        self.seed = seed
        self.class_balance = class_balance
        self.clf: list = []
        self.reg: list = []

    def fit(self, X: pd.DataFrame, y_class: pd.Series, y_ret: pd.Series) -> Ensemble:
        Xc = X[self.columns]
        p = self.params
        self.clf = [_lgbm("clf", p.get("lgbm", {}), self.seed, self.class_balance),
                    _cat("clf", p.get("catboost", {}), self.seed, self.class_balance)]
        self.reg = [_lgbm("reg", p.get("lgbm", {}), self.seed),
                    _cat("reg", p.get("catboost", {}), self.seed)]
        for m in self.clf:
            m.fit(Xc, y_class)
        for m in self.reg:
            m.fit(Xc, y_ret)
        return self

    def predict(self, X: pd.DataFrame) -> pd.DataFrame:
        Xc = X[self.columns]
        proba = np.mean([_align_proba(m, m.predict_proba(Xc)) for m in self.clf], axis=0)
        ret = np.mean([np.ravel(m.predict(Xc)) for m in self.reg], axis=0)
        out = pd.DataFrame(proba, index=X.index, columns=PROB_COLS)
        out["pred_ret"] = ret
        return out
