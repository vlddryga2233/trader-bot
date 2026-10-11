from __future__ import annotations

import numpy as np
import pandas as pd


def compute_metrics(equity: pd.Series, trades: pd.DataFrame) -> dict:
    """equity: bar-level equity curve (DatetimeIndex). Sharpe is annualised from daily returns."""
    daily = equity.resample("1D").last().dropna()
    rets = daily.pct_change().dropna()
    sharpe = float(rets.mean() / rets.std() * np.sqrt(365)) if len(rets) > 1 and rets.std() > 0 else 0.0
    peak = equity.cummax()
    max_dd = float(((equity - peak) / peak).min()) if len(equity) else 0.0
    years = max((equity.index[-1] - equity.index[0]).days / 365.25, 1e-9) if len(equity) > 1 else 1e-9
    total = float(equity.iloc[-1] / equity.iloc[0] - 1) if len(equity) else 0.0
    out = {
        "sharpe": sharpe,
        "max_drawdown": max_dd,
        "total_return": total,
        "cagr": float((1 + total) ** (1 / years) - 1) if total > -1 else -1.0,
        "n_trades": int(len(trades)),
    }
    if len(trades):
        wins = trades.loc[trades["pnl"] > 0, "pnl"].sum()
        losses = -trades.loc[trades["pnl"] < 0, "pnl"].sum()
        out["win_rate"] = float((trades["pnl"] > 0).mean())
        out["profit_factor"] = float(wins / losses) if losses > 0 else float("inf")
        out["avg_trade"] = float(trades["pnl"].mean())
    return out


def passes_gate(m: dict, min_sharpe: float = 1.5, max_dd: float = 0.15) -> bool:
    return m["sharpe"] > min_sharpe and abs(m["max_drawdown"]) < max_dd
