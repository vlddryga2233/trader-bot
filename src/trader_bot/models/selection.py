"""Selection gates shared by research experiments."""

from __future__ import annotations


def symbol_gates(m: dict, mode: str, t_min: float, min_trades: int) -> dict[str, bool]:
    """Evaluate a symbol's metrics against its mode-specific comparator."""
    if mode == "limit":
        mean, t = m.get("limit_mean_net_r_nonoverlap"), m.get("limit_t_stat_nonoverlap")
        positive_folds = m.get("folds_with_positive_limit_r", 0)
        comparator = m.get("mean_net_r_nonoverlap") or 0.0
    else:
        mean, t = m.get("mean_net_r_nonoverlap"), m.get("t_stat_nonoverlap")
        positive_folds = m.get("folds_with_positive_nonoverlap_r", 0)
        if mode == "meta":
            comparator = m.get("primary_mean_net_r_nonoverlap") or 0.0
        else:
            comparator = max(m["always_long_mean_net_r"], m["always_short_mean_net_r"])
    return {
        "rr_floor": bool(m["rr_meets_risk_floor"]),
        "signals_in_4_folds": m.get("folds_with_signals", 0) >= 4,
        "positive_in_4_folds": positive_folds >= 4,
        "enough_trades": m.get("n_trades_nonoverlap", 0) >= min_trades,
        "t_stat": t is not None and t >= t_min,
        "beats_comparator": mean is not None and mean > max(comparator, 0.0),
    }


def select(report: dict, mode: str, t_min: float = 3.0, min_trades: int = 40) -> list[dict]:
    """Return one joint pass/fail row per experiment cell."""
    symbols = list(report["symbols"])
    if not symbols:
        return []
    cells_by_symbol = [report["symbols"][symbol] for symbol in symbols]
    rows = []
    for cells in zip(*cells_by_symbol, strict=True):
        gates = {
            symbol: symbol_gates(cell["metrics"], mode, t_min, min_trades)
            for symbol, cell in zip(symbols, cells, strict=True)
        }
        row = {key: value for key, value in cells[0].items() if key != "metrics"}
        row["gates_met"] = sum(sum(g.values()) for g in gates.values())
        row["gates_total"] = sum(len(g) for g in gates.values())
        row["passed"] = row["gates_met"] == row["gates_total"]
        row["failed"] = {
            symbol: [name for name, ok in gate.items() if not ok] for symbol, gate in gates.items()
        }
        rows.append(row)
    return rows
