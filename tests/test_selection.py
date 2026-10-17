from trader_bot.models.selection import select, symbol_gates


def _m(**over):
    m = {
        "rr_meets_risk_floor": True,
        "folds_with_signals": 5,
        "folds_with_positive_nonoverlap_r": 4,
        "n_trades_nonoverlap": 80,
        "t_stat_nonoverlap": 3.5,
        "mean_net_r_nonoverlap": 0.2,
        "always_long_mean_net_r": -0.05,
        "always_short_mean_net_r": -0.06,
    }
    m.update(over)
    return m


def test_all_gates_pass():
    assert all(symbol_gates(_m(), "barrier", 3.0, 40).values())


def test_each_gate_can_fail():
    assert not symbol_gates(_m(t_stat_nonoverlap=2.9), "barrier", 3.0, 40)["t_stat"]
    assert not symbol_gates(_m(n_trades_nonoverlap=39), "barrier", 3.0, 40)["enough_trades"]
    assert not symbol_gates(_m(mean_net_r_nonoverlap=-0.01), "barrier", 3.0, 40)["beats_comparator"]


def test_meta_mode_compares_with_unfiltered_rule():
    gates = symbol_gates(_m(primary_mean_net_r_nonoverlap=0.25), "meta", 3.0, 40)
    assert not gates["beats_comparator"]


def test_select_requires_every_symbol():
    report = {
        "symbols": {
            "BTCUSDT": [{"horizon_candles": 6, "tp_atr_mult": 3.0, "metrics": _m()}],
            "ETHUSDT": [
                {"horizon_candles": 6, "tp_atr_mult": 3.0, "metrics": _m(folds_with_signals=3)}
            ],
        }
    }
    row = select(report, "barrier")[0]
    assert not row["passed"]
    assert row["failed"]["ETHUSDT"] == ["signals_in_4_folds"]
    assert row["gates_met"] == 11
