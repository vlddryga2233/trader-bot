import json
from datetime import UTC, datetime

import pytest

from trader_bot.config.settings import RiskCfg, load_strategy
from trader_bot.risk import limits
from trader_bot.risk.engine import (
    Rejection,
    Signal,
    TradePlan,
    make_ev_signal,
    make_signal,
    plan_trade,
    round_trip_cost,
)
from trader_bot.risk.kill_switch import KillSwitch

CFG = RiskCfg(risk_per_trade=0.01, leverage=3, atr_stop_mult=2.0, min_reward_risk=1.0)


def test_signal_requires_confidence_and_agreeing_regressor():
    assert make_signal(0.05, 0.15, 0.80, 0.02, 0.75) == Signal(1, 0.80, 0.02)
    assert make_signal(0.05, 0.25, 0.70, 0.02, 0.75) is None       # low confidence
    assert make_signal(0.80, 0.15, 0.05, 0.02, 0.75) is None       # regressor disagrees
    assert make_signal(0.05, 0.90, 0.05, 0.0, 0.75) is None        # flat


def test_ev_signal_long_when_expected_value_positive():
    sig = make_ev_signal(0.5, 0.1, 100.0, 1.0, 2.0, 3.0, 0.0, 0.05)
    assert sig is not None and sig.direction == 1
    assert sig.pred_ret == pytest.approx(0.03)


def test_ev_signal_none_when_expected_value_negative():
    assert make_ev_signal(0.35, 0.1, 100.0, 1.0, 2.0, 3.0, 0.0, 0.05) is None


def test_ev_signal_costs_can_veto():
    assert make_ev_signal(0.44, 0.1, 100.0, 1.0, 2.0, 3.0, 0.0, 0.05) is not None
    assert make_ev_signal(0.44, 0.1, 100.0, 1.0, 2.0, 3.0, 0.0014, 0.05) is None


def test_ev_signal_short_side_and_plan_tp():
    sig = make_ev_signal(0.1, 0.6, 100.0, 1.0, 2.0, 3.0, 0.0, 0.05)
    assert sig.direction == -1
    plan = plan_trade(sig, 10_000, 100.0, 1.0, RiskCfg(min_reward_risk=1.5))
    assert not isinstance(plan, Rejection)
    assert plan.take_profit == pytest.approx(97.0)


def test_round_trip_cost():
    assert round_trip_cost(0.0005, 2.0) == pytest.approx(0.0014)


def test_position_size_risks_one_percent():
    sig = Signal(1, 0.9, 0.05)
    plan = plan_trade(sig, equity=10_000, price=100.0, atr=1.0, cfg=CFG)
    assert isinstance(plan, TradePlan)
    assert plan.stop == pytest.approx(98.0)
    assert plan.qty * (plan.entry - plan.stop) == pytest.approx(100.0)  # 1% of 10k
    assert plan.take_profit == pytest.approx(105.0)


def test_short_plan_mirrors_long():
    plan = plan_trade(Signal(-1, 0.9, -0.05), 10_000, 100.0, 1.0, CFG)
    assert plan.stop == pytest.approx(102.0) and plan.take_profit == pytest.approx(95.0)


def test_leverage_cap_shrinks_position():
    # tight stop would need 10x leverage for 1% risk; must be clamped to 3x
    plan = plan_trade(Signal(1, 0.9, 0.5), 10_000, 100.0, 0.05, CFG.model_copy(update={"atr_stop_mult": 1.0}))
    assert plan.notional <= 10_000 * 3 + 1e-6
    assert plan.risk_amount < 100.0


def test_config_cannot_exceed_hard_limits():
    wild = RiskCfg(risk_per_trade=0.5, leverage=50, max_open_positions=99,
                   daily_loss_limit=0.9, weekly_loss_limit=0.9).clamped()
    assert wild.risk_per_trade == limits.MAX_RISK_PER_TRADE
    assert wild.leverage == limits.MAX_LEVERAGE
    assert wild.max_open_positions == limits.MAX_OPEN_POSITIONS
    assert wild.daily_loss_limit == limits.MAX_DAILY_LOSS_LIMIT
    # plan_trade clamps independently of config clamping
    plan = plan_trade(Signal(1, 0.9, 0.5), 10_000, 100.0, 0.5,
                      RiskCfg(risk_per_trade=0.5, leverage=50, min_reward_risk=0.1))
    assert plan.leverage == limits.MAX_LEVERAGE
    assert plan.risk_amount <= 10_000 * limits.MAX_RISK_PER_TRADE + 1e-6


def test_rejections():
    s = Signal(1, 0.9, 0.001)
    assert isinstance(plan_trade(s, 10_000, 100.0, 1.0, CFG), Rejection)            # R/R too low
    assert isinstance(plan_trade(Signal(1, 0.9, 0.05), 10_000, 100.0, 1.0, CFG,
                                 open_positions=2), Rejection)                       # max positions
    assert isinstance(plan_trade(Signal(1, 0.9, 0.05), 10_000, 100.0, 0.0, CFG), Rejection)


def test_kill_switch_daily_and_weekly(tmp_path):
    ks = KillSwitch(tmp_path, daily_limit=0.03, weekly_limit=0.07)
    day = lambda d: datetime(2024, 1, d, 10, tzinfo=UTC)  # Jan 1 2024 is a Monday
    assert ks.check(10_000, day(1)) is None                    # anchors set
    assert ks.check(9_800, day(1)) is None                     # -2% day
    assert "daily" in ks.check(9_690, day(1))                  # -3.1% day
    ks2 = KillSwitch(tmp_path / "w", daily_limit=0.03, weekly_limit=0.05)
    assert ks2.check(10_000, day(1)) is None
    assert ks2.check(9_800, day(2)) is None
    assert ks2.check(9_600, day(3)) is None                    # -2.04% day, -4% week
    assert "weekly" in ks2.check(9_400, day(4))                # -2.08% day, -6% week


def test_kill_switch_lock_persists_until_reset(tmp_path):
    ks = KillSwitch(tmp_path, 0.03, 0.07)
    assert not ks.is_locked()
    ks.lock("test")
    assert KillSwitch(tmp_path, 0.03, 0.07).is_locked()        # survives "restart"
    assert json.loads(ks.lock_path.read_text())["reason"] == "test"
    ks.reset()
    assert not ks.is_locked()


def test_strategy_yaml_loads_and_is_clamped(tmp_path):
    cfg = load_strategy("config/strategy.yaml")
    assert cfg.risk.leverage <= limits.MAX_LEVERAGE and cfg.symbols == ["BTCUSDT", "ETHUSDT"]


def test_barrier_mode_rejects_tp_below_reward_risk_floor(tmp_path):
    path = tmp_path / "s.yaml"
    path.write_text("signal_mode: barrier\nbarrier: {tp_atr_mult: 2.0}\n"
                    "risk: {atr_stop_mult: 2.0, min_reward_risk: 1.5}\n")
    with pytest.raises(ValueError):
        load_strategy(path)
