import json

from trader_bot.models.registry import ModelRegistry
from trader_bot.models.signal_study import _select_validation_candidate


def _metrics(**overrides):
    return {
        "balanced_accuracy": 0.42,
        "directional_signal_precision": 0.55,
        "directional_signal_coverage": 0.02,
        **overrides,
    }


def test_registry_requires_directional_quality_before_first_promotion(tmp_path):
    registry = ModelRegistry(tmp_path)

    assert registry.should_promote("BTCUSDT", _metrics())
    assert not registry.should_promote(
        "BTCUSDT", _metrics(directional_signal_precision=0.2)
    )
    assert not registry.should_promote("BTCUSDT", _metrics(balanced_accuracy=0.34))
    assert not registry.should_promote(
        "BTCUSDT", _metrics(directional_signal_coverage=float("nan"))
    )


def test_registry_compares_candidate_directional_quality(tmp_path):
    registry = ModelRegistry(tmp_path)
    model_dir = registry._dir("BTCUSDT")
    model_dir.mkdir(parents=True)
    (model_dir / "current.json").write_text(json.dumps({"version": "old"}))
    old_metrics = _metrics(
        directional_signal_precision=0.5,
        directional_signal_coverage=0.01,
    )
    (model_dir / "old").mkdir()
    (model_dir / "old" / "meta.json").write_text(
        json.dumps({"metrics": old_metrics})
    )

    assert registry.should_promote(
        "BTCUSDT",
        _metrics(directional_signal_precision=0.6, directional_signal_coverage=0.01),
    )
    assert not registry.should_promote(
        "BTCUSDT",
        _metrics(directional_signal_precision=0.45, directional_signal_coverage=0.01),
    )


def test_signal_study_rejects_high_trade_count_without_positive_edge():
    weak = {
        "confidence_threshold": 0.3,
        "min_reward_risk": 0.5,
        "portfolio": {
            "n_trades": 26,
            "max_drawdown": -0.02,
            "sharpe": -0.2,
            "profit_factor": 0.9,
        },
    }

    assert _select_validation_candidate([weak]) is None


def test_signal_study_selects_only_candidates_passing_all_gates():
    weak = {
        "confidence_threshold": 0.3,
        "min_reward_risk": 0.5,
        "portfolio": {
            "n_trades": 30,
            "max_drawdown": -0.02,
            "sharpe": 1.2,
            "profit_factor": 0.95,
        },
    }
    viable = {
        "confidence_threshold": 0.5,
        "min_reward_risk": 1.0,
        "portfolio": {
            "n_trades": 24,
            "max_drawdown": -0.04,
            "sharpe": 0.4,
            "profit_factor": 1.2,
        },
    }

    assert _select_validation_candidate([weak, viable]) is viable
