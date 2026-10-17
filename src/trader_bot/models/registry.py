from __future__ import annotations

import json
import math
from datetime import UTC, datetime
from pathlib import Path

import joblib

from trader_bot.models.ensemble import Ensemble


class ModelRegistry:
    """Versioned artifacts: <root>/<symbol>/<version>/{ensemble.joblib,meta.json}; current.json points to live one."""

    def __init__(self, root: Path | str):
        self.root = Path(root) / "models"

    def _dir(self, symbol: str) -> Path:
        return self.root / symbol

    def save(self, symbol: str, ens: Ensemble, metrics: dict, extra: dict | None = None) -> str:
        version = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
        d = self._dir(symbol) / version
        d.mkdir(parents=True, exist_ok=True)
        joblib.dump(ens, d / "ensemble.joblib")
        meta = {"symbol": symbol, "version": version, "columns": ens.columns,
                "metrics": metrics, **(extra or {})}
        (d / "meta.json").write_text(json.dumps(meta, indent=2))
        return version

    def current_version(self, symbol: str) -> str | None:
        p = self._dir(symbol) / "current.json"
        return json.loads(p.read_text())["version"] if p.exists() else None

    def meta(self, symbol: str, version: str) -> dict:
        return json.loads((self._dir(symbol) / version / "meta.json").read_text())

    def promote(self, symbol: str, version: str) -> None:
        (self._dir(symbol) / "current.json").write_text(json.dumps({"version": version}))

    def should_promote(
        self,
        symbol: str,
        new_metrics: dict,
        min_balanced_accuracy: float = 0.38,
        min_signal_precision: float = 0.40,
        min_signal_coverage: float = 0.005,
    ) -> bool:
        """Promote only candidates with useful directional quality and coverage."""
        balanced_accuracy = new_metrics.get("balanced_accuracy")
        precision = new_metrics.get("directional_signal_precision")
        coverage = new_metrics.get("directional_signal_coverage")
        if (
            balanced_accuracy is None
            or precision is None
            or coverage is None
            or not all(math.isfinite(value) for value in (balanced_accuracy, precision, coverage))
            or balanced_accuracy < min_balanced_accuracy
            or precision < min_signal_precision
            or coverage < min_signal_coverage
        ):
            return False
        cur = self.current_version(symbol)
        if cur is None:
            return True
        old = self.meta(symbol, cur)["metrics"]
        old_precision = old.get("directional_signal_precision")
        old_coverage = old.get("directional_signal_coverage")
        old_quality = (
            old_precision * old_coverage
            if old_precision is not None and old_coverage is not None
            and math.isfinite(old_precision) and math.isfinite(old_coverage) else None
        )
        new_quality = precision * coverage
        return old_quality is None or new_quality > old_quality

    def load_current(self, symbol: str) -> tuple[Ensemble, str]:
        v = self.current_version(symbol)
        if v is None:
            raise FileNotFoundError(f"no promoted model for {symbol}; run `trader-bot train`")
        return joblib.load(self._dir(symbol) / v / "ensemble.joblib"), v
