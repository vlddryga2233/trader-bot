from __future__ import annotations

import json
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

    def should_promote(self, symbol: str, new_metrics: dict, tolerance: float = 0.0) -> bool:
        """Promote only if out-of-sample logloss is no worse than the current model's."""
        cur = self.current_version(symbol)
        if cur is None:
            return True
        return new_metrics["logloss"] <= self.meta(symbol, cur)["metrics"]["logloss"] + tolerance

    def load_current(self, symbol: str) -> tuple[Ensemble, str]:
        v = self.current_version(symbol)
        if v is None:
            raise FileNotFoundError(f"no promoted model for {symbol}; run `trader-bot train`")
        return joblib.load(self._dir(symbol) / v / "ensemble.joblib"), v
