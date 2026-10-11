from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path

LOCK_NAME = "KILL_SWITCH.lock"
ANCHORS_NAME = "anchors.json"


class KillSwitch:
    """Daily / weekly loss limiter with a persistent lock.

    check() only decides; the caller must lock() first, then close everything. A lock file
    survives restarts and is removed only by the manual reset command.
    """

    def __init__(self, state_dir: Path | str, daily_limit: float, weekly_limit: float):
        self.dir = Path(state_dir)
        self.dir.mkdir(parents=True, exist_ok=True)
        self.daily_limit = daily_limit
        self.weekly_limit = weekly_limit

    @property
    def lock_path(self) -> Path:
        return self.dir / LOCK_NAME

    def is_locked(self) -> bool:
        return self.lock_path.exists()

    def lock_info(self) -> dict | None:
        return json.loads(self.lock_path.read_text()) if self.is_locked() else None

    def lock(self, reason: str, now: datetime | None = None) -> None:
        now = now or datetime.now(UTC)
        self.lock_path.write_text(json.dumps({"reason": reason, "at": now.isoformat()}))

    def reset(self) -> None:
        self.lock_path.unlink(missing_ok=True)
        (self.dir / ANCHORS_NAME).unlink(missing_ok=True)

    def _load(self) -> dict:
        p = self.dir / ANCHORS_NAME
        return json.loads(p.read_text()) if p.exists() else {}

    def check(self, equity: float, now: datetime | None = None) -> str | None:
        """Return a reason string if a loss limit is breached, else None."""
        now = now or datetime.now(UTC)
        day_key = now.date().isoformat()
        iso = now.isocalendar()
        week_key = f"{iso.year}-W{iso.week:02d}"
        a = self._load()
        if a.get("day_key") != day_key:
            a["day_key"], a["day_equity"] = day_key, equity
        if a.get("week_key") != week_key:
            a["week_key"], a["week_equity"] = week_key, equity
        (self.dir / ANCHORS_NAME).write_text(json.dumps(a))

        day_loss = (a["day_equity"] - equity) / a["day_equity"]
        week_loss = (a["week_equity"] - equity) / a["week_equity"]
        if day_loss >= self.daily_limit:
            return f"daily loss {day_loss:.2%} >= {self.daily_limit:.2%}"
        if week_loss >= self.weekly_limit:
            return f"weekly loss {week_loss:.2%} >= {self.weekly_limit:.2%}"
        return None
