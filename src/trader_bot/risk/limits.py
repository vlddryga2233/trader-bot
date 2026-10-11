"""Hard risk limits. Config values are clamped to these; they cannot be raised at runtime."""

MAX_LEVERAGE = 5
MAX_RISK_PER_TRADE = 0.02
MAX_OPEN_POSITIONS = 3
MAX_DAILY_LOSS_LIMIT = 0.10
MAX_WEEKLY_LOSS_LIMIT = 0.20
MIN_ATR_STOP_MULT = 1.0
