# trader-bot

ML-driven Binance USD-M futures bot (BTCUSDT, ETHUSDT, 4h decisions). Scope: data → ML → backtest → **testnet**. Live trading is intentionally disabled in the CLI.

Full documentation: [docs/](docs/README.md).

## Quick start
```bash
cp .env.example .env            # add testnet keys; never commit .env
docker compose up -d timescaledb
uv sync
uv run trader-bot init-db
uv run trader-bot backfill      # klines, funding, OI/long-short metrics (data.binance.vision)
uv run trader-bot train [--tune]
uv run trader-bot backtest      # walk-forward OOS; gate: Sharpe > 1.5 and |MaxDD| < 15%
# research; barrier holdout is a one-time final evaluation
uv run trader-bot baselines --holdout-start 2025-10-01
uv run trader-bot barrier-experiment --holdout-start 2025-10-01 # research
uv run trader-bot barrier-backtest --holdout-start 2025-10-01   # research, run once
uv run trader-bot collect       # live collectors (run alongside; accumulates liquidations)
uv run trader-bot run --testnet
uv run trader-bot killswitch-reset   # manual unlock after a kill-switch event
uv run pytest
```

## Layout
`data/` sources + ingest · `features/` leakage-safe builder · `labeling/` · `models/` CatBoost+LightGBM ensemble, purged CV, Optuna, registry · `backtest/` · `risk/` hard limits + kill switch · `execution/` Binance SDK adapter, order manager, user stream · `scheduler/` 4h loop + weekly retrain. Settings: `config/strategy.yaml`.

## Rules worth knowing
- **Risk is final.** `risk/limits.py` caps risk/trade (2%), leverage (5x), open positions, loss limits; config can only go lower.
- Every entry gets a close-position stop and take-profit (Algo Order API). If protective orders fail, the position is closed immediately. Startup reconciliation flattens unprotected positions.
- **Kill switch:** daily/weekly loss → lock file in `state/` first, then cancel + flatten all, then halt. The bot refuses to start while locked. The code cannot revoke API keys: rotate/disable them on Binance manually.
- **Binance key hygiene:** withdrawals disabled, IP whitelist, testnet keys only for now.
- **Data limits:** Binance OI/long-short REST history is ~30 days, so history comes from the public archive. `backfill-liquidations` loads 4-hour total liquidation quantity and average-price aggregates from API BRICKS (`API_BRIKS_API_KEY`). Historical rows have no long/short side; Binance live events retain it. Estimated USD totals are approximate. The feature group is currently off: its walk-forward comparison did not pass the trading gate. On-chain adapters are stubs (`data/sources/adapters.py`). See [data and ML docs](docs/data-and-ml.md#api-bricks-historical-liquidation-data).
- Backtest splits capital equally across symbols (independent books). Stop is checked before target within a bar.


Это бот для торговли бессрочными фьючерсами Binance по BTCUSDT и ETHUSDT. Он принимает решения на закрытии каждой 4‑часовой свечи. Сейчас CLI разрешает запуск только на **testnet**: live‑торговля заблокирована.

### Как он принимает решение

1. **Получает данные.** Исторические свечи, funding и метрики деривативов загружаются в TimescaleDB. Команда `collect` собирает данные в реальном времени, а основной цикл перед решением дополнительно обновляет последние данные.
2. **Строит признаки.** Использует цену, волатильность, объём, funding, открытый интерес и соотношения long/short. Расчёты устроены так, чтобы не использовать данные, которые ещё не были доступны на момент решения.
3. **Получает прогноз модели.** Ансамбль CatBoost и LightGBM оценивает вероятности падения, бокового движения и роста, а также предполагаемую доходность.
4. **Фильтрует сигнал.** Сделка рассматривается, если наиболее вероятное направление — рост или падение, уверенность не ниже `0.75`, а прогноз доходности модели согласуется с направлением.
5. **Проверяет риск и рассчитывает позицию.** Риск на сделку по умолчанию — 1% капитала, плечо — до 3x; код дополнительно ограничивает риск 2%, плечо 5x и число открытых позиций. Стоп рассчитывается от ATR, тейк-профит — от прогнозируемого движения. Если отношение потенциальной прибыли к риску ниже заданного порога, сделка отклоняется.
6. **Размещает ордера.** На testnet бот открывает позицию рыночным ордером и ставит защитные стоп и тейк-профит. Если защитные ордера не удалось выставить, он пытается закрыть позицию.

Основной цикл и фильтры сигналов описаны в src/trader\_bot/scheduler/runner.py и src/trader\_bot/risk/engine.py, а параметры — в config/strategy.yaml.

### Обучение и проверка

Модель обучается на исторических данных. Для оценки используется пошаговая проверка на будущих отрезках данных: при обучении очередной модели будущие свечи исключаются. `backtest` затем симулирует сделки с комиссиями, проскальзыванием и funding. Это оценка на истории, а не гарантия будущего результата.

### Защита и ограничения

При достижении дневного или недельного лимита убытка бот записывает блокировку, пытается отменить ордера и закрыть позиции, после чего останавливается. При перезапуске он проверяет открытые позиции и защитные ордера. Решения и причины принятия или отклонения сигналов сохраняются в таблице `signals`.

Сейчас признаки ликвидаций и ончейн-данные выключены. Историю агрегатов ликвидаций за 4 часа можно загрузить через API BRICKS командой `backfill-liquidations`; нужен `API_BRIKS_API_KEY`. В агрегатах нет стороны LONG/SHORT, но живые события Binance продолжают её содержать. Ончейн-адаптеры не реализованы. Сценарий запуска и команды описаны в README.md.
