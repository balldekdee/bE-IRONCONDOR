"""
main.py
=======
0DTE Break-Even Iron Condor Bot — Main Runner (with Regime Intelligence)
=======================================================================

วิธีใช้:
  export ALPACA_API_KEY="your_key"
  export ALPACA_API_SECRET="your_secret"
  export SUPABASE_URL="https://xxx.supabase.co"      # optional
  export SUPABASE_KEY="your_service_key"              # optional
  python main.py

Bot จะ:
  - ป้อน price bar เข้า RegimeEnsemble ทุก loop → คำนวณ regime posterior
  - เปิด IC เฉพาะเมื่อ regime filter อนุญาต (เทรดเฉพาะโอกาส)
  - เรียนรู้ regime-conditional edge เอง (self-improving)
  - บันทึกทุกอย่างลง Supabase
"""

import time
import logging

from core.broker import AlpacaBroker
from core.executor import TradeExecutor
from core.risk_manager import DailyRiskTracker
from core.self_improve import SelfImprovingFilter
from core.database import Database
from core.regime import RegimeEnsemble
from utils.logger import setup_logger
from utils.market import (
    is_market_open, is_past_entry_delay,
    can_open_new_trade, now_est
)
from config import (
    TRADE_INTERVAL_MINUTES, UNDERLYING_SYMBOL,
    REGIME_ENABLED, ENCODER_SEQ_LEN, ENCODER_DEVICE,
    MIN_REGIME_CONFIDENCE, MAX_CHANGE_POINT_PROB,
    MIN_EXPECTED_EDGE, FILTER_EXPLORATION, REGIME_SNAPSHOT_EVERY,
    SUPABASE_URL, SUPABASE_KEY,
)

logger = setup_logger("0dte_bot")

POLL_INTERVAL_SECONDS = 60


def run():
    logger.info("=" * 60)
    logger.info("  🤖 0DTE Break-Even Iron Condor Bot + Regime AI")
    logger.info(f"  Underlying: {UNDERLYING_SYMBOL} | Paper Trading Mode")
    logger.info("=" * 60)

    # ── Initialize ────────────────────────────────────────────────────────────
    broker = AlpacaBroker()
    account = broker.get_account()
    logger.info(f"💼 Portfolio: ${account['portfolio_value']:,.2f} | "
                f"BP: ${account['buying_power']:,.2f}")

    risk_mgr = DailyRiskTracker(portfolio_value=account["portfolio_value"])

    # Database (optional)
    db = Database(SUPABASE_URL, SUPABASE_KEY)

    # Regime engine + self-improving filter
    regime_engine = None
    regime_filter = None
    if REGIME_ENABLED:
        regime_engine = RegimeEnsemble(encoder_seq_len=ENCODER_SEQ_LEN, device=ENCODER_DEVICE)
        regime_filter = SelfImprovingFilter(
            min_regime_confidence=MIN_REGIME_CONFIDENCE,
            max_change_point_prob=MAX_CHANGE_POINT_PROB,
            min_expected_edge=MIN_EXPECTED_EDGE,
            exploration=FILTER_EXPLORATION,
        )
        # resume self-improving state + model state จาก Supabase
        saved_post = db.load_regime_posteriors()
        if saved_post:
            regime_filter.load_state_dict(saved_post)
            logger.info("🧠 Loaded self-improving posteriors from DB")
        saved_model = db.load_model_state()
        if saved_model:
            try:
                regime_engine.load_state_dict(saved_model)
                logger.info("🧭 Resumed regime model state from DB")
            except Exception as e:
                logger.warning(f"⚠️ Could not load model state: {e}")

        # warmup regime engine ด้วยข้อมูลย้อนหลังวันนี้
        logger.info("🔥 Warming up regime engine with historical bars...")
        warmup_bars = broker.get_bars_5min(UNDERLYING_SYMBOL, limit=60)
        for b in warmup_bars:
            regime_engine.update(b)
        if regime_engine.last_state:
            logger.info("\n" + regime_engine.last_state.pretty())

    executor = TradeExecutor(
        broker=broker, risk_mgr=risk_mgr,
        regime_filter=regime_filter, db=db,
    )

    last_trade_hour = -1
    bar_counter = 0

    logger.info("🟢 Bot started. Waiting for market open...")

    while True:
        try:
            now = now_est()

            if not is_market_open():
                logger.info(f"💤 Market closed ({now.strftime('%H:%M')} EST). Sleeping 5 min...")
                time.sleep(300)
                continue

            # ── EOD cleanup ───────────────────────────────────────────────────
            if now.hour == 15 and now.minute >= 45:
                executor.close_all_positions_eod()
                _persist_state(db, regime_engine, regime_filter)
                _print_eod_summary(risk_mgr, regime_filter)
                logger.info("🔴 EOD complete. Bot stopping until next market day.")
                break

            # ── ป้อน bar ล่าสุดเข้า regime engine ─────────────────────────────
            regime_state = None
            if regime_engine is not None:
                latest = broker.get_bars_5min(UNDERLYING_SYMBOL, limit=1)
                if latest:
                    regime_state = regime_engine.update(latest[-1])
                    bar_counter += 1
                    # snapshot ลง DB เป็นระยะ
                    if regime_state and bar_counter % REGIME_SNAPSHOT_EVERY == 0:
                        px = broker.get_underlying_price()
                        db.insert_regime_snapshot(regime_state, px)

            # ── Monitor open positions ────────────────────────────────────────
            executor.monitor_open_trades()

            # ── ตรวจ entry (ชั่วโมงละ 1 ครั้ง) ───────────────────────────────
            current_hour = now.hour
            should_try_entry = (
                is_past_entry_delay()
                and can_open_new_trade()
                and current_hour != last_trade_hour
            )

            if should_try_entry:
                logger.info(f"\n⏰ {now.strftime('%H:%M')} EST – Checking entry...")
                if regime_state is not None:
                    logger.info("\n" + regime_state.pretty())

                bars = broker.get_bars_5min(UNDERLYING_SYMBOL, limit=5)
                ic = executor.try_open_new_trade(bars, regime_state=regime_state)
                if ic:
                    logger.info(f"✅ Trade opened: {ic.trade_id}")
                last_trade_hour = current_hour

            # ── Periodic risk + regime update ─────────────────────────────────
            if now.minute in (0, 30):
                summary = risk_mgr.summary()
                logger.info(
                    f"📊 Risk | Trades: {summary['trades_opened']} | "
                    f"Open Risk: ${summary['current_open_risk']:.0f} | "
                    f"Day PnL: ${summary['realized_pnl']:.0f} | "
                    f"Used: {summary['risk_utilization_pct']}%"
                )
                _persist_state(db, regime_engine, regime_filter)

            time.sleep(POLL_INTERVAL_SECONDS)

        except KeyboardInterrupt:
            logger.info("\n⚠️  Stopped by user. Closing positions + saving state...")
            executor.close_all_positions_eod()
            _persist_state(db, regime_engine, regime_filter)
            break
        except Exception as e:
            logger.error(f"❌ Unexpected error: {e}", exc_info=True)
            time.sleep(60)


def _persist_state(db, regime_engine, regime_filter):
    """บันทึก self-improving + model state ลง Supabase"""
    if regime_filter is not None:
        db.save_regime_posteriors(regime_filter.state_dict())
    if regime_engine is not None:
        db.save_model_state(regime_engine.state_dict())


def _print_eod_summary(risk_mgr, regime_filter):
    summary = risk_mgr.summary()
    logger.info(f"\n{'='*50}")
    logger.info("📊 END OF DAY SUMMARY")
    logger.info(f"   Trades: {summary['trades_opened']} | Stops: {summary['stops_hit']}")
    logger.info(f"   Day PnL: ${summary['realized_pnl']:,.2f}")
    logger.info(f"   Risk Used: {summary['risk_utilization_pct']}%")
    if regime_filter is not None:
        logger.info("\n🧠 REGIME-CONDITIONAL EDGE (learned):")
        for name, stats in regime_filter.summary().items():
            logger.info(f"   {name:<22} n={stats['n']:<3} "
                        f"E[PnL]=${stats['expected_pnl']:<7} WR={stats['win_rate']}%")
    logger.info(f"{'='*50}\n")


if __name__ == "__main__":
    run()
