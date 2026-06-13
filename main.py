"""
main.py
=======
0DTE Break-Even Iron Condor Bot - Main Runner
=============================================

วิธีใช้:
  1. ตั้ง environment variables:
       export ALPACA_API_KEY="your_key"
       export ALPACA_API_SECRET="your_secret"
  2. รัน: python main.py

Bot จะ:
  - ตรวจสอบทุก 60 วินาที
  - เปิด Iron Condor ชุดใหม่ทุกชั่วโมง (ถ้าผ่านทุกเงื่อนไข)
  - Monitor open positions ต่อเนื่อง
  - หยุดอัตโนมัติเมื่อตลาดปิด
"""

import time
import logging
from datetime import datetime

from core.broker import AlpacaBroker
from core.executor import TradeExecutor
from core.risk_manager import DailyRiskTracker
from utils.logger import setup_logger
from utils.market import (
    is_market_open, is_past_entry_delay,
    can_open_new_trade, now_est
)
from config import TRADE_INTERVAL_MINUTES, UNDERLYING_SYMBOL

logger = setup_logger("0dte_bot")

POLL_INTERVAL_SECONDS = 60   # ตรวจทุก 60 วินาที


def run():
    logger.info("=" * 60)
    logger.info("  🤖 0DTE Break-Even Iron Condor Bot")
    logger.info(f"  Underlying: {UNDERLYING_SYMBOL} | Paper Trading Mode")
    logger.info("=" * 60)

    # ── Initialize Components ─────────────────────────────────────────────────
    broker = AlpacaBroker()

    account = broker.get_account()
    logger.info(f"💼 Portfolio Value: ${account['portfolio_value']:,.2f}")
    logger.info(f"💰 Buying Power:    ${account['buying_power']:,.2f}")

    risk_mgr = DailyRiskTracker(portfolio_value=account["portfolio_value"])
    executor = TradeExecutor(broker=broker, risk_mgr=risk_mgr)

    last_trade_hour = -1   # ติดตามว่าชั่วโมงไหนเปิดไปแล้ว

    # ── Main Loop ─────────────────────────────────────────────────────────────
    logger.info("🟢 Bot started. Waiting for market open...")

    while True:
        try:
            now = now_est()

            # ── ตลาดปิดอยู่ → รอ ──────────────────────────────────────────────
            if not is_market_open():
                logger.info(f"💤 Market closed ({now.strftime('%H:%M')} EST). Sleeping 5 min...")
                time.sleep(300)
                continue

            # ── EOD Cleanup 15:45 EST ─────────────────────────────────────────
            if now.hour == 15 and now.minute >= 45:
                executor.close_all_positions_eod()
                summary = risk_mgr.summary()
                logger.info(f"\n{'='*50}")
                logger.info(f"📊 END OF DAY SUMMARY")
                logger.info(f"   Trades: {summary['trades_opened']} | Stops Hit: {summary['stops_hit']}")
                logger.info(f"   Day PnL: ${summary['realized_pnl']:,.2f}")
                logger.info(f"   Risk Used: {summary['risk_utilization_pct']}% of daily limit")
                logger.info(f"{'='*50}\n")
                logger.info("🔴 EOD complete. Bot stopping until next market day.")
                break

            # ── Monitor open positions (ทุก loop) ─────────────────────────────
            executor.monitor_open_trades()

            # ── ตรวจว่าถึงเวลาเปิดชุดใหม่ไหม (ชั่วโมงละ 1 ครั้ง) ─────────────
            current_hour = now.hour

            should_try_entry = (
                is_past_entry_delay()
                and can_open_new_trade()
                and current_hour != last_trade_hour
            )

            if should_try_entry:
                logger.info(f"\n⏰ {now.strftime('%H:%M')} EST – Checking entry conditions...")

                # ดึงแท่งเทียน 5 นาที สำหรับ flat market signal
                bars = broker.get_bars_5min(UNDERLYING_SYMBOL, limit=5)

                ic = executor.try_open_new_trade(bars)
                if ic:
                    last_trade_hour = current_hour
                    logger.info(f"✅ Trade opened: {ic.trade_id}")
                else:
                    logger.info("⏭️  Conditions not met – will retry next hour")
                    last_trade_hour = current_hour  # ป้องกัน spam retry ในชั่วโมงเดิม

            # ── Risk Summary ทุก 30 นาที ─────────────────────────────────────
            if now.minute == 0 or now.minute == 30:
                summary = risk_mgr.summary()
                logger.info(
                    f"📊 Risk Update | "
                    f"Trades: {summary['trades_opened']} | "
                    f"Open Risk: ${summary['current_open_risk']:.0f} | "
                    f"Day PnL: ${summary['realized_pnl']:.0f} | "
                    f"Risk Used: {summary['risk_utilization_pct']}%"
                )

            time.sleep(POLL_INTERVAL_SECONDS)

        except KeyboardInterrupt:
            logger.info("\n⚠️  Bot stopped by user (Ctrl+C)")
            logger.info("🔒 Closing all open positions for safety...")
            executor.close_all_positions_eod()
            break
        except Exception as e:
            logger.error(f"❌ Unexpected error: {e}", exc_info=True)
            logger.info(f"⏳ Retrying in 60 seconds...")
            time.sleep(60)


if __name__ == "__main__":
    run()
