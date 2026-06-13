"""
core/executor.py
================
Orchestrator หลัก: จัดการ lifecycle ของแต่ละ Iron Condor
Entry → Stop Loss setup → Take Profit → Monitoring → Close
"""

from __future__ import annotations
import uuid
import logging
from datetime import date
from typing import Optional

from core.broker import AlpacaBroker
from core.options_engine import (
    IronCondor, find_strike_by_delta, calculate_mid_price,
    build_iron_condor_structure, select_wing_width,
    is_premium_acceptable, should_take_profit
)
from core.risk_manager import DailyRiskTracker
from utils.logger import log_trade
from utils.market import is_flat_market
from config import (
    UNDERLYING_SYMBOL, TARGET_DELTA, DEFAULT_WING_WIDTH,
    TARGET_PREMIUM_PER_SIDE, MIN_PREMIUM_PER_SIDE
)

logger = logging.getLogger("0dte_bot")


class TradeExecutor:
    """จัดการ flow ทั้งหมดของการเทรด"""

    def __init__(self, broker: AlpacaBroker, risk_mgr: DailyRiskTracker):
        self.broker   = broker
        self.risk_mgr = risk_mgr
        self.active_trades: list[IronCondor] = []

    def get_today_expiry(self) -> str:
        """คืน expiry วันนี้ format YYYY-MM-DD"""
        return date.today().isoformat()

    def try_open_new_trade(self, bars: list[dict]) -> Optional[IronCondor]:
        """
        พยายามเปิด Iron Condor ชุดใหม่
        1. ตรวจ flat market signal
        2. ดึง option chain + เลือก strikes
        3. ตรวจ risk limit
        4. ส่งคำสั่ง (Call spread ก่อน, Put spread ทันที)
        5. ตั้ง Stop Loss OCO + Take Profit
        """

        # ── Step 1: Flat Market Signal ──────────────────────────────────────
        if not is_flat_market(bars):
            logger.info("📶 Market not flat – skipping entry")
            return None

        # ── Step 2: Get current price & option chain ─────────────────────────
        underlying_price = self.broker.get_underlying_price()
        expiry = self.get_today_expiry()
        chain  = self.broker.get_option_chain(expiry)

        if not chain:
            logger.warning("⚠️ Empty option chain – skipping")
            return None

        # ── Step 3: Select strikes by delta ──────────────────────────────────
        call_short_opt = find_strike_by_delta(chain, "C", TARGET_DELTA)
        put_short_opt  = find_strike_by_delta(chain, "P", TARGET_DELTA)

        if not call_short_opt or not put_short_opt:
            logger.warning("⚠️ Could not find suitable strikes")
            return None

        call_short_strike = float(call_short_opt.get("strike", underlying_price + 10))
        put_short_strike  = float(put_short_opt.get("strike",  underlying_price - 10))

        # ── Step 4: Calculate premiums & adjust wings ─────────────────────────
        call_premium_est = calculate_mid_price(call_short_opt)
        put_premium_est  = calculate_mid_price(put_short_opt)

        if not is_premium_acceptable(call_premium_est) or not is_premium_acceptable(put_premium_est):
            logger.info(
                f"💸 Premium out of range | Call: ${call_premium_est:.2f} | Put: ${put_premium_est:.2f} | "
                f"Target: ${MIN_PREMIUM_PER_SIDE}-${TARGET_PREMIUM_PER_SIDE}"
            )
            return None

        call_width, put_width = select_wing_width(call_premium_est, put_premium_est)
        call_long_strike = call_short_strike + call_width
        put_long_strike  = put_short_strike  - put_width

        # ── Step 5: Build IC structure ────────────────────────────────────────
        trade_id = f"IC_{uuid.uuid4().hex[:8].upper()}"
        ic = build_iron_condor_structure(
            trade_id=trade_id,
            expiry=expiry,
            underlying_price=underlying_price,
            call_short_strike=call_short_strike,
            call_long_strike=call_long_strike,
            put_short_strike=put_short_strike,
            put_long_strike=put_long_strike,
            underlying_symbol=UNDERLYING_SYMBOL,
        )
        ic.call_premium = call_premium_est
        ic.put_premium  = put_premium_est

        # ── Step 6: Risk check ────────────────────────────────────────────────
        account = self.broker.get_account()
        can_open, reason = self.risk_mgr.can_open_new_trade(ic, account["buying_power"])
        if not can_open:
            logger.warning(f"🚫 Risk check failed: {reason}")
            return None

        paused, pause_reason = self.risk_mgr.should_pause_trading()
        if paused:
            logger.warning(f"⏸️ Trading paused: {pause_reason}")
            return None

        # ── Step 7: Execute – Call spread first, then Put spread ──────────────
        logger.info(
            f"\n{'─'*50}\n"
            f"🚀 Opening IC | {trade_id}\n"
            f"   Call Spread: {call_short_strike}C / {call_long_strike}C | Premium: ${call_premium_est:.2f}\n"
            f"   Put  Spread: {put_short_strike}P / {put_long_strike}P  | Premium: ${put_premium_est:.2f}\n"
            f"   Total Premium: ${ic.total_premium:.2f} | Stop Loss: ${ic.stop_loss_value:.2f}\n"
            f"{'─'*50}"
        )

        call_short_id = self.broker.place_credit_spread(
            ic.call_short.symbol, ic.call_long.symbol
        )
        put_short_id = self.broker.place_credit_spread(
            ic.put_short.symbol, ic.put_long.symbol
        )

        if not call_short_id or not put_short_id:
            logger.error("❌ Failed to place spreads – aborting trade")
            return None

        ic.call_short.order_id = call_short_id
        ic.put_short.order_id  = put_short_id
        ic.status = "open"

        # ── Step 8: Set Stop Loss OCO on SHORT legs ───────────────────────────
        sl1, sl2 = self.broker.place_oco_stop_on_short(
            ic.call_short.symbol, ic.stop_loss_value
        )
        pl1, pl2 = self.broker.place_oco_stop_on_short(
            ic.put_short.symbol, ic.stop_loss_value
        )
        ic.call_stop_order_id = sl1
        ic.put_stop_order_id  = pl1

        # ── Step 9: Set Take Profit at $0.05 ─────────────────────────────────
        self.broker.place_take_profit_on_short(ic.call_short.symbol)
        self.broker.place_take_profit_on_short(ic.put_short.symbol)

        # ── Step 10: Register with risk manager & log ─────────────────────────
        self.risk_mgr.register_trade_open(ic)
        self.active_trades.append(ic)

        log_trade({
            "timestamp":         __import__("datetime").datetime.now().isoformat(),
            "trade_id":          ic.trade_id,
            "action":            "OPEN",
            "call_short_strike": ic.call_short.strike,
            "call_long_strike":  ic.call_long.strike,
            "put_short_strike":  ic.put_short.strike,
            "put_long_strike":   ic.put_long.strike,
            "call_premium":      ic.call_premium,
            "put_premium":       ic.put_premium,
            "total_premium":     ic.total_premium,
            "stop_loss_value":   ic.stop_loss_value,
            "expiry":            ic.expiry,
        })

        return ic

    def monitor_open_trades(self):
        """
        ตรวจ open trades ทุก loop:
        - ถ้า short ราคา <= $0.05 → Take Profit (ปิด short, เก็บ long ไว้ reuse)
        - ถ้า stop ถูกทริกเกอร์ → ปิด long ทิ้ง, update risk
        """
        for ic in [t for t in self.active_trades if t.status == "open"]:
            call_price = self.broker.get_option_quote(ic.call_short.symbol)
            put_price  = self.broker.get_option_quote(ic.put_short.symbol)

            if call_price is not None and should_take_profit(call_price):
                logger.info(f"🎯 Take Profit triggered on Call Short | {ic.trade_id} @ ${call_price:.2f}")
                self._close_short_take_profit(ic, "call")

            if put_price is not None and should_take_profit(put_price):
                logger.info(f"🎯 Take Profit triggered on Put Short | {ic.trade_id} @ ${put_price:.2f}")
                self._close_short_take_profit(ic, "put")

    def _close_short_take_profit(self, ic: IronCondor, side: str):
        """ปิดขา Short ที่ถึง TP แล้ว – เก็บ Long ไว้ reuse"""
        if side == "call":
            # ยกเลิก stop orders ของฝั่ง call
            if ic.call_stop_order_id:
                self.broker.cancel_order(ic.call_stop_order_id)
            logger.info(f"✅ Call Short closed at TP | Long {ic.call_long.symbol} retained for reuse")
        else:
            if ic.put_stop_order_id:
                self.broker.cancel_order(ic.put_stop_order_id)
            logger.info(f"✅ Put Short closed at TP | Long {ic.put_long.symbol} retained for reuse")

    def tighten_stop_losses(self, ic: IronCondor, new_stop_value: float):
        """
        ขยับ Stop Loss ให้แน่นขึ้น (Trailing Stop)
        ใช้เมื่อกำไรสะสมแล้ว หรือก่อนเปิดสัญญาชุดใหม่เพื่อลด exposure
        """
        logger.info(f"🔧 Tightening stops on {ic.trade_id} → ${new_stop_value:.2f}")
        # ยกเลิก stop เก่า แล้วตั้งใหม่
        if ic.call_stop_order_id:
            self.broker.cancel_order(ic.call_stop_order_id)
        if ic.put_stop_order_id:
            self.broker.cancel_order(ic.put_stop_order_id)

        sl1, sl2 = self.broker.place_oco_stop_on_short(ic.call_short.symbol, new_stop_value)
        pl1, pl2 = self.broker.place_oco_stop_on_short(ic.put_short.symbol, new_stop_value)
        ic.call_stop_order_id = sl1
        ic.put_stop_order_id  = pl1

    def close_all_positions_eod(self):
        """ปิดทุก position ก่อนตลาดปิด (End of Day cleanup)"""
        logger.info("⏰ EOD: Closing all remaining open positions...")
        for ic in [t for t in self.active_trades if t.status == "open"]:
            self.broker.close_position(ic.call_short.symbol)
            self.broker.close_position(ic.call_long.symbol)
            self.broker.close_position(ic.put_short.symbol)
            self.broker.close_position(ic.put_long.symbol)
            ic.status = "closed"
            logger.info(f"🔒 EOD Closed: {ic.trade_id}")
