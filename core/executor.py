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
    IronCondor, find_strike_by_delta, calculate_mid_price, calculate_mid_per_share,
    build_iron_condor_structure, select_wing_width,
    is_premium_acceptable, should_take_profit
)
from core.risk_manager import DailyRiskTracker
from core.self_improve import SelfImprovingFilter
from core.database import Database
from utils.logger import log_trade
from utils.market import is_flat_market
from config import (
    UNDERLYING_SYMBOL, TARGET_DELTA, DEFAULT_WING_WIDTH,
    TARGET_PREMIUM_PER_SIDE, MIN_PREMIUM_PER_SIDE
)

logger = logging.getLogger("0dte_bot")


class TradeExecutor:
    """จัดการ flow ทั้งหมดของการเทรด"""

    def __init__(self, broker: AlpacaBroker, risk_mgr: DailyRiskTracker,
                 regime_filter: SelfImprovingFilter = None,
                 db: Database = None):
        self.broker   = broker
        self.risk_mgr = risk_mgr
        self.regime_filter = regime_filter
        self.db = db
        self.active_trades: list[IronCondor] = []
        # เก็บ regime ตอนเข้า ต่อ trade_id (สำหรับ self-improvement)
        self._regime_at_entry: dict[str, int] = {}

    def get_today_expiry(self) -> str:
        """#10: คืน expiry วันนี้ตาม US/Eastern (ไม่ใช่ system tz) — สำคัญมากสำหรับ 0DTE"""
        from utils.market import now_est
        return now_est().date().isoformat()

    def try_open_new_trade(self, bars: list[dict], regime_state=None) -> Optional[IronCondor]:
        """
        พยายามเปิด Iron Condor ชุดใหม่
        0. ตรวจ regime filter (เทรดเฉพาะ regime ที่มี edge)
        1. ตรวจ flat market signal
        2. ดึง option chain + เลือก strikes
        3. ตรวจ risk limit
        4. ส่งคำสั่ง (Call spread ก่อน, Put spread ทันที)
        5. ตั้ง Stop Loss OCO + Take Profit
        """

        # ── Step 0: Regime Filter (self-improving) ──────────────────────────
        regime_at_entry = None
        expected_edge = None
        if self.regime_filter is not None and regime_state is not None:
            allow, reason, diag = self.regime_filter.should_trade(regime_state)
            logger.info(f"🧭 Regime Filter | {reason}")
            if not allow:
                return None
            regime_at_entry = regime_state.map_regime
            expected_edge = diag.get("expected_edge")
        elif regime_state is None and self.regime_filter is not None:
            logger.info("🧭 Regime not ready (warmup) — skipping entry")
            return None

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
        # premium per-contract (ดอลลาร์) สำหรับ stop loss / risk
        call_premium_est = calculate_mid_price(call_short_opt)
        put_premium_est  = calculate_mid_price(put_short_opt)
        # per-share net credit (ใช้ตั้ง limit price ของ MLEG order)
        call_credit_per_share = calculate_mid_per_share(call_short_opt) - calculate_mid_per_share(
            next((o for o in chain if o["type"] == "C" and o["strike"] == call_short_strike + DEFAULT_WING_WIDTH), call_short_opt))
        put_credit_per_share = calculate_mid_per_share(put_short_opt) - calculate_mid_per_share(
            next((o for o in chain if o["type"] == "P" and o["strike"] == put_short_strike - DEFAULT_WING_WIDTH), put_short_opt))
        call_credit_per_share = max(call_credit_per_share, 0.05)
        put_credit_per_share  = max(put_credit_per_share, 0.05)

        if not is_premium_acceptable(call_premium_est) or not is_premium_acceptable(put_premium_est):
            logger.info(
                f"💸 Premium out of range | Call: ${call_premium_est:.0f} | Put: ${put_premium_est:.0f} | "
                f"Target: ${MIN_PREMIUM_PER_SIDE}-${TARGET_PREMIUM_PER_SIDE} per contract"
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

        # ── Step 7: Execute spreads as atomic MLEG orders (#5/#6) ─────────────
        logger.info(
            f"\n{'─'*50}\n"
            f"🚀 Opening IC | {trade_id}\n"
            f"   Call Spread: {call_short_strike}C / {call_long_strike}C | Premium: ${call_premium_est:.2f}\n"
            f"   Put  Spread: {put_short_strike}P / {put_long_strike}P  | Premium: ${put_premium_est:.2f}\n"
            f"   Total Premium: ${ic.total_premium:.2f} | Stop Loss: ${ic.stop_loss_value:.2f}\n"
            f"{'─'*50}"
        )

        # Call spread (atomic) — net credit per-share สำหรับ limit price
        call_order_id = self.broker.place_credit_spread(
            ic.call_short.symbol, ic.call_long.symbol, net_credit=call_credit_per_share,
        )
        if not call_order_id:
            logger.error("❌ Call spread failed — aborting trade (no legs opened)")
            return None

        # Put spread (atomic) — net credit per-share
        put_order_id = self.broker.place_credit_spread(
            ic.put_short.symbol, ic.put_long.symbol, net_credit=put_credit_per_share,
        )
        if not put_order_id:
            # #6 rollback: call spread เปิดไปแล้ว → ต้องปิดทันที ไม่ให้เหลือ exposure ข้างเดียว
            logger.error("❌ Put spread failed — rolling back call spread to avoid one-sided exposure")
            self.broker.close_option_leg(ic.call_short.symbol, is_short=True)
            self.broker.close_option_leg(ic.call_long.symbol,  is_short=False)
            return None

        ic.call_short.order_id = call_order_id
        ic.put_short.order_id  = put_order_id
        ic.status = "open"

        # ── Step 8: Stop loss — stop-limit + stop-market backstop (#9) ────────
        ic.call_stop_order_id     = self.broker.place_oco_stop_on_short(ic.call_short.symbol, ic.stop_loss_value)
        ic.call_backstop_order_id = self.broker.place_stop_market_backstop(ic.call_short.symbol, ic.stop_market_trigger)
        ic.put_stop_order_id      = self.broker.place_oco_stop_on_short(ic.put_short.symbol, ic.stop_loss_value)
        ic.put_backstop_order_id  = self.broker.place_stop_market_backstop(ic.put_short.symbol, ic.stop_market_trigger)

        # ── Step 9: Take Profit at $0.05 (เก็บ id ไว้ verify fill — #8) ───────
        ic.call_tp_order_id = self.broker.place_take_profit_on_short(ic.call_short.symbol)
        ic.put_tp_order_id  = self.broker.place_take_profit_on_short(ic.put_short.symbol)

        # ── Step 10: Register with risk manager & log ─────────────────────────
        self.risk_mgr.register_trade_open(ic)
        self.active_trades.append(ic)

        # เก็บ regime ตอนเข้า (สำหรับ self-improvement ตอนปิด)
        if regime_at_entry is not None:
            self._regime_at_entry[ic.trade_id] = regime_at_entry

        trade_record = {
            "timestamp":         __import__("datetime").datetime.now().isoformat(),
            "trade_id":          ic.trade_id,
            "action":            "OPEN",
            "underlying_price":  ic.underlying_price,
            "call_short_strike": ic.call_short.strike,
            "call_long_strike":  ic.call_long.strike,
            "put_short_strike":  ic.put_short.strike,
            "put_long_strike":   ic.put_long.strike,
            "call_premium":      ic.call_premium,
            "put_premium":       ic.put_premium,
            "total_premium":     ic.total_premium,
            "stop_loss_value":   ic.stop_loss_value,
            "expiry":            ic.expiry,
        }
        log_trade(trade_record)

        # ── Supabase persist ──────────────────────────────────────────────────
        if self.db is not None:
            db_record = dict(trade_record)
            if regime_state is not None:
                db_record["regime_at_entry"]   = regime_at_entry
                db_record["regime_probs"]      = regime_state.regime_probs.tolist()
                db_record["regime_confidence"] = regime_state.confidence
                db_record["expected_edge"]     = expected_edge
            self.db.insert_trade(db_record)

        return ic

    def monitor_open_trades(self):
        """
        ตรวจ open trades ทุก loop:
        - #8: ตรวจ TP order ว่า fill จริงไหม (ไม่ใช่แค่ดู quote แล้ว log closed)
        - #9: ถ้า stop-limit fill → cancel stop-market backstop (sibling), และกลับกัน
        - ปิด long ที่เหลือเมื่อ short ถูก stop
        """
        for ic in [t for t in self.active_trades if t.status == "open"]:
            self._reconcile_side(ic, "call")
            self._reconcile_side(ic, "put")

            # ถ้าทั้งสอง short ปิดหมดแล้ว → trade จบ
            if ic.call_short_closed and ic.put_short_closed:
                self._finalize_trade(ic)

    def _reconcile_side(self, ic: IronCondor, side: str):
        """
        Sync state ฝั่งหนึ่งกับ order status จริงจาก Alpaca (#8, #9)
        ลำดับการตรวจ:
          1. TP order filled?     → short ปิดด้วยกำไร, cancel stops, เก็บ long reuse
          2. stop-limit filled?   → short ปิดด้วย loss, cancel backstop + TP, ปิด long
          3. backstop filled?     → เหมือน stop, cancel stop-limit + TP, ปิด long
          4. (DRY-RUN) fallback   → ใช้ quote-based ประเมิน
        """
        if side == "call":
            short_sym, long_sym = ic.call_short.symbol, ic.call_long.symbol
            tp_id, stop_id, backstop_id = ic.call_tp_order_id, ic.call_stop_order_id, ic.call_backstop_order_id
            already_closed = ic.call_short_closed
        else:
            short_sym, long_sym = ic.put_short.symbol, ic.put_long.symbol
            tp_id, stop_id, backstop_id = ic.put_tp_order_id, ic.put_stop_order_id, ic.put_backstop_order_id
            already_closed = ic.put_short_closed

        if already_closed:
            return

        from config import DRY_RUN

        # ── LIVE: ตรวจ fill จาก order status จริง ──
        if not DRY_RUN:
            if tp_id and self.broker.is_order_filled(tp_id):
                # #8: TP fill จริง → mark closed + cancel stops (sibling)
                for oid in (stop_id, backstop_id):
                    if oid:
                        self.broker.cancel_order(oid)
                logger.info(f"✅ [{side}] Short closed at TP (fill confirmed) | "
                            f"Long {long_sym} retained for reuse")
                self._mark_short_closed(ic, side)
                return

            stop_filled = stop_id and self.broker.is_order_filled(stop_id)
            backstop_filled = backstop_id and self.broker.is_order_filled(backstop_id)
            if stop_filled or backstop_filled:
                # #9: stop ฝั่งใดฝั่งหนึ่ง fill → cancel sibling ที่เหลือ + TP
                survivors = []
                if stop_filled:
                    survivors = [backstop_id, tp_id]
                else:
                    survivors = [stop_id, tp_id]
                for oid in survivors:
                    if oid:
                        self.broker.cancel_order(oid)
                # ปิด long ที่เหลือ (#7: long → sell to close)
                self.broker.close_option_leg(long_sym, is_short=False)
                logger.warning(f"🛑 [{side}] Short stopped out (fill confirmed) | closing long {long_sym}")
                self._mark_short_closed(ic, side, stopped=True)
                return
            return

        # ── DRY-RUN: ไม่มี order จริง → ประเมินจาก quote ──
        price = self.broker.get_option_quote(short_sym)   # per-share
        if price is None:
            return
        if should_take_profit(price):
            logger.info(f"🧪 [DRY-RUN][{side}] TP hit @ ${price:.2f} | short closed, long reuse")
            self._mark_short_closed(ic, side)
        elif price * 100 >= ic.stop_loss_value:   # per-share→per-contract เทียบ stop
            logger.warning(f"🧪 [DRY-RUN][{side}] Stop hit @ ${price*100:.0f}/contract | closing long")
            self.broker.close_option_leg(long_sym, is_short=False)
            self._mark_short_closed(ic, side, stopped=True)

    def _mark_short_closed(self, ic: IronCondor, side: str, stopped: bool = False):
        if side == "call":
            ic.call_short_closed = True
        else:
            ic.put_short_closed = True
        if stopped:
            ic._any_stopped = True  # type: ignore[attr-defined]

    def _finalize_trade(self, ic: IronCondor):
        """รวมผลทั้งสองฝั่ง → record outcome จริง (เรียกครั้งเดียวต่อ trade)"""
        # ประเมิน PnL คร่าวๆ จาก quote ปัจจุบัน (LIVE ใช้ fill price จริงผ่าน account ได้)
        stopped = getattr(ic, "_any_stopped", False)
        # PnL = premium ที่เก็บได้ - ต้นทุนปิด (approximation; live ใช้ realized จาก account)
        call_cost = self.broker.get_option_quote(ic.call_short.symbol) or 0.0
        put_cost  = self.broker.get_option_quote(ic.put_short.symbol) or 0.0
        gross_credit = ic.total_premium
        close_cost = (call_cost + put_cost) * 100
        pnl = gross_credit - close_cost
        outcome = "stopped" if stopped else ("win" if pnl > 0 else "loss")
        self.record_trade_outcome(ic, pnl, outcome)

    def tighten_stop_losses(self, ic: IronCondor, new_stop_value: float):
        """ขยับ Stop Loss ให้แน่นขึ้น (Trailing Stop)"""
        logger.info(f"🔧 Tightening stops on {ic.trade_id} → ${new_stop_value:.2f}")
        for oid in (ic.call_stop_order_id, ic.put_stop_order_id):
            if oid:
                self.broker.cancel_order(oid)
        ic.call_stop_order_id = self.broker.place_oco_stop_on_short(ic.call_short.symbol, new_stop_value)
        ic.put_stop_order_id  = self.broker.place_oco_stop_on_short(ic.put_short.symbol, new_stop_value)

    def record_trade_outcome(self, ic: IronCondor, pnl: float, outcome: str):
        """
        เรียกเมื่อ trade ปิด (TP / stop / EOD)
        → feed self-improving filter + บันทึก Supabase + update risk
        """
        ic.status = "closed"
        ic.pnl = pnl

        # self-improvement: อัปเดต regime-conditional edge
        regime = self._regime_at_entry.get(ic.trade_id)
        if regime is not None and self.regime_filter is not None:
            self.regime_filter.record_outcome(regime, pnl)

        # risk manager
        if outcome == "stopped":
            self.risk_mgr.register_stop_hit(ic.trade_id, abs(pnl))
        else:
            self.risk_mgr.register_trade_close(ic.trade_id, pnl)

        # Supabase
        if self.db is not None:
            self.db.close_trade(ic.trade_id, pnl, outcome)

        log_trade({
            "timestamp": __import__("datetime").datetime.now().isoformat(),
            "trade_id":  ic.trade_id,
            "action":    "CLOSE",
            "pnl":       pnl,
            "outcome":   outcome,
        })

    def close_all_positions_eod(self):
        """ปิดทุก position ก่อนตลาดปิด — #7: short→buy_to_close, long→sell_to_close"""
        logger.info("⏰ EOD: Closing all remaining open positions...")
        for ic in [t for t in self.active_trades if t.status == "open"]:
            # cancel resting orders ก่อน
            for oid in (ic.call_stop_order_id, ic.call_backstop_order_id, ic.call_tp_order_id,
                        ic.put_stop_order_id, ic.put_backstop_order_id, ic.put_tp_order_id):
                if oid:
                    self.broker.cancel_order(oid)
            # ปิด legs ให้ถูกทาง (short = BUY, long = SELL)
            if not ic.call_short_closed:
                self.broker.close_option_leg(ic.call_short.symbol, is_short=True)
            self.broker.close_option_leg(ic.call_long.symbol, is_short=False)
            if not ic.put_short_closed:
                self.broker.close_option_leg(ic.put_short.symbol, is_short=True)
            self.broker.close_option_leg(ic.put_long.symbol, is_short=False)

            # record outcome (EOD)
            call_cost = self.broker.get_option_quote(ic.call_short.symbol) or 0.0
            put_cost  = self.broker.get_option_quote(ic.put_short.symbol) or 0.0
            pnl = ic.total_premium - (call_cost + put_cost) * 100
            self.record_trade_outcome(ic, pnl, "eod")
            logger.info(f"🔒 EOD Closed: {ic.trade_id} | PnL≈${pnl:.0f}")
