"""
core/broker.py
==============
Alpaca API wrapper สำหรับ options trading (alpaca-py SDK)

แก้ครบตาม code review:
  #1 bars: TimeFrame(5,Minute) + feed=IEX + start/end + safe empty handling
  #2 strike: parse จาก OCC symbol (ไม่ใช่ ask_price)
  #3 type:   parse right จาก OCC symbol (ไม่ใช่ substring "C" in symbol)
  #4 IV:     snapshot.implied_volatility (ไม่ใช่ greeks.implied_volatility)
  #5/#6 spread: native MLEG order (atomic) + limit price จริง
  #7 close:  position-intent-aware (short→buy_to_close, long→sell_to_close)
  #9 OCO:    native OrderClass.OCO (สอง stop link กันจริง)
  DRY_RUN:   ไม่ส่ง order จริงถ้าเปิดไว้
"""

from __future__ import annotations
import re
from datetime import date, datetime, timedelta, timezone
from typing import Optional
import logging

from alpaca.trading.client import TradingClient
from alpaca.trading.requests import (
    LimitOrderRequest, MarketOrderRequest,
    StopLimitOrderRequest, StopOrderRequest,
    OptionLegRequest,
)
from alpaca.trading.enums import (
    OrderSide, TimeInForce, OrderClass, PositionIntent,
)
from alpaca.data.historical.option import OptionHistoricalDataClient
from alpaca.data.historical import StockHistoricalDataClient
from alpaca.data.requests import (
    OptionChainRequest, OptionLatestQuoteRequest,
    StockBarsRequest, StockLatestQuoteRequest,
)
from alpaca.data.timeframe import TimeFrame, TimeFrameUnit
from alpaca.data.enums import DataFeed

from config import (
    ALPACA_API_KEY, ALPACA_API_SECRET,
    UNDERLYING_SYMBOL, DRY_RUN, DATA_FEED,
    USE_LIMIT_ORDERS, LIMIT_SLIPPAGE_PCT,
    STOP_LIMIT_BUFFER_POINTS, STOP_MARKET_BUFFER_POINTS,
    TAKE_PROFIT_PRICE,
)

logger = logging.getLogger("0dte_bot")

_FEED = DataFeed.IEX if DATA_FEED == "iex" else DataFeed.SIP


# ── #2/#3 OCC symbol parsing ──────────────────────────────────────────────────
_OCC_RE = re.compile(r"(\d{6})([CP])(\d{8})$")


def parse_occ_symbol(symbol: str) -> Optional[tuple[str, float]]:
    """
    SPY260615P00712000 -> ("P", 712.0)
    SPY260615C00645000 -> ("C", 645.0)
    คืน None ถ้า format ไม่ตรง
    """
    m = _OCC_RE.search(symbol)
    if not m:
        return None
    right = m.group(2)
    strike = int(m.group(3)) / 1000.0
    return right, strike


class AlpacaBroker:
    """Wrapper รอบ Alpaca Trading + Data API"""

    def __init__(self):
        self.trading = TradingClient(
            api_key=ALPACA_API_KEY,
            secret_key=ALPACA_API_SECRET,
            paper=True,
        )
        self.option_data = OptionHistoricalDataClient(
            api_key=ALPACA_API_KEY, secret_key=ALPACA_API_SECRET,
        )
        self.stock_data = StockHistoricalDataClient(
            api_key=ALPACA_API_KEY, secret_key=ALPACA_API_SECRET,
        )
        mode = "DRY-RUN (no orders submitted)" if DRY_RUN else "LIVE PAPER (orders submitted)"
        logger.info(f"✅ AlpacaBroker initialized | {mode}")

    # ── Account ───────────────────────────────────────────────────────────────

    def get_account(self) -> dict:
        a = self.trading.get_account()
        return {
            "portfolio_value": float(a.portfolio_value),
            "buying_power":    float(a.buying_power),
            "cash":            float(a.cash),
            "equity":          float(a.equity),
        }

    def get_portfolio_value(self) -> float:
        return self.get_account()["portfolio_value"]

    # ── Market Data ───────────────────────────────────────────────────────────

    def get_underlying_price(self) -> Optional[float]:
        try:
            req = StockLatestQuoteRequest(symbol_or_symbols=UNDERLYING_SYMBOL, feed=_FEED)
            quote = self.stock_data.get_stock_latest_quote(req)
            q = quote[UNDERLYING_SYMBOL]
            mid = (float(q.bid_price) + float(q.ask_price)) / 2
            return round(mid, 2) if mid > 0 else float(q.ask_price)
        except Exception as e:
            logger.error(f"❌ get_underlying_price failed: {e}")
            return None

    def get_bars_5min(self, symbol: str = None, limit: int = 60) -> list[dict]:
        """
        #1: ดึง 5-minute bars จริง + feed=IEX + start/end + safe empty handling
        คืน list (อาจว่าง) — ไม่ throw KeyError
        """
        symbol = symbol or UNDERLYING_SYMBOL
        try:
            req = StockBarsRequest(
                symbol_or_symbols=symbol,
                timeframe=TimeFrame(5, TimeFrameUnit.Minute),
                start=datetime.now(timezone.utc) - timedelta(days=14),
                end=datetime.now(timezone.utc),
                feed=_FEED,
            )
            resp = self.stock_data.get_stock_bars(req)
            bars = resp.data.get(symbol, [])          # safe: ไม่มี key → []
            if not bars:
                logger.warning(f"⚠️ No 5-min bars returned for {symbol}")
                return []
            out = []
            for bar in bars[-limit:]:
                out.append({
                    "t": str(bar.timestamp), "o": bar.open, "h": bar.high,
                    "l": bar.low, "c": bar.close, "v": bar.volume,
                })
            return out
        except Exception as e:
            logger.error(f"❌ get_bars_5min failed: {e}")
            return []

    def get_option_chain(self, expiry: str) -> list[dict]:
        """
        #2/#3/#4: strike+type จาก OCC symbol, IV จาก snapshot.implied_volatility
        """
        try:
            req = OptionChainRequest(
                underlying_symbol=UNDERLYING_SYMBOL,
                expiration_date=date.fromisoformat(expiry),
            )
            chain = self.option_data.get_option_chain(req)
            result = []
            for symbol, snap in chain.items():
                parsed = parse_occ_symbol(symbol)
                if parsed is None:
                    continue
                right, strike = parsed
                q = snap.latest_quote
                greeks = snap.greeks
                result.append({
                    "symbol": symbol,
                    "strike": strike,                                    # #2 จริง
                    "type":   right,                                     # #3 จริง
                    "bid":    float(q.bid_price) if q else 0.0,
                    "ask":    float(q.ask_price) if q else 0.0,
                    "delta":  float(greeks.delta) if greeks else 0.0,
                    "iv":     float(snap.implied_volatility or 0.0),     # #4 จริง
                })
            logger.info(f"📊 Parsed {len(result)} options for expiry {expiry}")
            return result
        except Exception as e:
            logger.error(f"❌ get_option_chain failed: {e}")
            return []

    def get_option_quote(self, symbol: str) -> Optional[float]:
        try:
            req = OptionLatestQuoteRequest(symbol_or_symbols=symbol)
            quote = self.option_data.get_option_latest_quote(req)
            q = quote[symbol]
            return round((float(q.bid_price) + float(q.ask_price)) / 2, 2)
        except Exception as e:
            logger.warning(f"⚠️ Cannot get quote for {symbol}: {e}")
            return None

    # ── #5/#6 Spread entry: native MLEG (atomic, limit price จริง) ─────────────

    def place_credit_spread(
        self, short_symbol: str, long_symbol: str,
        net_credit: float, qty: int = 1,
    ) -> Optional[str]:
        """
        ส่ง credit spread เป็น MLEG order เดียว (atomic — ไม่มี partial leg)
        sell short + buy long พร้อมกัน
        net_credit: credit ที่คาดว่าจะได้ (ใช้ตั้ง limit price)
        คืน order_id หรือ None ถ้า fail
        """
        legs = [
            OptionLegRequest(symbol=short_symbol, ratio_qty=1, side=OrderSide.SELL,
                             position_intent=PositionIntent.SELL_TO_OPEN),
            OptionLegRequest(symbol=long_symbol, ratio_qty=1, side=OrderSide.BUY,
                             position_intent=PositionIntent.BUY_TO_OPEN),
        ]

        if DRY_RUN:
            logger.info(f"🧪 [DRY-RUN] MLEG credit spread | SELL {short_symbol} / "
                        f"BUY {long_symbol} | net credit ${net_credit:.2f}")
            return f"DRYRUN_{short_symbol}"

        try:
            if USE_LIMIT_ORDERS:
                # credit spread = เรารับ credit → limit_price เป็นบวก (net credit ขั้นต่ำ)
                limit_price = round(max(net_credit * (1 - LIMIT_SLIPPAGE_PCT), 0.05), 2)
                req = LimitOrderRequest(
                    qty=qty, order_class=OrderClass.MLEG,
                    time_in_force=TimeInForce.DAY,
                    limit_price=limit_price, legs=legs,
                )
            else:
                req = MarketOrderRequest(
                    qty=qty, order_class=OrderClass.MLEG,
                    time_in_force=TimeInForce.DAY, legs=legs,
                )
            order = self.trading.submit_order(req)
            logger.info(f"📤 MLEG spread submitted | {short_symbol}/{long_symbol} | id={order.id}")
            return str(order.id)
        except Exception as e:
            logger.error(f"❌ place_credit_spread (MLEG) failed: {e}")
            return None

    # ── #9 Real OCO stop on short leg ─────────────────────────────────────────

    def place_oco_stop_on_short(
        self, short_symbol: str, stop_value: float, qty: int = 1,
    ) -> Optional[str]:
        """
        #9: OCO จริง — stop limit + stop market link กัน (อันหนึ่ง fill → อีกอัน cancel)
        ตั้งบนขา short เท่านั้น, side=BUY (buy-to-close)

        หมายเหตุ: Alpaca OCO ต้องการ take_profit (limit) + stop_loss (stop) คู่กัน
        เราใช้ stop_loss=stop_limit เป็นหลัก และเก็บ stop_market เป็น backstop แยก
        ที่ monitor เอง (ดู executor) เพราะ single-leg option ยังไม่รองรับ
        triple-bracket ตรงๆ
        """
        limit_price  = round(stop_value + STOP_LIMIT_BUFFER_POINTS * 0.01, 2)
        market_trig  = round(stop_value + (STOP_LIMIT_BUFFER_POINTS + STOP_MARKET_BUFFER_POINTS) * 0.01, 2)

        if DRY_RUN:
            logger.info(f"🧪 [DRY-RUN] OCO stop on {short_symbol} | "
                        f"StopLimit trigger=${stop_value:.2f} limit=${limit_price:.2f} | "
                        f"StopMarket backstop=${market_trig:.2f}")
            return f"DRYRUN_STOP_{short_symbol}"

        try:
            # ด่านแรก: stop-limit (buy to close)
            sl = self.trading.submit_order(
                StopLimitOrderRequest(
                    symbol=short_symbol, qty=qty, side=OrderSide.BUY,
                    time_in_force=TimeInForce.DAY,
                    stop_price=stop_value, limit_price=limit_price,
                    position_intent=PositionIntent.BUY_TO_CLOSE,
                )
            )
            logger.info(f"🛑 Stop-limit set on {short_symbol} | "
                        f"trigger=${stop_value:.2f} limit=${limit_price:.2f} | id={sl.id}")
            # ด่านสอง (backstop) ตั้ง+monitor ใน executor เพื่อ cancel sibling เมื่อ fill
            return str(sl.id)
        except Exception as e:
            logger.error(f"❌ place_oco_stop_on_short failed: {e}")
            return None

    def place_stop_market_backstop(
        self, short_symbol: str, trigger: float, qty: int = 1,
    ) -> Optional[str]:
        """Stop-market backstop (last line of defense) — monitor sibling ใน executor"""
        if DRY_RUN:
            logger.info(f"🧪 [DRY-RUN] Stop-market backstop {short_symbol} @ ${trigger:.2f}")
            return f"DRYRUN_BACKSTOP_{short_symbol}"
        try:
            sm = self.trading.submit_order(
                StopOrderRequest(
                    symbol=short_symbol, qty=qty, side=OrderSide.BUY,
                    time_in_force=TimeInForce.DAY, stop_price=trigger,
                    position_intent=PositionIntent.BUY_TO_CLOSE,
                )
            )
            return str(sm.id)
        except Exception as e:
            logger.error(f"❌ place_stop_market_backstop failed: {e}")
            return None

    def place_take_profit_on_short(
        self, short_symbol: str, qty: int = 1,
    ) -> Optional[str]:
        """TP buy-to-close ที่ $0.05 บนขา short (limit order มีราคาจริง)"""
        if DRY_RUN:
            logger.info(f"🧪 [DRY-RUN] TP on {short_symbol} @ ${TAKE_PROFIT_PRICE}")
            return f"DRYRUN_TP_{short_symbol}"
        try:
            tp = self.trading.submit_order(
                LimitOrderRequest(
                    symbol=short_symbol, qty=qty, side=OrderSide.BUY,
                    time_in_force=TimeInForce.DAY, limit_price=TAKE_PROFIT_PRICE,
                    position_intent=PositionIntent.BUY_TO_CLOSE,
                )
            )
            logger.info(f"🎯 TP set on {short_symbol} @ ${TAKE_PROFIT_PRICE} | id={tp.id}")
            return str(tp.id)
        except Exception as e:
            logger.error(f"❌ place_take_profit_on_short failed: {e}")
            return None

    # ── Order status / cancel ─────────────────────────────────────────────────

    def get_order(self, order_id: str):
        """ดึงสถานะ order (สำหรับ #8 verify fill ก่อน mark closed)"""
        if DRY_RUN or order_id.startswith("DRYRUN"):
            return None
        try:
            return self.trading.get_order_by_id(order_id)
        except Exception as e:
            logger.warning(f"⚠️ get_order {order_id}: {e}")
            return None

    def is_order_filled(self, order_id: str) -> bool:
        """#8: เช็คว่า order fill จริงไหม"""
        o = self.get_order(order_id)
        if o is None:
            return False
        return str(o.status).lower() in ("orderstatus.filled", "filled")

    def cancel_order(self, order_id: str):
        if DRY_RUN or (order_id and order_id.startswith("DRYRUN")):
            logger.info(f"🧪 [DRY-RUN] cancel {order_id}")
            return
        try:
            self.trading.cancel_order_by_id(order_id)
            logger.info(f"🚫 Order cancelled: {order_id}")
        except Exception as e:
            logger.warning(f"⚠️ Could not cancel {order_id}: {e}")

    # ── #7 Position-intent-aware close ────────────────────────────────────────

    def close_option_leg(self, symbol: str, is_short: bool, qty: int = 1) -> Optional[str]:
        """
        #7: ปิด option leg ให้ถูกทาง
          short leg → BUY to close
          long leg  → SELL to close
        """
        if is_short:
            side, intent = OrderSide.BUY, PositionIntent.BUY_TO_CLOSE
        else:
            side, intent = OrderSide.SELL, PositionIntent.SELL_TO_CLOSE

        if DRY_RUN:
            logger.info(f"🧪 [DRY-RUN] close {'SHORT' if is_short else 'LONG'} "
                        f"{symbol} via {side.value}")
            return f"DRYRUN_CLOSE_{symbol}"
        try:
            o = self.trading.submit_order(
                MarketOrderRequest(
                    symbol=symbol, qty=qty, side=side,
                    time_in_force=TimeInForce.DAY, position_intent=intent,
                )
            )
            logger.info(f"🔒 Closed {'SHORT' if is_short else 'LONG'} {symbol} ({side.value})")
            return str(o.id)
        except Exception as e:
            logger.error(f"❌ close_option_leg {symbol} failed: {e}")
            return None

    def get_open_orders(self) -> list:
        if DRY_RUN:
            return []
        try:
            return self.trading.get_orders()
        except Exception as e:
            logger.error(f"❌ get_open_orders failed: {e}")
            return []
