"""
core/broker.py
==============
Alpaca API wrapper สำหรับ options trading
ใช้ alpaca-py library (alpaca.markets/sdk)
"""

from __future__ import annotations
import uuid
from datetime import date
from typing import Optional
import logging

from alpaca.trading.client import TradingClient
from alpaca.trading.requests import (
    LimitOrderRequest, MarketOrderRequest,
    StopLimitOrderRequest, StopOrderRequest,
)
from alpaca.trading.enums import OrderSide, TimeInForce
from alpaca.data.historical.option import OptionHistoricalDataClient
from alpaca.data.requests import OptionChainRequest

from config import (
    ALPACA_API_KEY, ALPACA_API_SECRET, ALPACA_BASE_URL,
    UNDERLYING_SYMBOL
)
from core.options_engine import IronCondor, TAKE_PROFIT_PRICE

logger = logging.getLogger("0dte_bot")


class AlpacaBroker:
    """Wrapper รอบ Alpaca Trading + Data API"""

    def __init__(self):
        self.trading = TradingClient(
            api_key=ALPACA_API_KEY,
            secret_key=ALPACA_API_SECRET,
            paper=True                        # Paper trading mode
        )
        self.data = OptionHistoricalDataClient(
            api_key=ALPACA_API_KEY,
            secret_key=ALPACA_API_SECRET
        )
        logger.info("✅ AlpacaBroker initialized (PAPER MODE)")

    # ── Account Info ──────────────────────────────────────────────────────────

    def get_account(self) -> dict:
        account = self.trading.get_account()
        return {
            "portfolio_value": float(account.portfolio_value),
            "buying_power":    float(account.buying_power),
            "cash":            float(account.cash),
            "equity":          float(account.equity),
        }

    def get_portfolio_value(self) -> float:
        return self.get_account()["portfolio_value"]

    # ── Market Data ───────────────────────────────────────────────────────────

    def get_underlying_price(self) -> float:
        """ดึงราคาปัจจุบันของ underlying (SPY)"""
        from alpaca.data.historical import StockHistoricalDataClient
        from alpaca.data.requests import StockLatestQuoteRequest
        stock_client = StockHistoricalDataClient(ALPACA_API_KEY, ALPACA_API_SECRET)
        req = StockLatestQuoteRequest(symbol_or_symbols=UNDERLYING_SYMBOL)
        quote = stock_client.get_stock_latest_quote(req)
        return float(quote[UNDERLYING_SYMBOL].ask_price)

    def get_option_chain(self, expiry: str) -> list[dict]:
        """
        ดึง option chain สำหรับ expiry วันนี้
        expiry format: "YYYY-MM-DD"
        """
        req = OptionChainRequest(
            underlying_symbol=UNDERLYING_SYMBOL,
            expiration_date=date.fromisoformat(expiry),
        )
        try:
            chain = self.data.get_option_chain(req)
            result = []
            for symbol, snapshot in chain.items():
                greeks = snapshot.greeks if snapshot.greeks else None
                result.append({
                    "symbol": symbol,
                    "strike": float(snapshot.latest_quote.ask_price) if snapshot.latest_quote else 0,
                    "type":   "C" if "C" in symbol else "P",
                    "bid":    float(snapshot.latest_quote.bid_price) if snapshot.latest_quote else 0,
                    "ask":    float(snapshot.latest_quote.ask_price) if snapshot.latest_quote else 0,
                    "delta":  float(greeks.delta) if greeks else 0,
                    "iv":     float(greeks.implied_volatility) if greeks else 0,
                })
            logger.info(f"📊 Fetched {len(result)} options for expiry {expiry}")
            return result
        except Exception as e:
            logger.error(f"❌ Failed to fetch option chain: {e}")
            return []

    def get_bars_5min(self, symbol: str, limit: int = 10) -> list[dict]:
        """ดึงแท่งเทียน 5 นาทีล่าสุด สำหรับเช็ค flat market signal"""
        from alpaca.data.historical import StockHistoricalDataClient
        from alpaca.data.requests import StockBarsRequest
        from alpaca.data.timeframe import TimeFrame
        stock_client = StockHistoricalDataClient(ALPACA_API_KEY, ALPACA_API_SECRET)
        req = StockBarsRequest(
            symbol_or_symbols=UNDERLYING_SYMBOL,
            timeframe=TimeFrame.Minute,
            limit=limit * 5,
        )
        bars = stock_client.get_stock_bars(req)
        result = []
        for bar in bars[UNDERLYING_SYMBOL][-limit:]:
            result.append({
                "t": str(bar.timestamp),
                "o": bar.open,
                "h": bar.high,
                "l": bar.low,
                "c": bar.close,
                "v": bar.volume,
            })
        return result

    # ── Order Placement ───────────────────────────────────────────────────────

    def place_credit_spread(
        self,
        short_symbol: str,
        long_symbol: str,
        qty: int = 1
    ) -> Optional[str]:
        """
        ส่ง Credit Spread (Call หรือ Put) แยกทีละ leg
        ตามคู่มือ: ส่ง Call ก่อน แล้วตาม Put ทันที เพื่อลด slippage
        คืน order_id ของ short leg
        """
        try:
            # Sell Short leg
            short_order = self.trading.submit_order(
                LimitOrderRequest(
                    symbol=short_symbol,
                    qty=qty,
                    side=OrderSide.SELL,
                    time_in_force=TimeInForce.DAY,
                    limit_price=None,   # Market for paper trading
                )
            )
            # Buy Long leg (immediately after)
            long_order = self.trading.submit_order(
                LimitOrderRequest(
                    symbol=long_symbol,
                    qty=qty,
                    side=OrderSide.BUY,
                    time_in_force=TimeInForce.DAY,
                    limit_price=None,
                )
            )
            logger.info(f"📤 Spread placed | Short: {short_symbol} | Long: {long_symbol}")
            return str(short_order.id)
        except Exception as e:
            logger.error(f"❌ Failed to place spread: {e}")
            return None

    def place_oco_stop_on_short(
        self,
        short_symbol: str,
        stop_value: float,
        qty: int = 1
    ) -> tuple[Optional[str], Optional[str]]:
        """
        ตั้ง OCO Stop บนขา Short เท่านั้น (ตามคู่มือ)
        ด่าน 1: Stop Limit Order  (stop=stop_value, limit=stop_value + 40pts)
        ด่าน 2: Stop Market Order (trigger = stop_value + 70pts)
        คืน (stop_limit_id, stop_market_id)
        """
        from config import STOP_LIMIT_BUFFER_POINTS, STOP_MARKET_BUFFER_POINTS

        limit_price  = round(stop_value + STOP_LIMIT_BUFFER_POINTS * 0.01, 2)
        market_price = round(stop_value + (STOP_LIMIT_BUFFER_POINTS + STOP_MARKET_BUFFER_POINTS) * 0.01, 2)

        try:
            sl_order = self.trading.submit_order(
                StopLimitOrderRequest(
                    symbol=short_symbol,
                    qty=qty,
                    side=OrderSide.BUY,         # Buy to Close
                    time_in_force=TimeInForce.DAY,
                    stop_price=stop_value,
                    limit_price=limit_price,
                )
            )

            sm_order = self.trading.submit_order(
                StopOrderRequest(
                    symbol=short_symbol,
                    qty=qty,
                    side=OrderSide.BUY,
                    time_in_force=TimeInForce.DAY,
                    stop_price=market_price,
                )
            )
            logger.info(
                f"🛑 OCO Stop set on {short_symbol} | "
                f"StopLimit=${stop_value:.2f}→${limit_price:.2f} | "
                f"StopMarket trigger=${market_price:.2f}"
            )
            return str(sl_order.id), str(sm_order.id)
        except Exception as e:
            logger.error(f"❌ Failed to place OCO stop: {e}")
            return None, None

    def place_take_profit_on_short(
        self,
        short_symbol: str,
        qty: int = 1
    ) -> Optional[str]:
        """
        ตั้ง Take Profit Buy-to-Close ที่ $0.05 บนขา Short
        """
        try:
            tp_order = self.trading.submit_order(
                LimitOrderRequest(
                    symbol=short_symbol,
                    qty=qty,
                    side=OrderSide.BUY,
                    time_in_force=TimeInForce.DAY,
                    limit_price=TAKE_PROFIT_PRICE,
                )
            )
            logger.info(f"🎯 Take Profit set on {short_symbol} @ $0.05")
            return str(tp_order.id)
        except Exception as e:
            logger.error(f"❌ Failed to place take profit: {e}")
            return None

    def cancel_order(self, order_id: str):
        """ยกเลิก order ด้วย ID"""
        try:
            self.trading.cancel_order_by_id(order_id)
            logger.info(f"🚫 Order cancelled: {order_id}")
        except Exception as e:
            logger.warning(f"⚠️ Could not cancel order {order_id}: {e}")

    def close_position(self, symbol: str, qty: int = 1):
        """ปิด position ด้วย Market Order"""
        try:
            self.trading.submit_order(
                MarketOrderRequest(
                    symbol=symbol,
                    qty=qty,
                    side=OrderSide.SELL,
                    time_in_force=TimeInForce.DAY,
                )
            )
            logger.info(f"🔒 Position closed: {symbol}")
        except Exception as e:
            logger.error(f"❌ Failed to close position {symbol}: {e}")

    def get_option_quote(self, symbol: str) -> Optional[float]:
        """ดึงราคา mid price ของ option แบบ real-time"""
        try:
            from alpaca.data.requests import OptionLatestQuoteRequest
            req = OptionLatestQuoteRequest(symbol_or_symbols=symbol)
            quote = self.data.get_option_latest_quote(req)
            q = quote[symbol]
            return round((float(q.bid_price) + float(q.ask_price)) / 2, 2)
        except Exception as e:
            logger.warning(f"⚠️ Cannot get quote for {symbol}: {e}")
            return None

    def get_open_orders(self) -> list:
        """ดึง open orders ทั้งหมด"""
        try:
            return self.trading.get_orders()
        except Exception as e:
            logger.error(f"❌ get_open_orders failed: {e}")
            return []
