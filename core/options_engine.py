"""
core/options_engine.py
======================
Logic หลักสำหรับเลือก Strike, คำนวณ Stop Loss,
และสร้าง Iron Condor structure
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import Optional
import math
from config import (
    TARGET_DELTA, TARGET_PREMIUM_PER_SIDE, MIN_PREMIUM_PER_SIDE,
    DEFAULT_WING_WIDTH, WING_WIDTH_OPTIONS,
    STOP_LIMIT_BUFFER_POINTS, STOP_MARKET_BUFFER_POINTS,
    TAKE_PROFIT_PRICE
)


@dataclass
class Leg:
    symbol: str
    strike: float
    right: str          # "C" or "P"
    expiry: str         # YYYY-MM-DD
    action: str         # "sell" or "buy"
    qty: int = 1
    order_id: Optional[str] = None
    fill_price: Optional[float] = None


@dataclass
class IronCondor:
    """โครงสร้าง Iron Condor 1 ชุด"""
    trade_id: str
    expiry: str
    underlying_price: float

    # Legs
    call_short: Optional[Leg] = None
    call_long:  Optional[Leg] = None
    put_short:  Optional[Leg] = None
    put_long:   Optional[Leg] = None

    # Premiums (filled after execution)
    call_premium: float = 0.0
    put_premium:  float = 0.0

    # Stop Loss Orders
    call_stop_order_id: Optional[str] = None
    put_stop_order_id:  Optional[str] = None
    # Stop-market backstop (last line of defense) — monitored to cancel sibling
    call_backstop_order_id: Optional[str] = None
    put_backstop_order_id:  Optional[str] = None
    # Take-profit order ids (#8: verify fill before marking closed)
    call_tp_order_id: Optional[str] = None
    put_tp_order_id:  Optional[str] = None
    # which short legs already closed (TP filled)
    call_short_closed: bool = False
    put_short_closed:  bool = False

    # Status
    status: str = "pending"   # pending | open | closed | stopped
    pnl: float = 0.0

    @property
    def total_premium(self) -> float:
        return self.call_premium + self.put_premium

    @property
    def stop_loss_value(self) -> float:
        """
        Stop Loss = Total Premium ที่เก็บได้ทั้งสองฝั่ง
        ตั้งที่ขา Short แต่ละฝั่ง
        """
        return self.total_premium

    @property
    def stop_limit_price(self) -> float:
        """ราคา Stop สำหรับ Stop Limit Order (OCO ด่านแรก)"""
        return self.stop_loss_value

    @property
    def stop_limit_limit_price(self) -> float:
        """Limit price ของ Stop Limit = stop + 40 จุด buffer"""
        return self.stop_loss_value + STOP_LIMIT_BUFFER_POINTS * 0.01

    @property
    def stop_market_trigger(self) -> float:
        """Stop Market trigger (last line of defense) = stop + 70 จุด"""
        return self.stop_loss_value + (STOP_LIMIT_BUFFER_POINTS + STOP_MARKET_BUFFER_POINTS) * 0.01


def select_wing_width(call_premium_at_default: float,
                      put_premium_at_default: float) -> tuple[int, int]:
    """
    ปรับ wing width แต่ละฝั่งให้ได้ premium สมดุลกัน (equal premium)
    คืนค่า (call_width, put_width)
    """
    ratio = call_premium_at_default / max(put_premium_at_default, 0.01)

    if ratio > 1.2:
        # call premium สูงกว่ามาก → แคบ call, ขยาย put
        return DEFAULT_WING_WIDTH - 5, DEFAULT_WING_WIDTH + 5
    elif ratio < 0.8:
        # put premium สูงกว่ามาก → ขยาย call, แคบ put
        return DEFAULT_WING_WIDTH + 5, DEFAULT_WING_WIDTH - 5
    else:
        return DEFAULT_WING_WIDTH, DEFAULT_WING_WIDTH


def find_strike_by_delta(option_chain: list[dict],
                          right: str,
                          target_delta: float) -> Optional[dict]:
    """
    หา strike ที่ใกล้กับ target_delta มากที่สุด
    option_chain: list of option dicts with 'strike', 'delta', 'ask', 'bid'
    right: 'C' or 'P'
    """
    candidates = [o for o in option_chain if o.get("type", "").upper() == right]
    if not candidates:
        return None

    # Delta สำหรับ Put จะเป็นลบ ใช้ absolute value
    best = min(candidates,
               key=lambda o: abs(abs(float(o.get("delta", 0))) - target_delta))
    return best


def calculate_mid_price(option: dict) -> float:
    """
    คำนวณ premium ต่อ contract (ดอลลาร์) จาก bid/ask
    quote ของ option เป็นราคา per-share → คูณ 100 (1 contract = 100 shares)
    เช่น mid $1.50/share → $150/contract
    """
    bid = float(option.get("bid", 0))
    ask = float(option.get("ask", 0))
    mid_per_share = (bid + ask) / 2
    return round(mid_per_share * 100, 2)   # → dollars per contract


def calculate_mid_per_share(option: dict) -> float:
    """mid price per-share (ใช้ตอนตั้ง limit price ของ order ซึ่งคิดเป็น per-share)"""
    bid = float(option.get("bid", 0))
    ask = float(option.get("ask", 0))
    return round((bid + ask) / 2, 2)


def is_premium_acceptable(premium: float) -> bool:
    """ตรวจว่า premium (ต่อ contract, ดอลลาร์) อยู่ในเป้าหมาย"""
    return MIN_PREMIUM_PER_SIDE <= premium <= TARGET_PREMIUM_PER_SIDE * 1.5


def build_iron_condor_structure(
    trade_id: str,
    expiry: str,
    underlying_price: float,
    call_short_strike: float,
    call_long_strike: float,
    put_short_strike: float,
    put_long_strike: float,
    underlying_symbol: str = "SPY"
) -> IronCondor:
    """
    สร้าง IronCondor object จาก strikes ที่เลือกแล้ว
    call_long_strike > call_short_strike
    put_long_strike  < put_short_strike
    """
    def make_symbol(strike: float, right: str) -> str:
        """สร้าง OCC option symbol: SPY250117C00580000"""
        exp = expiry.replace("-", "")[2:]  # YYMMDD
        strike_int = int(strike * 1000)
        return f"{underlying_symbol}{exp}{right}{strike_int:08d}"

    ic = IronCondor(
        trade_id=trade_id,
        expiry=expiry,
        underlying_price=underlying_price,
        call_short=Leg(make_symbol(call_short_strike, "C"), call_short_strike, "C", expiry, "sell"),
        call_long =Leg(make_symbol(call_long_strike,  "C"), call_long_strike,  "C", expiry, "buy"),
        put_short =Leg(make_symbol(put_short_strike,  "P"), put_short_strike,  "P", expiry, "sell"),
        put_long  =Leg(make_symbol(put_long_strike,   "P"), put_long_strike,   "P", expiry, "buy"),
    )
    return ic


def calculate_max_loss(ic: IronCondor) -> float:
    """
    Max loss per side = (wing_width * 100) - premium_collected_per_side
    คืนค่า max loss รวมทั้ง IC (กรณีชน stop ทั้งสองฝั่ง)
    """
    call_wing = abs(ic.call_long.strike - ic.call_short.strike)
    put_wing  = abs(ic.put_short.strike - ic.put_long.strike)
    call_max_loss = (call_wing * 100) - (ic.call_premium * 100)
    put_max_loss  = (put_wing  * 100) - (ic.put_premium  * 100)
    # กรณีชน double stop = max loss ทั้งสองฝั่ง - total premium
    return call_max_loss + put_max_loss


def should_take_profit(current_short_price: float) -> bool:
    """ตรวจว่าถึงเวลา Take Profit ที่ $0.05 แล้วไหม"""
    return current_short_price <= TAKE_PROFIT_PRICE
