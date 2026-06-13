"""
utils/market.py - Market Hours & Signal Detection
"""

from datetime import datetime, time
import pytz
from config import (
    MARKET_OPEN_HOUR, MARKET_OPEN_MIN,
    MARKET_CLOSE_HOUR, MARKET_CLOSE_MIN,
    LAST_ENTRY_HOUR, ENTRY_DELAY_MINUTES,
    SIGNAL_MIN_FLAT_BARS, FLAT_BAR_RANGE_THRESHOLD
)

EST = pytz.timezone("US/Eastern")


def now_est() -> datetime:
    return datetime.now(EST)


def is_market_open() -> bool:
    """ตรวจสอบว่าตลาดเปิดอยู่ไหม (ไม่นับวันหยุด - Alpaca จัดการให้)"""
    n = now_est()
    market_open  = time(MARKET_OPEN_HOUR, MARKET_OPEN_MIN)
    market_close = time(MARKET_CLOSE_HOUR, MARKET_CLOSE_MIN)
    return market_open <= n.time() < market_close


def is_past_entry_delay() -> bool:
    """รอ 15 นาทีหลังเปิดตลาดก่อนเข้าชุดแรก"""
    n = now_est()
    earliest = time(MARKET_OPEN_HOUR, MARKET_OPEN_MIN + ENTRY_DELAY_MINUTES)
    return n.time() >= earliest


def can_open_new_trade() -> bool:
    """ไม่เปิดสัญญาใหม่หลัง LAST_ENTRY_HOUR"""
    n = now_est()
    return n.time() < time(LAST_ENTRY_HOUR, 0)


def is_flat_market(bars: list) -> bool:
    """
    ตรวจว่าตลาดนิ่งพอจะเข้าเทรดไหม
    bars: list of dicts with keys 'high', 'low', 'close'
    ต้องมี flat bars ติดกันอย่างน้อย SIGNAL_MIN_FLAT_BARS แท่ง
    """
    if len(bars) < SIGNAL_MIN_FLAT_BARS:
        return False

    recent = bars[-SIGNAL_MIN_FLAT_BARS:]
    for bar in recent:
        high  = float(bar["h"])
        low   = float(bar["l"])
        close = float(bar["c"])
        if close == 0:
            return False
        bar_range_pct = (high - low) / close
        if bar_range_pct > FLAT_BAR_RANGE_THRESHOLD:
            return False
    return True


def minutes_to_next_hour() -> int:
    """คืนจำนวนนาทีที่เหลือไปถึงชั่วโมงถัดไป"""
    n = now_est()
    return 60 - n.minute
