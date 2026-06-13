"""
pretrain_regime.py
==================
สคริปต์ pretrain regime engine ด้วยข้อมูลย้อนหลัง (optional)
รันครั้งเดียวก่อนเริ่มเทรดจริง เพื่อให้ encoder + HSMM warmed up
แล้วเซฟ state ลง Supabase (bot จะ resume ตอน start)

วิธีใช้:
  python pretrain_regime.py --days 30

ดึง 5-min bars ย้อนหลังจาก Alpaca → feed เข้า ensemble → save state
"""

import argparse
import logging
from datetime import datetime, timedelta

from alpaca.data.historical import StockHistoricalDataClient
from alpaca.data.requests import StockBarsRequest
from alpaca.data.timeframe import TimeFrame

from core.regime import RegimeEnsemble
from core.database import Database
from utils.logger import setup_logger
from config import (
    ALPACA_API_KEY, ALPACA_API_SECRET, UNDERLYING_SYMBOL,
    ENCODER_SEQ_LEN, SUPABASE_URL, SUPABASE_KEY,
)

logger = setup_logger("pretrain")


def fetch_historical_bars(days: int) -> list[dict]:
    client = StockHistoricalDataClient(ALPACA_API_KEY, ALPACA_API_SECRET)
    end = datetime.now()
    start = end - timedelta(days=days)
    req = StockBarsRequest(
        symbol_or_symbols=UNDERLYING_SYMBOL,
        timeframe=TimeFrame(5, TimeFrame.Minute.unit_value) if hasattr(TimeFrame, "unit_value") else TimeFrame.Minute,
        start=start,
        end=end,
    )
    bars = client.get_stock_bars(req)
    out = []
    for bar in bars[UNDERLYING_SYMBOL]:
        out.append({"o": bar.open, "h": bar.high, "l": bar.low,
                    "c": bar.close, "v": bar.volume})
    return out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--days", type=int, default=30, help="วันย้อนหลัง")
    parser.add_argument("--epochs", type=int, default=3, help="รอบ replay เพิ่มเติม")
    args = parser.parse_args()

    logger.info(f"📥 Fetching {args.days} days of 5-min bars for {UNDERLYING_SYMBOL}...")
    bars = fetch_historical_bars(args.days)
    logger.info(f"   Got {len(bars)} bars")

    eng = RegimeEnsemble(encoder_seq_len=ENCODER_SEQ_LEN)

    # multi-pass เพื่อให้ encoder เทรนได้เพียงพอ
    for epoch in range(args.epochs):
        for bar in bars:
            eng.update(bar)
        logger.info(f"   Epoch {epoch+1}/{args.epochs} | "
                    f"encoder batches: {eng.encoder._trained_batches} | "
                    f"warmed: {eng.encoder.is_warmed}")

    if eng.last_state:
        logger.info("\n📊 Final regime estimate after pretraining:")
        logger.info("\n" + eng.last_state.pretty())

    # save ลง DB + local encoder checkpoint
    db = Database(SUPABASE_URL, SUPABASE_KEY)
    db.save_model_state(eng.state_dict())
    eng.encoder.save("logs/encoder_pretrained.pt")
    logger.info("✅ Saved model state to DB + encoder to logs/encoder_pretrained.pt")


if __name__ == "__main__":
    main()
