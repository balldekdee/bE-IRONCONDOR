"""
utils/logger.py - Logging & Trade Journal
"""

import logging
import csv
import os
from datetime import datetime
from config import LOG_DIR, LOG_LEVEL, TRADE_LOG_FILE


def setup_logger(name: str = "0dte_bot") -> logging.Logger:
    """ตั้งค่า logger ให้แสดงทั้ง console และ file"""
    os.makedirs(LOG_DIR, exist_ok=True)

    logger = logging.getLogger(name)
    logger.setLevel(getattr(logging, LOG_LEVEL))

    fmt = logging.Formatter(
        "%(asctime)s | %(levelname)-8s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )

    # Console handler
    ch = logging.StreamHandler()
    ch.setFormatter(fmt)
    logger.addHandler(ch)

    # File handler
    log_file = os.path.join(LOG_DIR, f"bot_{datetime.now().strftime('%Y%m%d')}.log")
    fh = logging.FileHandler(log_file)
    fh.setFormatter(fmt)
    logger.addHandler(fh)

    return logger


TRADE_LOG_HEADERS = [
    "timestamp", "trade_id", "action",
    "call_short_strike", "call_long_strike",
    "put_short_strike", "put_long_strike",
    "call_premium", "put_premium", "total_premium",
    "stop_loss_value", "expiry",
    "pnl", "outcome", "notes"
]


def log_trade(record: dict):
    """บันทึก trade record ลง CSV (สร้างไฟล์อัตโนมัติถ้ายังไม่มี)"""
    os.makedirs(LOG_DIR, exist_ok=True)
    file_exists = os.path.exists(TRADE_LOG_FILE)

    with open(TRADE_LOG_FILE, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=TRADE_LOG_HEADERS)
        if not file_exists:
            writer.writeheader()
        # เติม key ที่ขาดหายไปด้วย empty string
        row = {k: record.get(k, "") for k in TRADE_LOG_HEADERS}
        writer.writerow(row)
