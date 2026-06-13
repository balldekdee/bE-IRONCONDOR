"""
0DTE Break-Even Iron Condor - Configuration
==========================================
แก้ค่าตรงนี้ก่อนรัน bot ทุกครั้ง
"""

import os

# ─── Alpaca API Credentials ───────────────────────────────────────────────────
# ใส่ key จาก https://app.alpaca.markets/paper-trading/overview
ALPACA_API_KEY    = os.getenv("ALPACA_API_KEY", "YOUR_PAPER_API_KEY")
ALPACA_API_SECRET = os.getenv("ALPACA_API_SECRET", "YOUR_PAPER_SECRET")
ALPACA_BASE_URL   = "https://paper-api.alpaca.markets"   # Paper trading endpoint

# ─── Underlying ───────────────────────────────────────────────────────────────
UNDERLYING_SYMBOL = "SPY"   # ใช้ SPY แทน SPX เพราะ Alpaca รองรับ SPY options
                              # SPX ยังไม่ fully supported ใน Alpaca options API

# ─── Risk Management ──────────────────────────────────────────────────────────
MAX_DAILY_RISK_PCT        = 0.02   # 2% ของพอร์ตต่อวัน (maximum total loss ที่ยอมรับได้)
MAX_BUYING_POWER_PCT      = 0.50   # ใช้ Buying Power ไม่เกิน 50% ของพอร์ต
MAX_TRADES_PER_DAY        = 7      # เปิดได้สูงสุด 7 ชุดต่อวัน
TRADE_INTERVAL_MINUTES    = 60     # เปิดทีละ 1 ชุดต่อชั่วโมง

# ─── Entry Parameters ─────────────────────────────────────────────────────────
ENTRY_DELAY_MINUTES       = 15     # รอหลังตลาดเปิด 15 นาที ก่อนเปิดชุดแรก
TARGET_DELTA              = 0.12   # Short strike delta target (10-15 delta)
TARGET_PREMIUM_PER_SIDE   = 150    # เป้าหมายพรีเมียมต่อฝั่ง ($100-$200)
MIN_PREMIUM_PER_SIDE      = 80     # ขั้นต่ำ ถ้าต่ำกว่านี้ไม่เปิด
DEFAULT_WING_WIDTH        = 30     # ความกว้างปีก 30 จุด (default)
WING_WIDTH_OPTIONS        = [25, 30, 35, 40]  # ตัวเลือกปรับปีกให้ equal premium

# ─── Stop Loss Parameters ─────────────────────────────────────────────────────
# Stop Loss = เท่ากับ Total Premium ที่เก็บได้ทั้ง Iron Condor
# ตั้ง Stop บน SHORT leg เท่านั้น (ไม่ตั้งบน spread)
STOP_LIMIT_BUFFER_POINTS  = 40     # ช่วงห่าง Stop vs Limit (OCO ด่านแรก)
STOP_MARKET_BUFFER_POINTS = 30     # ห่างออกไปอีก 30 จุด (OCO ด่านสอง / last resort)

# ─── Take Profit ──────────────────────────────────────────────────────────────
TAKE_PROFIT_PRICE         = 0.05   # ปิดขา Short เมื่อราคาลงมาถึง $0.05

# ─── Candle Filter (Entry Signal) ─────────────────────────────────────────────
SIGNAL_TIMEFRAME          = "5Min"   # กรอบเวลา 5 นาที
SIGNAL_MIN_FLAT_BARS      = 2        # จำนวนแท่งเทียนนิ่ง (range แคบ) ก่อนเข้าเทรด
FLAT_BAR_RANGE_THRESHOLD  = 0.0015   # ถ้า range < 0.15% ของราคา = "นิ่ง" (SPY ≈ 0.15% per 5-min bar)

# ─── Market Hours (EST) ───────────────────────────────────────────────────────
MARKET_OPEN_HOUR   = 9
MARKET_OPEN_MIN    = 30
MARKET_CLOSE_HOUR  = 16
MARKET_CLOSE_MIN   = 0
LAST_ENTRY_HOUR    = 15   # ไม่เปิดสัญญาใหม่หลัง 15:00 EST

# ─── Regime Intelligence Layer ────────────────────────────────────────────────
REGIME_ENABLED            = True   # เปิด/ปิด regime filter
ENCODER_SEQ_LEN           = 32     # ความยาว sequence ของ deep encoder (bars)
ENCODER_DEVICE            = "cpu"  # "cuda" ถ้ามี GPU

# Trade filter thresholds (ปรับความเข้มงวดของตัวกรอง)
MIN_REGIME_CONFIDENCE     = 0.45   # confidence ขั้นต่ำที่ยอมเทรด
MAX_CHANGE_POINT_PROB     = 0.35   # ถ้า change point สูงกว่านี้ → งดเทรด
MIN_EXPECTED_EDGE         = 0.0    # expected edge ($) ขั้นต่ำต่อ trade
FILTER_EXPLORATION        = True   # True=Thompson sampling, False=LCB (conservative)

# บันทึก regime snapshot ทุกกี่ bar (ลด DB write)
REGIME_SNAPSHOT_EVERY     = 3

# ─── Supabase ─────────────────────────────────────────────────────────────────
SUPABASE_URL  = os.getenv("SUPABASE_URL",  "YOUR_SUPABASE_URL")
SUPABASE_KEY  = os.getenv("SUPABASE_KEY",  "YOUR_SUPABASE_SERVICE_KEY")

# ─── Execution Safety ─────────────────────────────────────────────────────────
# DRY_RUN อ่านจาก env: export DRY_RUN=false เพื่อเปิดส่ง order จริง (default = true ปลอดภัย)
DRY_RUN          = os.getenv("DRY_RUN", "true").lower() in ("true", "1", "yes", "on")
DATA_FEED        = os.getenv("DATA_FEED", "iex")   # "iex" สำหรับ free/paper
USE_LIMIT_ORDERS = True    # True = limit order ที่ mid price, False = market order
LIMIT_SLIPPAGE_PCT = 0.10  # ยอมจ่ายเกิน mid ได้กี่ % (สำหรับ marketable limit)
FILL_TIMEOUT_SEC   = 10    # รอ MLEG fill นานสุดกี่วินาทีก่อน cancel/abort

# ─── Logging ──────────────────────────────────────────────────────────────────
LOG_DIR            = "logs"
LOG_LEVEL          = "INFO"
TRADE_LOG_FILE     = "logs/trade_log.csv"
