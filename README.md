# 🤖 0DTE Break-Even Iron Condor Bot (Alpaca Paper Trading)

กลยุทธ์: 0DTE Break-Even Iron Condor บน SPY options  
Broker: Alpaca Paper Trading API  
ความเสี่ยง: 4/10 (เมื่อปฏิบัติตามกฎอย่างเคร่งครัด)

---

## โครงสร้างไฟล์

```
0dte_bot/
├── main.py                  ← รันตรงนี้
├── config.py                ← ตั้งค่าทั้งหมด (แก้ก่อนรัน)
├── requirements.txt
├── core/
│   ├── broker.py            ← Alpaca API wrapper
│   ├── options_engine.py    ← Strike selection, Stop Loss, Iron Condor structure
│   ├── executor.py          ← Trade lifecycle orchestrator
│   └── risk_manager.py      ← Daily risk tracking & position limits
├── utils/
│   ├── logger.py            ← Logging + Trade Log CSV
│   └── market.py            ← Market hours, flat market signal
└── logs/
    ├── bot_YYYYMMDD.log     ← Daily log
    └── trade_log.csv        ← Trade journal (วิเคราะห์ผลได้)
```

---

## วิธีตั้งค่าและรัน

### 1. ติดตั้ง dependencies
```bash
pip install -r requirements.txt
```

### 2. ตั้ง Alpaca API Keys
```bash
# รับ key จาก https://app.alpaca.markets/paper-trading/overview
export ALPACA_API_KEY="your_paper_api_key"
export ALPACA_API_SECRET="your_paper_api_secret"
```

หรือแก้ตรงใน `config.py`:
```python
ALPACA_API_KEY    = "your_paper_api_key"
ALPACA_API_SECRET = "your_paper_api_secret"
```

### 3. ตรวจสอบ config ใน config.py
ค่าสำคัญที่ควรปรับตามขนาดพอร์ต:
- `MAX_DAILY_RISK_PCT` = 0.02 (2%)
- `MAX_BUYING_POWER_PCT` = 0.50 (50%)
- `TARGET_PREMIUM_PER_SIDE` = 150 (เป้าหมาย $150/ฝั่ง)

### 4. รัน bot
```bash
cd 0dte_bot
python main.py
```

หยุด bot: `Ctrl+C` (bot จะปิด position ทั้งหมดก่อนหยุด)

---

## Logic สรุป

| เวลา EST | Bot ทำอะไร |
|----------|-----------|
| 09:30    | ตลาดเปิด – bot เริ่ม monitor |
| 09:45    | ตรวจ entry ชุดแรก (รอ 15 นาที) |
| ทุก :00  | ตรวจ flat market signal → เปิด IC ถ้าผ่านทุกเงื่อนไข |
| ตลอดวัน  | Monitor ทุก 60 วินาที: Take Profit $0.05, Stop Loss OCO |
| 15:45    | EOD cleanup: ปิดทุก position |

---

## กฎความเสี่ยงที่ bot บังคับใช้

1. **Max 7 trades/day** (ชั่วโมงละ 1 ชุด)
2. **Max daily loss = 2% ของพอร์ต** (หยุดเปิดใหม่ถ้าถึง)
3. **Stop Loss = Total Premium** (ตั้งบน Short leg เท่านั้น)
4. **OCO สองชั้น** = Stop Limit (40pts) + Stop Market (70pts)
5. **Take Profit = $0.05** บน Short legs
6. **Flat market required** = ต้องเห็นแท่งเทียน 5M นิ่ง 2 แท่งขึ้นไป

---

## ⚠️ ข้อควรระวัง

- Bot นี้ใช้สำหรับ **Paper Trading เท่านั้น** ในปัจจุบัน
- Alpaca Options API มี **SPY** แต่ยังไม่ fully support **SPX** index options
- ควรรัน paper trade อย่างน้อย 1 เดือนก่อนพิจารณา live trading
- ตรวจสอบ trade_log.csv ทุกวันเพื่อวิเคราะห์ผล

---

# 🧠 Regime Intelligence Layer (v2)

เพิ่ม regime prediction layer แบบ ensemble + self-improving + Supabase persistence
ทำให้ bot เทรด **เฉพาะ regime ที่มี edge จริง** แทนการเทรดทุกชั่วโมงแบบเดิม

## Ensemble (4 components)

| Component | ไฟล์ | หน้าที่ |
|-----------|------|---------|
| **Bayesian State-Space** (Kalman) | `core/regime/state_space.py` | ติดตาม latent vol/trend, denoise สัญญาณ + ผลิต "surprise" signal |
| **Online Change Point Detection** (BOCPD) | `core/regime/bocpd.py` | ตรวจจุดเปลี่ยน regime real-time → boost Transition + block entry |
| **Hidden Semi-Markov Model** | `core/regime/hsmm.py` | backbone 3-regime พร้อม duration modeling + online EM |
| **Deep Sequential Encoder** (GRU) | `core/regime/encoder.py` | learned representation, self-supervised + distilled จาก HSMM |

รวมกันด้วย **log-opinion pool** ใน `core/regime/ensemble.py` → output:

```
Regime 1: Risk-on / Low vol        62.0%  ███████████████
Regime 2: Transition / Choppy      25.0%  ██████
Regime 3: Crisis / High vol        13.0%  ███

Change-point prob:  1.2% | Confidence: 88.0% | Run length: 45 bars
```

## Self-Improving Loop (`core/self_improve.py`)

1. ทุก trade ที่ปิด → log `(regime ตอนเข้า, PnL)` เข้า **Normal-inverse-Gamma posterior** ต่อ regime
2. ตัวกรองคำนวณ `E[edge] = Σ P(regime) × edge(regime)` ด้วย **Thompson sampling**
3. เปิดเทรดเฉพาะเมื่อผ่าน 4 gates:
   - Regime confidence ≥ 45%
   - Change-point prob ≤ 35% (ไม่เข้าช่วง regime เปลี่ยน)
   - Expected edge > threshold
   - ไม่ใช่ Crisis regime
4. ยิ่งเทรดเยอะ → posterior ยิ่งแม่น → filter ปรับตัวเอง

## Supabase Setup

1. สร้าง project ที่ https://supabase.com
2. รัน `supabase_schema.sql` ใน SQL Editor
3. ตั้ง env:
   ```bash
   export SUPABASE_URL="https://xxxxx.supabase.co"
   export SUPABASE_KEY="your_service_role_key"
   ```
4. ตาราง: `trades`, `regime_snapshots`, `regime_posteriors`, `model_state`
   + view `regime_performance` (สรุป PnL/win-rate ต่อ regime)

> ไม่ตั้ง Supabase ก็รันได้ — bot จะ fallback เป็น local mode อัตโนมัติ

## Pretrain (optional)

```bash
python pretrain_regime.py --days 30 --epochs 3
```
warmup encoder + HSMM ด้วยข้อมูลย้อนหลัง แล้วเซฟ state ลง DB
bot จะ resume state นี้ตอน start (เรียนรู้ต่อเนื่องข้ามวัน)

## Data Flow

```
5-min bar
   ↓
FeatureEngine ──→ RunningNormalizer
   ↓
Kalman (latent vol/trend + surprise)
   ↓
BOCPD (change-point hazard)
   ↓
HSMM (regime posterior, online EM) ──┐
   ↓                                  │ distillation
Deep Encoder (learned posterior) ←────┘
   ↓
Log-opinion pool ──→ RegimeState
   ↓
SelfImprovingFilter (4 gates + learned edge)
   ↓
TradeExecutor (เปิด IC เฉพาะเมื่อผ่าน)
   ↓
Supabase (trades + regime_snapshots + learned posteriors)
```

---

# 🔧 Execution Hardening (v3) — Code Review Fixes

แก้ตาม code review ก่อนรัน paper trading จริง (ทั้งหมดผ่าน test):

| # | Bug | Fix |
|---|-----|-----|
| 1 | bars ใช้ TimeFrame.Minute + ไม่มี feed/start/end → `KeyError: SPY` | `TimeFrame(5,Minute)` + `feed=IEX` + start/end + `resp.data.get(symbol, [])` safe |
| 2 | strike = `ask_price` (ผิดร้ายแรง) | `parse_occ_symbol()` → strike จริงจาก OCC symbol |
| 3 | type = `"C" in symbol` (substring) | parse `right` จาก OCC symbol |
| 4 | IV = `greeks.implied_volatility` | `snapshot.implied_volatility` |
| 5 | `LimitOrderRequest(limit_price=None)` | limit price จริง (per-share) หรือ MarketOrderRequest |
| 6 | spread 2 legs แยก ไม่มี rollback | native **MLEG** order (atomic) + rollback ถ้า leg 2 fail |
| 7 | `close_position()` ใช้ SELL เสมอ | `close_option_leg(is_short)`: short→BUY_TO_CLOSE, long→SELL_TO_CLOSE |
| 8 | TP log "closed" โดยไม่ verify fill | `is_order_filled()` ก่อน mark closed; submit จริงถ้ายังไม่ fill |
| 9 | stop limit + stop market ไม่ link OCO | monitor sibling — fill อันหนึ่ง → cancel อีกอัน |
| 10 | expiry = `date.today()` (system tz) | `now_est().date()` (US/Eastern) — สำคัญสำหรับ 0DTE |
| + | premium per-share vs per-contract ปนกัน | `calculate_mid_price()` → per-contract; per-share แยกสำหรับ limit |

## ⚠️ DRY_RUN mode (default = True)

```python
# config.py
DRY_RUN = True   # ไม่ส่ง order จริง — log อย่างเดียว ทดสอบ flow ได้ปลอดภัย
```

รันทดสอบ flow ทั้งหมดได้โดยไม่แตะ paper account จริง พอมั่นใจแล้วค่อยตั้ง `DRY_RUN = False`

**ลำดับแนะนำก่อน live paper:**
1. `DRY_RUN=True` → ดู log ว่า entry/exit logic ถูกต้อง
2. `DRY_RUN=False` + เงินน้อย → ดูว่า order ส่งเข้า Alpaca ถูก
3. รัน paper เต็มเดือน → ดู regime filter + self-improving edge

---

# 🔬 Unit Consistency Pass (v4)

แก้ unit mismatch per-share vs per-contract ทั้งระบบ (bug สำคัญที่สุด) + harden order lifecycle

## Unit Convention (บังคับใช้ทั้ง codebase)

| ประเภท | หน่วย | ตัวอย่าง |
|--------|-------|---------|
| ราคา option (quote, stop trigger, limit, TP) | **per-share** $ | $1.50, $4.50, $0.05 |
| premium / risk / max-loss / PnL | **per-contract** $ (= per-share × 100) | $150, $300, $5700 |

## Fixes

| Item | Fix |
|------|-----|
| `calculate_max_loss()` คูณ ×100 ซ้ำ | premium เป็น per-contract แล้ว → ไม่คูณซ้ำ → 30-wide+$150/$150 = **$5700** |
| stop price เป็น 300 (per-contract) | stop trigger = short_entry + `stop_loss_per_share` ($3.00) = **$4.50/share** |
| TP/stop ใช้หน่วยปน | TP=$0.05/share, stop=per-share, risk=per-contract แยกชัด |
| ไม่รอ fill ก่อนตั้ง TP/stop | `wait_for_fill()` หลัง submit MLEG ก่อนตั้ง stop/TP |
| MLEG rejected/partial/pending | handle ทุก status → ไม่ register trade, ไม่ส่ง stop/TP |
| rollback ไม่ cancel pending | `_rollback_call_spread()`: pending→cancel, filled→close legs |
| `DRY_RUN` hardcoded | อ่านจาก env: `export DRY_RUN=false` |
| comment OCO ผิด | แก้เป็น "synthetic OCO" (monitor sibling) ตรงกับ implementation จริง |

## Tests (`tests/test_units_and_orders.py`)

```bash
python tests/test_units_and_orders.py   # 9/9 passed
```

ครอบคลุม: max_loss=$5700, stop offset=$3.00, not-filled→no orders,
rejected→no trade, put-fail→rollback, DRY_RUN true/false submit behavior
