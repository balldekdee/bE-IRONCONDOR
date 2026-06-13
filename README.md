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
