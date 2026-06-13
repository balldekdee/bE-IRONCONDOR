"""
tests/test_units_and_orders.py
==============================
Test ครอบคลุม checklist ของ code review:
  - calculate_max_loss → $5700 (30-wide + $150/$150)
  - stop offset per-share → 3.00 (ไม่ใช่ 300)
  - opening MLEG not filled → no TP/stop sent
  - opening rejected → no trade registered
  - put fail after call filled → rollback closes call
  - DRY_RUN=true → ไม่ call submit_order
  - DRY_RUN=false → เปิดทางให้ submit_order

รัน: python -m pytest tests/ -v   (หรือ python tests/test_units_and_orders.py)
"""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import config
from core.options_engine import (
    build_iron_condor_structure, calculate_max_loss, IronCondor,
)


# ── Unit tests: max loss + stop units ─────────────────────────────────────────

def test_max_loss_30wide_150_150():
    """30-wide + $150/$150 premium → max loss $5700 (ไม่คูณ 100 ซ้ำ)"""
    ic = build_iron_condor_structure("T", "2025-06-13", 580.0,
                                     595, 625, 565, 535)  # 30-wide ทั้งสองฝั่ง
    ic.call_premium = 150.0   # per-contract $
    ic.put_premium  = 150.0
    ml = calculate_max_loss(ic)
    assert ml == 5700.0, f"expected 5700, got {ml}"
    print(f"✅ calculate_max_loss = ${ml:.0f}  (30-wide + $150/$150)")


def test_stop_offset_is_per_share_3():
    """stop_loss_value=$300/contract → stop_loss_per_share=$3.00 (ไม่ใช่ 300)"""
    ic = build_iron_condor_structure("T", "2025-06-13", 580.0, 595, 625, 565, 535)
    ic.call_premium = 150.0
    ic.put_premium  = 150.0
    assert ic.stop_loss_value == 300.0, ic.stop_loss_value
    assert ic.stop_loss_per_share == 3.00, ic.stop_loss_per_share
    print(f"✅ stop_loss_value=${ic.stop_loss_value:.0f}/contract → "
          f"per-share offset=${ic.stop_loss_per_share:.2f}")


def test_stop_trigger_uses_entry_plus_offset():
    """stop trigger (per-share) = short entry + 3.00 ; order price ไม่เคยเป็น 300"""
    ic = build_iron_condor_structure("T", "2025-06-13", 580.0, 595, 625, 565, 535)
    ic.call_premium = 150.0
    ic.put_premium  = 150.0
    ic.call_short_entry_per_share = 1.50
    trig = ic.call_stop_trigger()
    assert trig == 4.50, trig                      # 1.50 + 3.00
    assert trig < 10, "stop order price ต้อง per-share (ไม่ใช่ 300)"
    print(f"✅ stop trigger = ${trig:.2f}/share (entry 1.50 + offset 3.00)")


# ── Order flow tests (mock broker) ────────────────────────────────────────────

class MockBroker:
    """Mock ที่นับ submit + ควบคุม fill status"""
    def __init__(self, fill_status="filled", put_submit_ok=True):
        self.submit_calls = []
        self.cancel_calls = []
        self.close_calls  = []
        self._fill_status = fill_status
        self._put_submit_ok = put_submit_ok
        self.quotes = {}

    def get_account(self): return {"portfolio_value":500000,"buying_power":200000,"cash":200000,"equity":500000}
    def get_underlying_price(self): return 580.0
    def get_option_chain(self, expiry):
        chain=[]
        for k in range(540, 620, 5):
            chain.append({"symbol":f"SPY260613C{k*1000:08d}","strike":float(k),"type":"C","bid":1.4,"ask":1.6,"delta":0.12 if k>590 else 0.3,"iv":0.2})
            chain.append({"symbol":f"SPY260613P{k*1000:08d}","strike":float(k),"type":"P","bid":1.4,"ask":1.6,"delta":-0.12 if k<570 else -0.3,"iv":0.2})
        return chain
    def place_credit_spread(self, s, l, net_credit, qty=1):
        from core.broker import parse_occ_symbol
        parsed = parse_occ_symbol(s)
        is_put = parsed is not None and parsed[0] == "P"
        if is_put and not self._put_submit_ok:
            return None
        self.submit_calls.append(("spread", s, l))
        return f"ORD_{s}"
    def wait_for_fill(self, oid, timeout=None, poll=1.0): return self._fill_status
    def order_status(self, oid): return self._fill_status
    def place_stop_limit_on_short(self, s, t, lp, qty=1):
        self.submit_calls.append(("stop_limit", s)); return f"SL_{s}"
    def place_stop_market_backstop(self, s, t, qty=1):
        self.submit_calls.append(("stop_market", s)); return f"SM_{s}"
    def place_take_profit_on_short(self, s, qty=1):
        self.submit_calls.append(("tp", s)); return f"TP_{s}"
    def close_option_leg(self, s, is_short, qty=1):
        self.close_calls.append((s, is_short)); return f"CL_{s}"
    def cancel_order(self, oid): self.cancel_calls.append(oid)
    def is_order_filled(self, oid): return self._fill_status == "filled"
    def get_option_quote(self, s): return self.quotes.get(s, 0.80)


def _make_executor(broker):
    from core.executor import TradeExecutor
    from core.risk_manager import DailyRiskTracker
    from core.self_improve import SelfImprovingFilter
    risk = DailyRiskTracker(portfolio_value=500000)
    filt = SelfImprovingFilter(min_regime_confidence=0.0)  # ปิด gate regime ใน test
    return TradeExecutor(broker, risk, regime_filter=filt, db=None)


class _FakeRegime:
    """regime state ปลอม — ผ่าน filter เสมอ"""
    import numpy as _np
    regime_probs = _np.array([1.0, 0.0, 0.0])
    map_regime = 0
    regime_name = "Risk-on / Low vol"
    change_point_prob = 0.0
    confidence = 0.99


def _bars():
    return [{"o":580,"h":580.3,"l":579.7,"c":580.0,"v":5e6}]*3


def test_filled_path_sends_stops_and_tp():
    """fill สำเร็จ → ต้องส่ง stop + tp ครบ"""
    config.DRY_RUN = True
    config.DEFAULT_WING_WIDTH = 10
    broker = MockBroker(fill_status="filled")
    ex = _make_executor(broker)
    ic = ex.try_open_new_trade(_bars(), regime_state=_FakeRegime())
    assert ic is not None
    kinds = [c[0] for c in broker.submit_calls]
    assert kinds.count("stop_limit") == 2, kinds
    assert kinds.count("stop_market") == 2, kinds
    assert kinds.count("tp") == 2, kinds
    print("✅ filled path → 2 stop-limit + 2 backstop + 2 TP sent")


def test_not_filled_no_tp_or_stops():
    """MLEG ไม่ fill (pending) → ห้ามส่ง TP/stop, ห้าม register trade"""
    config.DRY_RUN = True
    config.DEFAULT_WING_WIDTH = 10
    broker = MockBroker(fill_status="pending")
    ex = _make_executor(broker)
    ic = ex.try_open_new_trade(_bars(), regime_state=_FakeRegime())
    assert ic is None, "ไม่ควร register trade เมื่อ order ไม่ fill"
    kinds = [c[0] for c in broker.submit_calls]
    assert "stop_limit" not in kinds and "tp" not in kinds, kinds
    assert len(ex.active_trades) == 0
    print("✅ not-filled → no TP/stop, no trade registered")


def test_rejected_no_trade_registered():
    """MLEG rejected → ไม่ register trade"""
    config.DRY_RUN = True
    config.DEFAULT_WING_WIDTH = 10
    broker = MockBroker(fill_status="rejected")
    ex = _make_executor(broker)
    ic = ex.try_open_new_trade(_bars(), regime_state=_FakeRegime())
    assert ic is None
    assert len(ex.active_trades) == 0
    print("✅ rejected → no trade registered")


def test_put_fail_rolls_back_call():
    """put spread submit fail หลัง call filled → ต้องปิด call legs"""
    config.DRY_RUN = True
    config.DEFAULT_WING_WIDTH = 10
    broker = MockBroker(fill_status="filled", put_submit_ok=False)
    ex = _make_executor(broker)
    ic = ex.try_open_new_trade(_bars(), regime_state=_FakeRegime())
    assert ic is None
    # call legs ต้องถูกปิด (rollback)
    assert len(broker.close_calls) == 2, broker.close_calls
    sides = sorted([s[1] for s in broker.close_calls])   # [False(long), True(short)]
    assert sides == [False, True], sides
    print("✅ put fail → call spread rolled back (short BUY-close, long SELL-close)")


def test_dry_run_true_no_real_submit():
    """DRY_RUN=True → broker จริงไม่ควรเรียก trading.submit_order"""
    config.DRY_RUN = True
    import core.broker as bmod
    bmod.DRY_RUN = True
    calls = {"submit": 0}
    class TC:
        def submit_order(self, req): calls["submit"] += 1; return type("O",(),{"id":"x"})()
    b = object.__new__(bmod.AlpacaBroker)
    b.trading = TC()
    # place spread / stop / tp ใน DRY_RUN → ต้องไม่แตะ submit_order
    b.place_credit_spread("SPY...C","SPY...C", net_credit=1.0)
    b.place_stop_limit_on_short("SPY...C", 4.5, 4.9)
    b.place_take_profit_on_short("SPY...C")
    assert calls["submit"] == 0, "DRY_RUN ต้องไม่ submit_order"
    print("✅ DRY_RUN=True → submit_order ไม่ถูกเรียก")


def test_dry_run_false_allows_submit():
    """DRY_RUN=False → path เปิดให้ submit_order"""
    import core.broker as bmod
    old = bmod.DRY_RUN
    bmod.DRY_RUN = False
    calls = {"submit": 0}
    class TC:
        def submit_order(self, req):
            calls["submit"] += 1
            return type("O",(),{"id":"x"})()
    b = object.__new__(bmod.AlpacaBroker)
    b.trading = TC()
    b.place_take_profit_on_short("SPY260613C00595000")
    bmod.DRY_RUN = old
    assert calls["submit"] == 1, "DRY_RUN=False ต้องเปิดให้ submit_order"
    print("✅ DRY_RUN=False → submit_order ถูกเรียก")


if __name__ == "__main__":
    import logging; logging.disable(logging.CRITICAL)
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    passed = 0
    for t in tests:
        try:
            t(); passed += 1
        except AssertionError as e:
            print(f"❌ {t.__name__}: {e}")
        except Exception as e:
            print(f"💥 {t.__name__}: {type(e).__name__}: {e}")
    print(f"\n{passed}/{len(tests)} tests passed")
