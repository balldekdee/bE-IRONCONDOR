"""
core/database.py
================
Supabase Persistence Layer
--------------------------
เก็บ:
  - trades:          ทุก Iron Condor ที่เปิด/ปิด พร้อม regime ตอนเข้า
  - regime_snapshots: posterior regime ทุก bar (สำหรับ analysis/backtest)
  - regime_posteriors: self-improving state (เรียนรู้ข้ามวัน)
  - model_state:     serialized ensemble state (resume ข้ามวัน)

ถ้าไม่ตั้ง Supabase credentials → fallback เป็น local mode (ข้าม DB ไม่ crash)
"""

from __future__ import annotations
import json
import logging
from datetime import datetime, timezone
from typing import Optional

logger = logging.getLogger("0dte_bot")

try:
    from supabase import create_client, Client
    _SUPABASE_AVAILABLE = True
except ImportError:
    _SUPABASE_AVAILABLE = False


class Database:
    def __init__(self, url: str, key: str):
        self.enabled = False
        self.client: Optional["Client"] = None

        if not _SUPABASE_AVAILABLE:
            logger.warning("⚠️ supabase package not installed — DB disabled")
            return
        if not url or not key or url.startswith("YOUR_"):
            logger.warning("⚠️ Supabase credentials not set — running in LOCAL mode (no DB)")
            return

        try:
            self.client = create_client(url, key)
            self.enabled = True
            logger.info("✅ Supabase connected")
        except Exception as e:
            logger.error(f"❌ Supabase connection failed: {e} — LOCAL mode")

    # ── Trades ───────────────────────────────────────────────────────────────

    def insert_trade(self, trade: dict):
        """บันทึก trade ตอนเปิด"""
        if not self.enabled:
            return
        try:
            self.client.table("trades").insert({
                "trade_id":          trade["trade_id"],
                "opened_at":         datetime.now(timezone.utc).isoformat(),
                "underlying_price":  trade.get("underlying_price"),
                "call_short_strike": trade.get("call_short_strike"),
                "call_long_strike":  trade.get("call_long_strike"),
                "put_short_strike":  trade.get("put_short_strike"),
                "put_long_strike":   trade.get("put_long_strike"),
                "call_premium":      trade.get("call_premium"),
                "put_premium":       trade.get("put_premium"),
                "total_premium":     trade.get("total_premium"),
                "stop_loss_value":   trade.get("stop_loss_value"),
                "expiry":            trade.get("expiry"),
                "regime_at_entry":   trade.get("regime_at_entry"),
                "regime_probs":      json.dumps(trade.get("regime_probs", [])),
                "regime_confidence": trade.get("regime_confidence"),
                "expected_edge":     trade.get("expected_edge"),
                "status":            "open",
            }).execute()
        except Exception as e:
            logger.error(f"❌ insert_trade failed: {e}")

    def close_trade(self, trade_id: str, pnl: float, outcome: str):
        """อัปเดต trade ตอนปิด"""
        if not self.enabled:
            return
        try:
            self.client.table("trades").update({
                "closed_at": datetime.now(timezone.utc).isoformat(),
                "pnl":       pnl,
                "outcome":   outcome,
                "status":    "closed",
            }).eq("trade_id", trade_id).execute()
        except Exception as e:
            logger.error(f"❌ close_trade failed: {e}")

    # ── Regime snapshots ─────────────────────────────────────────────────────

    def insert_regime_snapshot(self, regime_state, underlying_price: float):
        if not self.enabled:
            return
        try:
            self.client.table("regime_snapshots").insert({
                "ts":                  datetime.now(timezone.utc).isoformat(),
                "underlying_price":    underlying_price,
                "map_regime":          regime_state.map_regime,
                "regime_name":         regime_state.regime_name,
                "prob_risk_on":        float(regime_state.regime_probs[0]),
                "prob_transition":     float(regime_state.regime_probs[1]),
                "prob_crisis":         float(regime_state.regime_probs[2]),
                "change_point_prob":   regime_state.change_point_prob,
                "confidence":          regime_state.confidence,
                "latent_vol":          regime_state.latent_vol,
                "expected_run_length": regime_state.expected_run_length,
            }).execute()
        except Exception as e:
            logger.error(f"❌ insert_regime_snapshot failed: {e}")

    # ── Self-improving posteriors ────────────────────────────────────────────

    def save_regime_posteriors(self, posterior_dict: dict):
        """upsert self-improving state"""
        if not self.enabled:
            return
        try:
            self.client.table("regime_posteriors").upsert({
                "id": 1,
                "updated_at": datetime.now(timezone.utc).isoformat(),
                "state": json.dumps(posterior_dict),
            }).execute()
        except Exception as e:
            logger.error(f"❌ save_regime_posteriors failed: {e}")

    def load_regime_posteriors(self) -> Optional[dict]:
        if not self.enabled:
            return None
        try:
            res = self.client.table("regime_posteriors").select("state").eq("id", 1).execute()
            if res.data:
                return json.loads(res.data[0]["state"])
        except Exception as e:
            logger.warning(f"⚠️ load_regime_posteriors: {e}")
        return None

    # ── Ensemble model state ─────────────────────────────────────────────────

    def save_model_state(self, state: dict):
        if not self.enabled:
            return
        try:
            self.client.table("model_state").upsert({
                "id": 1,
                "updated_at": datetime.now(timezone.utc).isoformat(),
                "state": json.dumps(state),
            }).execute()
        except Exception as e:
            logger.error(f"❌ save_model_state failed: {e}")

    def load_model_state(self) -> Optional[dict]:
        if not self.enabled:
            return None
        try:
            res = self.client.table("model_state").select("state").eq("id", 1).execute()
            if res.data:
                return json.loads(res.data[0]["state"])
        except Exception as e:
            logger.warning(f"⚠️ load_model_state: {e}")
        return None
