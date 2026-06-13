"""
core/risk_manager.py
====================
ควบคุมความเสี่ยงรายวัน:
- Max Daily Risk (1-2% ของพอร์ต)
- Buying Power Limit (50%)
- Total Exposure ณ ปัจจุบัน
- หยุดเปิดสัญญาใหม่ถ้าความเสี่ยงสะสมสูงเกิน
"""

from __future__ import annotations
import logging
from dataclasses import dataclass, field
from typing import List
from config import (
    MAX_DAILY_RISK_PCT, MAX_BUYING_POWER_PCT, MAX_TRADES_PER_DAY
)
from core.options_engine import IronCondor, calculate_max_loss

logger = logging.getLogger("0dte_bot")


@dataclass
class DailyRiskTracker:
    portfolio_value: float
    realized_pnl: float = 0.0
    open_trades: List[IronCondor] = field(default_factory=list)
    trades_opened_today: int = 0
    stop_losses_hit_today: int = 0

    @property
    def max_daily_loss_allowed(self) -> float:
        return self.portfolio_value * MAX_DAILY_RISK_PCT

    @property
    def current_open_risk(self) -> float:
        """ความเสี่ยงของทุกสัญญาที่ยังเปิดอยู่ รวมกัน"""
        return sum(calculate_max_loss(ic) for ic in self.open_trades if ic.status == "open")

    @property
    def total_risk_today(self) -> float:
        """ความเสี่ยงรวมวันนี้ = ขาดทุนแล้ว + risk ที่เปิดค้างอยู่"""
        return abs(min(self.realized_pnl, 0)) + self.current_open_risk

    def can_open_new_trade(self, new_ic: IronCondor, buying_power: float) -> tuple[bool, str]:
        """
        ตรวจสอบว่าสามารถเปิดสัญญาใหม่ได้ไหม
        คืน (allowed: bool, reason: str)
        """
        # ตรวจ 1: จำนวนสัญญาต่อวัน
        if self.trades_opened_today >= MAX_TRADES_PER_DAY:
            return False, f"Daily trade limit reached ({MAX_TRADES_PER_DAY})"

        # ตรวจ 2: Max daily loss ถึง limit แล้วไหม
        if abs(min(self.realized_pnl, 0)) >= self.max_daily_loss_allowed:
            return False, f"Max daily loss hit: ${abs(self.realized_pnl):.0f} >= ${self.max_daily_loss_allowed:.0f}"

        # ตรวจ 3: ความเสี่ยงรวมหลังเพิ่มสัญญาใหม่เกิน limit ไหม
        projected_risk = self.total_risk_today + calculate_max_loss(new_ic)
        if projected_risk > self.max_daily_loss_allowed:
            return False, f"Adding new trade would exceed daily risk limit (${projected_risk:.0f} > ${self.max_daily_loss_allowed:.0f})"

        # ตรวจ 4: Buying Power limit
        max_bp = self.portfolio_value * MAX_BUYING_POWER_PCT
        if buying_power < max_bp * 0.1:   # เหลือ BP น้อยกว่า 10% ของ limit แล้ว
            return False, f"Buying power too low: ${buying_power:.0f}"

        return True, "OK"

    def register_trade_open(self, ic: IronCondor):
        self.open_trades.append(ic)
        self.trades_opened_today += 1
        logger.info(
            f"📋 Risk Tracker | Trades today: {self.trades_opened_today} | "
            f"Open risk: ${self.current_open_risk:.0f} | "
            f"Daily limit: ${self.max_daily_loss_allowed:.0f}"
        )

    def register_trade_close(self, trade_id: str, realized_pnl: float):
        for ic in self.open_trades:
            if ic.trade_id == trade_id:
                ic.status = "closed"
                ic.pnl = realized_pnl
                break
        self.realized_pnl += realized_pnl
        logger.info(
            f"💰 Trade {trade_id} closed | PnL: ${realized_pnl:.0f} | "
            f"Day PnL: ${self.realized_pnl:.0f}"
        )

    def register_stop_hit(self, trade_id: str, loss: float):
        self.stop_losses_hit_today += 1
        self.register_trade_close(trade_id, -abs(loss))
        logger.warning(
            f"🛑 Stop Loss hit #{self.stop_losses_hit_today} today | "
            f"Loss: ${loss:.0f}"
        )

    def should_pause_trading(self) -> tuple[bool, str]:
        """
        ตามคู่มือ: ถ้ามี trade ใกล้ stop loss หลายตัว ให้ระงับการเปิดใหม่
        """
        threatened = [
            ic for ic in self.open_trades
            if ic.status == "open"
        ]
        if self.stop_losses_hit_today >= 2:
            return True, f"2+ stops hit today ({self.stop_losses_hit_today})"

        if self.total_risk_today >= self.max_daily_loss_allowed * 0.8:
            return True, f"Risk at 80% of daily limit"

        return False, "OK"

    def summary(self) -> dict:
        return {
            "portfolio_value":      self.portfolio_value,
            "trades_opened":        self.trades_opened_today,
            "stops_hit":            self.stop_losses_hit_today,
            "realized_pnl":         round(self.realized_pnl, 2),
            "current_open_risk":    round(self.current_open_risk, 2),
            "total_risk_today":     round(self.total_risk_today, 2),
            "max_daily_loss":       round(self.max_daily_loss_allowed, 2),
            "risk_utilization_pct": round(self.total_risk_today / max(self.max_daily_loss_allowed, 1) * 100, 1),
        }
