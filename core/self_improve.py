"""
core/self_improve.py
====================
Self-Improving Layer — Regime-Conditional Edge Learning
-------------------------------------------------------
หัวใจของ "self-improving": bot เรียนรู้ว่า regime ไหนทำกำไรจริง
แล้วค่อยๆ ปรับ filter ให้เทรดเฉพาะ regime ที่มี edge เชิงบวก

กลไก:
  1. ทุกครั้งที่เทรดปิด → log (regime ตอนเข้า, outcome) ลง posterior
  2. ใช้ Bayesian update (Normal-inverse-Gamma) ต่อ regime
     → ประเมิน E[PnL | regime] พร้อม uncertainty
  3. Trade filter ใช้ Thompson sampling / lower-confidence-bound
     → เทรดเฉพาะเมื่อ regime ปัจจุบันมี expected edge > threshold

ผลลัพธ์: ยิ่งเทรดเยอะ ตัวกรองยิ่งแม่น (มากกว่า rule คงที่)
ข้อมูล posterior ดึงจาก Supabase ตอน start → เรียนรู้ข้ามวันได้
"""

from __future__ import annotations
import numpy as np
from dataclasses import dataclass, field
from typing import Optional
import logging

from core.regime.hsmm import N_REGIMES, REGIME_NAMES

logger = logging.getLogger("0dte_bot")


@dataclass
class RegimePosterior:
    """
    Normal-inverse-Gamma posterior สำหรับ E[PnL] ต่อ regime
    (conjugate prior สำหรับ Gaussian ที่ไม่รู้ทั้ง mean และ variance)
    """
    # hyperparameters
    mu:     float = 0.0      # prior mean PnL
    kappa:  float = 1.0      # ความเชื่อมั่นใน mean
    alpha:  float = 2.0      # shape (variance)
    beta:   float = 50.0     # scale (variance) — prior var ~ beta/alpha
    n:      int = 0          # จำนวน observation
    wins:   int = 0
    total_pnl: float = 0.0

    def update(self, pnl: float):
        """Bayesian update เมื่อมี trade outcome ใหม่"""
        self.n += 1
        self.total_pnl += pnl
        if pnl > 0:
            self.wins += 1
        # Normal-inverse-Gamma sequential update
        kappa_new = self.kappa + 1
        mu_new = (self.kappa * self.mu + pnl) / kappa_new
        alpha_new = self.alpha + 0.5
        beta_new = self.beta + 0.5 * (self.kappa / kappa_new) * (pnl - self.mu) ** 2
        self.mu, self.kappa, self.alpha, self.beta = mu_new, kappa_new, alpha_new, beta_new

    @property
    def expected_pnl(self) -> float:
        return self.mu

    @property
    def pnl_variance(self) -> float:
        return self.beta / max(self.alpha - 1, 0.5)

    @property
    def win_rate(self) -> float:
        return self.wins / self.n if self.n > 0 else 0.0

    def sample_expected_pnl(self) -> float:
        """Thompson sampling: draw จาก posterior ของ mean PnL"""
        # marginal posterior ของ mean = Student-t
        var_of_mean = self.beta / (self.alpha * self.kappa)
        df = 2 * self.alpha
        return self.mu + np.random.standard_t(df) * np.sqrt(var_of_mean)

    def lcb(self, z: float = 1.0) -> float:
        """Lower confidence bound ของ expected PnL (conservative)"""
        var_of_mean = self.beta / (self.alpha * self.kappa)
        return self.mu - z * np.sqrt(var_of_mean)

    def state_dict(self) -> dict:
        return {"mu": self.mu, "kappa": self.kappa, "alpha": self.alpha,
                "beta": self.beta, "n": self.n, "wins": self.wins,
                "total_pnl": self.total_pnl}

    @classmethod
    def from_dict(cls, d: dict) -> "RegimePosterior":
        return cls(**d)


class SelfImprovingFilter:
    """
    ตัวกรองการเทรดที่เรียนรู้เอง
    ผสาน 2 สัญญาณ:
      (a) regime posterior ปัจจุบัน (จาก RegimeEnsemble)
      (b) regime-conditional edge ที่เรียนรู้มา (RegimePosterior)
    """
    def __init__(self,
                 min_regime_confidence: float = 0.45,
                 max_change_point_prob: float = 0.35,
                 min_expected_edge: float = 0.0,
                 exploration: bool = True):
        # posterior ต่อ regime
        self.posteriors = [RegimePosterior() for _ in range(N_REGIMES)]
        self.min_regime_confidence = min_regime_confidence
        self.max_change_point_prob = max_change_point_prob
        self.min_expected_edge = min_expected_edge
        self.exploration = exploration   # Thompson sampling vs LCB

        # โครงสร้างความเชื่อเริ่มต้น (prior knowledge):
        # Risk-on → ดีกับ IC, Crisis → แย่มาก (vol สูง = stop บ่อย)
        self.posteriors[0].mu =  40.0    # Risk-on: prior edge บวก
        self.posteriors[1].mu = -10.0    # Transition: prior edge ลบเล็กน้อย
        self.posteriors[2].mu = -80.0    # Crisis: prior edge ลบหนัก

    def should_trade(self, regime_state) -> tuple[bool, str, dict]:
        """
        ตัดสินใจว่าควรเข้าเทรดใน regime ปัจจุบันไหม
        regime_state: RegimeState จาก ensemble
        คืน (allow, reason, diagnostics)
        """
        probs = regime_state.regime_probs

        # ── Gate 1: confidence ต้องพอ ──
        if regime_state.confidence < self.min_regime_confidence:
            return False, f"Regime confidence ต่ำ ({regime_state.confidence:.2f} < {self.min_regime_confidence})", {}

        # ── Gate 2: ไม่เข้าช่วง change point (regime กำลังเปลี่ยน) ──
        if regime_state.change_point_prob > self.max_change_point_prob:
            return False, f"Change point สูง ({regime_state.change_point_prob:.2f}) — regime กำลังเปลี่ยน", {}

        # ── Gate 3: regime-weighted expected edge ──
        # E[edge] = Σ P(regime) * edge(regime)
        if self.exploration:
            regime_edges = np.array([p.sample_expected_pnl() for p in self.posteriors])
        else:
            regime_edges = np.array([p.lcb(z=0.5) for p in self.posteriors])

        expected_edge = float(np.sum(probs * regime_edges))

        diag = {
            "expected_edge": expected_edge,
            "regime_edges": regime_edges.tolist(),
            "dominant_regime": REGIME_NAMES[regime_state.map_regime],
            "regime_n_samples": [p.n for p in self.posteriors],
        }

        if expected_edge < self.min_expected_edge:
            return False, f"Expected edge ต่ำ (${expected_edge:.1f} < ${self.min_expected_edge})", diag

        # ── Gate 4: หลีกเลี่ยง Crisis regime เด็ดขาด ──
        if regime_state.map_regime == 2 and probs[2] > 0.5:
            return False, "Crisis regime ครอบงำ — งดเทรด IC", diag

        return True, f"✅ Regime OK | Edge=${expected_edge:.1f} | {REGIME_NAMES[regime_state.map_regime]}", diag

    def record_outcome(self, regime_at_entry: int, pnl: float):
        """อัปเดต posterior หลังเทรดปิด (self-improvement step)"""
        self.posteriors[regime_at_entry].update(pnl)
        p = self.posteriors[regime_at_entry]
        logger.info(
            f"🧠 Self-improve | {REGIME_NAMES[regime_at_entry]} | "
            f"n={p.n} | E[PnL]=${p.expected_pnl:.1f} | WR={p.win_rate*100:.0f}%"
        )

    def summary(self) -> dict:
        return {
            REGIME_NAMES[i]: {
                "n": p.n,
                "expected_pnl": round(p.expected_pnl, 1),
                "win_rate": round(p.win_rate * 100, 1),
                "lcb": round(p.lcb(), 1),
            }
            for i, p in enumerate(self.posteriors)
        }

    def state_dict(self) -> dict:
        return {"posteriors": [p.state_dict() for p in self.posteriors]}

    def load_state_dict(self, d: dict):
        self.posteriors = [RegimePosterior.from_dict(pd) for pd in d["posteriors"]]
