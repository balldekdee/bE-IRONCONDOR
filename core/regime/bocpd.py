"""
core/regime/bocpd.py
====================
Bayesian Online Change Point Detection (Adams & MacKay, 2007)
-------------------------------------------------------------
ตรวจจับจุดเปลี่ยน regime แบบ real-time โดยคำนวณ posterior ของ
"run length" r_t = จำนวน step นับจาก change point ล่าสุด

Output:
  - change_point_prob: P(r_t = 0) = ความน่าจะเป็นที่เพิ่งเกิด change point
  - expected_run_length: E[r_t] = อายุเฉลี่ยของ regime ปัจจุบัน

ใช้ Gaussian observation model พร้อม conjugate Normal-Gamma prior
(อัปเดต sufficient statistics online — ไม่ต้องเก็บ history ทั้งหมด)

สัญญาณนี้ใช้:
  1. boost ความน่าจะเป็นของ Transition regime ใน ensemble
  2. block การเข้าเทรดช่วงที่ change_point_prob สูง (ตลาดกำลังเปลี่ยน regime)
"""

from __future__ import annotations
import numpy as np
from dataclasses import dataclass, field


@dataclass
class BOCPD:
    """
    BOCPD ด้วย Student-t predictive (Normal-Gamma conjugate prior)
    hazard rate คงที่ = 1/lambda (geometric prior บน run length)
    """
    hazard_lambda: float = 100.0     # คาดว่า regime เปลี่ยนทุก ~100 bars
    max_run_length: int = 300        # ตัด run length distribution ที่ความยาวนี้

    # Normal-Gamma hyperparameters (prior)
    mu0:    float = 0.0
    kappa0: float = 1.0
    alpha0: float = 1.0
    beta0:  float = 1.0

    # run-length posterior P(r_t)
    R: np.ndarray = field(default=None)
    # sufficient stats ต่อ run length
    mu:    np.ndarray = field(default=None)
    kappa: np.ndarray = field(default=None)
    alpha: np.ndarray = field(default=None)
    beta:  np.ndarray = field(default=None)

    _t: int = 0

    def __post_init__(self):
        self.R = np.array([1.0])  # P(r_0 = 0) = 1
        self.mu    = np.array([self.mu0])
        self.kappa = np.array([self.kappa0])
        self.alpha = np.array([self.alpha0])
        self.beta  = np.array([self.beta0])

    def _student_t_pdf(self, x: float) -> np.ndarray:
        """predictive prob ของ x ต่อแต่ละ run length (Student-t)"""
        df = 2 * self.alpha
        scale = np.sqrt(self.beta * (self.kappa + 1) / (self.alpha * self.kappa))
        z = (x - self.mu) / scale
        # log Student-t density
        from scipy.special import gammaln
        log_pdf = (gammaln((df + 1) / 2) - gammaln(df / 2)
                   - 0.5 * np.log(df * np.pi) - np.log(scale)
                   - (df + 1) / 2 * np.log(1 + z ** 2 / df))
        return np.exp(log_pdf)

    def update(self, x: float) -> dict:
        """
        ป้อน observation 1 ตัว (แนะนำใช้ Kalman surprise หรือ realized_vol)
        คืน change point statistics
        """
        self._t += 1
        H = 1.0 / self.hazard_lambda            # hazard (constant)

        pred_prob = self._student_t_pdf(x)      # P(x | r_{t-1})

        # growth: run length ยาวขึ้น (ไม่เกิด change point)
        growth = self.R * pred_prob * (1 - H)
        # change point: run length reset เป็น 0
        cp = np.sum(self.R * pred_prob * H)

        new_R = np.concatenate([[cp], growth])
        new_R /= (new_R.sum() + 1e-12)          # normalize

        # ── อัปเดต sufficient statistics (Normal-Gamma) ──
        new_mu    = np.concatenate([[self.mu0],
                                    (self.kappa * self.mu + x) / (self.kappa + 1)])
        new_kappa = np.concatenate([[self.kappa0], self.kappa + 1])
        new_alpha = np.concatenate([[self.alpha0], self.alpha + 0.5])
        new_beta  = np.concatenate([[self.beta0],
                                    self.beta + (self.kappa * (x - self.mu) ** 2)
                                    / (2 * (self.kappa + 1))])

        # ตัดความยาวไม่ให้โตเกิน max_run_length
        if len(new_R) > self.max_run_length:
            new_R     = new_R[:self.max_run_length]
            new_mu    = new_mu[:self.max_run_length]
            new_kappa = new_kappa[:self.max_run_length]
            new_alpha = new_alpha[:self.max_run_length]
            new_beta  = new_beta[:self.max_run_length]
            new_R /= (new_R.sum() + 1e-12)

        self.R, self.mu, self.kappa, self.alpha, self.beta = \
            new_R, new_mu, new_kappa, new_alpha, new_beta

        run_lengths = np.arange(len(self.R))
        return {
            "change_point_prob":   float(self.R[0]),
            "expected_run_length": float(np.sum(run_lengths * self.R)),
            "map_run_length":      int(np.argmax(self.R)),
        }

    def state_dict(self) -> dict:
        return {k: getattr(self, k).tolist()
                for k in ["R", "mu", "kappa", "alpha", "beta"]} | {"_t": self._t}

    def load_state_dict(self, d: dict):
        for k in ["R", "mu", "kappa", "alpha", "beta"]:
            setattr(self, k, np.array(d[k]))
        self._t = d["_t"]
