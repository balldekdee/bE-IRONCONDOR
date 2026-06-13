"""
core/regime/state_space.py
==========================
Bayesian State-Space Model (Kalman Filter)
------------------------------------------
ติดตาม latent state ที่ขับเคลื่อน regime อย่างต่อเนื่อง:
  - latent log-volatility level (μ_vol)
  - latent trend / drift     (μ_trend)

ทำหน้าที่ "denoise" สัญญาณดิบก่อนป้อนเข้า HSMM
ทำให้ regime emission สะอาดขึ้น ลด false transition จาก noise

State vector:  x = [log_vol, trend]
Observation:   y = [observed_log_vol, observed_return]

ใช้ standard Kalman recursion (linear-Gaussian) — closed-form, เร็วมาก,
เหมาะกับ online streaming
"""

from __future__ import annotations
import numpy as np
from dataclasses import dataclass, field


@dataclass
class KalmanStateSpace:
    """
    2-state Kalman filter
    state x_t = F x_{t-1} + w,   w ~ N(0, Q)
    obs   y_t = H x_t     + v,   v ~ N(0, R)
    """
    # Transition matrix (random-walk + slight mean reversion บน vol)
    F: np.ndarray = field(default_factory=lambda: np.array([[0.98, 0.0],
                                                             [0.0,  0.90]]))
    # Observation matrix (สังเกตทั้ง vol และ trend ตรงๆ)
    H: np.ndarray = field(default_factory=lambda: np.eye(2))
    # Process noise
    Q: np.ndarray = field(default_factory=lambda: np.diag([0.02, 0.01]))
    # Observation noise
    R: np.ndarray = field(default_factory=lambda: np.diag([0.10, 0.05]))

    # Posterior state mean & covariance
    x: np.ndarray = field(default_factory=lambda: np.array([np.log(0.15), 0.0]))
    P: np.ndarray = field(default_factory=lambda: np.eye(2) * 1.0)

    _initialized: bool = False

    def update(self, realized_vol: float, log_return: float) -> dict:
        """
        ป้อน observation 1 step → คืน posterior latent state
        realized_vol: annualized realized vol (>0)
        log_return:   log return ของ bar นั้น
        """
        obs_log_vol = np.log(max(realized_vol, 1e-4))
        y = np.array([obs_log_vol, log_return])

        if not self._initialized:
            self.x = np.array([obs_log_vol, log_return])
            self._initialized = True

        # ── Predict ──
        x_pred = self.F @ self.x
        P_pred = self.F @ self.P @ self.F.T + self.Q

        # ── Update ──
        innovation = y - self.H @ x_pred           # residual
        S = self.H @ P_pred @ self.H.T + self.R    # innovation covariance
        K = P_pred @ self.H.T @ np.linalg.inv(S)   # Kalman gain

        self.x = x_pred + K @ innovation
        self.P = (np.eye(2) - K @ self.H) @ P_pred

        # innovation magnitude = surprise signal (ใช้ feed change-point detector)
        surprise = float(innovation.T @ np.linalg.inv(S) @ innovation)  # Mahalanobis²

        return {
            "latent_log_vol":  float(self.x[0]),
            "latent_vol":      float(np.exp(self.x[0])),
            "latent_trend":    float(self.x[1]),
            "vol_uncertainty": float(np.sqrt(self.P[0, 0])),
            "surprise":        surprise,   # high = สัญญาณ regime shift ที่เป็นไปได้
        }

    def state_dict(self) -> dict:
        return {"x": self.x.tolist(), "P": self.P.tolist(),
                "initialized": self._initialized}

    def load_state_dict(self, d: dict):
        self.x = np.array(d["x"])
        self.P = np.array(d["P"])
        self._initialized = d["initialized"]
