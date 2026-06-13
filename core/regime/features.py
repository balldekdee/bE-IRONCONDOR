"""
core/regime/features.py
=======================
แปลง raw bars → feature vector สำหรับ regime models
ทุก model ใน ensemble ใช้ feature ชุดเดียวกันนี้
"""

from __future__ import annotations
import numpy as np
from dataclasses import dataclass, field
from collections import deque
from typing import Optional


# ลำดับ feature คงที่ (สำคัญ — encoder/HSMM พึ่งพา index นี้)
FEATURE_NAMES = [
    "log_return",        # log return ต่อ bar
    "abs_return",        # |log_return| (proxy ความผันผวน)
    "realized_vol",      # rolling realized volatility (annualized)
    "range_pct",         # (high-low)/close
    "vol_of_vol",        # std ของ realized_vol (ความผันผวนของความผันผวน)
    "return_skew",       # rolling skew ของ returns
    "volume_z",          # z-score ของ volume
    "trend_strength",    # |EMA_fast - EMA_slow| / close
    "autocorr",          # lag-1 autocorrelation ของ returns (mean-reversion vs trending)
]
N_FEATURES = len(FEATURE_NAMES)


@dataclass
class FeatureEngine:
    """
    Stateful feature builder — ป้อน bar ทีละแท่ง คืน feature vector
    เก็บ rolling window ภายในเอง
    """
    vol_window: int = 20
    skew_window: int = 30
    volume_window: int = 30
    ema_fast: int = 12
    ema_slow: int = 26

    _closes:  deque = field(default_factory=lambda: deque(maxlen=200))
    _returns: deque = field(default_factory=lambda: deque(maxlen=200))
    _vols:    deque = field(default_factory=lambda: deque(maxlen=200))
    _volumes: deque = field(default_factory=lambda: deque(maxlen=200))
    _ema_f:   Optional[float] = None
    _ema_s:   Optional[float] = None

    def update(self, bar: dict) -> Optional[np.ndarray]:
        """
        bar: dict with keys o,h,l,c,v
        คืน feature vector (np.ndarray shape [N_FEATURES]) หรือ None ถ้า warmup ยังไม่พอ
        """
        c = float(bar["c"])
        h = float(bar["h"])
        l = float(bar["l"])
        v = float(bar.get("v", 0))

        prev_close = self._closes[-1] if self._closes else c
        log_ret = np.log(c / prev_close) if prev_close > 0 else 0.0

        self._closes.append(c)
        self._returns.append(log_ret)
        self._volumes.append(v)

        # EMA update
        alpha_f = 2 / (self.ema_fast + 1)
        alpha_s = 2 / (self.ema_slow + 1)
        self._ema_f = c if self._ema_f is None else alpha_f * c + (1 - alpha_f) * self._ema_f
        self._ema_s = c if self._ema_s is None else alpha_s * c + (1 - alpha_s) * self._ema_s

        # ต้องมี return อย่างน้อย vol_window แท่งก่อนเริ่มผลิต feature
        if len(self._returns) < self.vol_window:
            return None

        rets = np.array(self._returns)

        # realized vol (annualized; 252 days * 78 5-min bars ≈ 19656)
        recent_rets = rets[-self.vol_window:]
        realized_vol = float(np.std(recent_rets) * np.sqrt(19656))
        self._vols.append(realized_vol)

        # vol of vol
        vov = float(np.std(list(self._vols)[-self.vol_window:])) if len(self._vols) >= 5 else 0.0

        # skew
        skew_rets = rets[-self.skew_window:] if len(rets) >= self.skew_window else rets
        return_skew = float(_safe_skew(skew_rets))

        # volume z-score
        vols = np.array(self._volumes)[-self.volume_window:]
        volume_z = float((v - vols.mean()) / (vols.std() + 1e-9)) if len(vols) >= 5 else 0.0

        # trend strength
        trend = abs(self._ema_f - self._ema_s) / c if c > 0 else 0.0

        # lag-1 autocorr
        autocorr = float(_safe_autocorr(recent_rets))

        feat = np.array([
            log_ret,
            abs(log_ret),
            realized_vol,
            (h - l) / c if c > 0 else 0.0,
            vov,
            return_skew,
            volume_z,
            trend,
            autocorr,
        ], dtype=np.float32)

        return np.nan_to_num(feat, nan=0.0, posinf=0.0, neginf=0.0)

    def warmup_ready(self) -> bool:
        return len(self._returns) >= self.vol_window


def _safe_skew(x: np.ndarray) -> float:
    if len(x) < 3:
        return 0.0
    m = x.mean()
    s = x.std()
    if s < 1e-12:
        return 0.0
    return float(np.mean(((x - m) / s) ** 3))


def _safe_autocorr(x: np.ndarray) -> float:
    if len(x) < 3:
        return 0.0
    x = x - x.mean()
    denom = np.sum(x * x)
    if denom < 1e-12:
        return 0.0
    return float(np.sum(x[:-1] * x[1:]) / denom)


class RunningNormalizer:
    """
    Online z-score normalization (Welford's algorithm)
    ทำให้ feature scale คงที่สำหรับ encoder/HSMM — อัปเดตได้ตลอด (self-improving)
    """
    def __init__(self, n_features: int = N_FEATURES):
        self.n = 0
        self.mean = np.zeros(n_features, dtype=np.float64)
        self.M2 = np.ones(n_features, dtype=np.float64)

    def update(self, x: np.ndarray):
        self.n += 1
        delta = x - self.mean
        self.mean += delta / self.n
        delta2 = x - self.mean
        self.M2 += delta * delta2

    def transform(self, x: np.ndarray) -> np.ndarray:
        if self.n < 2:
            return x
        std = np.sqrt(self.M2 / self.n) + 1e-8
        return ((x - self.mean) / std).astype(np.float32)

    def state_dict(self) -> dict:
        return {"n": self.n, "mean": self.mean.tolist(), "M2": self.M2.tolist()}

    def load_state_dict(self, d: dict):
        self.n = d["n"]
        self.mean = np.array(d["mean"])
        self.M2 = np.array(d["M2"])
