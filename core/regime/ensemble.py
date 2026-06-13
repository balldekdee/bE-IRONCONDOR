"""
core/regime/ensemble.py
=======================
Regime Ensemble — รวม 4 components เป็น posterior เดียว
------------------------------------------------------
  1. Kalman State-Space  → latent vol/trend ที่ denoise แล้ว (feed HSMM)
  2. BOCPD               → change-point hazard (boost Transition + block entry)
  3. HSMM                → regime backbone posterior (น้ำหนักหลัก)
  4. Deep Encoder        → learned posterior (distilled, ค่อยๆ มีน้ำหนักขึ้น)

วิธีรวม: log-opinion pool (weighted geometric mean ของ posteriors)
น้ำหนัก encoder เริ่มต่ำ แล้วเพิ่มเมื่อ encoder warmed up (self-improving)

Output: RegimeState — ป้อนเข้า trade filter โดยตรง
"""

from __future__ import annotations
import numpy as np
from dataclasses import dataclass, field
from typing import Optional

from core.regime.features import FeatureEngine, RunningNormalizer
from core.regime.state_space import KalmanStateSpace
from core.regime.bocpd import BOCPD
from core.regime.hsmm import HSMM, REGIME_NAMES, N_REGIMES
from core.regime.encoder import EncoderManager


@dataclass
class RegimeState:
    """ผลลัพธ์ regime ณ bar ปัจจุบัน"""
    regime_probs:        np.ndarray            # [3] posterior สุดท้าย
    map_regime:          int                   # argmax
    regime_name:         str
    change_point_prob:   float
    expected_run_length: float
    latent_vol:          float
    latent_trend:        float
    confidence:          float                 # 1 - normalized entropy
    encoder_ready:       bool
    raw: dict = field(default_factory=dict)    # debug: posterior แยกแต่ละ component

    def pretty(self) -> str:
        """แสดงผลรูปแบบที่ Peter ขอ"""
        lines = []
        for i, name in enumerate(REGIME_NAMES):
            pct = self.regime_probs[i] * 100
            bar = "█" * int(pct / 4)
            lines.append(f"Regime {i+1}: {name:<22} {pct:5.1f}%  {bar}")
        lines.append(f"\nChange-point prob: {self.change_point_prob*100:4.1f}% | "
                     f"Confidence: {self.confidence*100:4.1f}% | "
                     f"Run length: {self.expected_run_length:.0f} bars")
        return "\n".join(lines)


class RegimeEnsemble:
    def __init__(self, encoder_seq_len: int = 32, device: str = "cpu"):
        self.features   = FeatureEngine()
        self.normalizer = RunningNormalizer()
        self.kalman     = KalmanStateSpace()
        self.bocpd      = BOCPD(hazard_lambda=80.0)
        self.hsmm       = HSMM()
        self.encoder    = EncoderManager(seq_len=encoder_seq_len, device=device)

        # น้ำหนัก log-opinion pool
        self.w_hsmm    = 1.0
        self.w_encoder_max = 0.8   # encoder น้ำหนักสูงสุดเมื่อ warmed up

        self._last_state: Optional[RegimeState] = None
        self._n_bars = 0

    def update(self, bar: dict) -> Optional[RegimeState]:
        """
        ป้อน 1 bar → คืน RegimeState (None ถ้ายัง warmup ไม่พอ)
        """
        feat = self.features.update(bar)
        if feat is None:
            return None

        self._n_bars += 1
        self.normalizer.update(feat)
        feat_norm = self.normalizer.transform(feat)

        # feat index: 0=log_ret, 2=realized_vol, 4=vol_of_vol, 7=trend
        log_ret      = float(feat[0])
        realized_vol = float(feat[2])
        vol_of_vol   = float(feat[4])
        trend        = float(feat[7])

        # ── 1. Kalman: denoise latent state ──
        ks = self.kalman.update(realized_vol, log_ret)

        # ── 2. BOCPD: change point บน Kalman surprise ──
        cp = self.bocpd.update(ks["surprise"])

        # ── 3. HSMM: posterior บน latent features ──
        hsmm_obs = np.array([ks["latent_vol"], abs(ks["latent_trend"]), vol_of_vol])
        hs = self.hsmm.update(hsmm_obs, change_point_prob=cp["change_point_prob"])
        hsmm_post = hs["regime_probs"]

        # ── 4. Encoder: learned posterior + online training ──
        enc = self.encoder.push_and_infer(
            feat_norm, next_vol_target=realized_vol, hsmm_post=hsmm_post
        )
        enc_post = enc["encoder_probs"]

        # ── 5. Log-opinion pool (weighted geometric mean) ──
        w_enc = self.w_encoder_max if enc["ready"] else 0.05
        log_pool = (self.w_hsmm * np.log(hsmm_post + 1e-9)
                    + w_enc * np.log(enc_post + 1e-9))
        final = np.exp(log_pool - log_pool.max())
        final /= final.sum()

        # ── 6. boost Transition regime เมื่อ change point สูง ──
        if cp["change_point_prob"] > 0.3:
            boost = cp["change_point_prob"]
            final[1] += boost * 0.5 * (1 - final[1])
            final /= final.sum()

        entropy = -np.sum(final * np.log(final + 1e-12))
        confidence = 1.0 - entropy / np.log(N_REGIMES)

        state = RegimeState(
            regime_probs=final,
            map_regime=int(np.argmax(final)),
            regime_name=REGIME_NAMES[int(np.argmax(final))],
            change_point_prob=cp["change_point_prob"],
            expected_run_length=cp["expected_run_length"],
            latent_vol=ks["latent_vol"],
            latent_trend=ks["latent_trend"],
            confidence=float(confidence),
            encoder_ready=enc["ready"],
            raw={"hsmm": hsmm_post.tolist(), "encoder": enc_post.tolist(),
                 "kalman_surprise": ks["surprise"]},
        )
        self._last_state = state
        return state

    @property
    def last_state(self) -> Optional[RegimeState]:
        return self._last_state

    # ── Persistence (self-improving state ข้ามวัน) ──
    def state_dict(self) -> dict:
        return {
            "normalizer": self.normalizer.state_dict(),
            "kalman":     self.kalman.state_dict(),
            "bocpd":      self.bocpd.state_dict(),
            "hsmm":       self.hsmm.state_dict(),
            "n_bars":     self._n_bars,
        }

    def load_state_dict(self, d: dict):
        self.normalizer.load_state_dict(d["normalizer"])
        self.kalman.load_state_dict(d["kalman"])
        self.bocpd.load_state_dict(d["bocpd"])
        self.hsmm.load_state_dict(d["hsmm"])
        self._n_bars = d.get("n_bars", 0)
