"""
core/regime/hsmm.py
===================
Hidden Semi-Markov Model (HSMM) — Regime Backbone
-------------------------------------------------
ต่างจาก HMM ตรงที่ HSMM โมเดล "duration" ของแต่ละ regime อย่างชัดเจน
(regime อยู่ได้นานแบบ variable ไม่ใช่ geometric เหมือน HMM)
→ เหมาะกับตลาดจริงที่ Risk-on อยู่ยาว แต่ Crisis มาเร็วไปเร็ว

3 Regimes (เรียงตาม latent vol จากต่ำ→สูง):
  0: Risk-on / Low vol
  1: Transition / Choppy
  2: Crisis / High vol

Emission: Gaussian บน latent features [latent_vol, |trend|, vol_of_vol]
Duration: Negative-Binomial (พารามิเตอร์ต่างกันต่อ regime)

Online inference: forward filtering ด้วย duration-augmented state
Self-improving: online EM อัปเดต emission means/vars เมื่อ data สะสมพอ
"""

from __future__ import annotations
import numpy as np
from dataclasses import dataclass, field
from typing import Optional


REGIME_NAMES = ["Risk-on / Low vol", "Transition / Choppy", "Crisis / High vol"]
N_REGIMES = 3


@dataclass
class HSMM:
    n_regimes: int = N_REGIMES
    n_emission_dims: int = 3   # [latent_vol, |trend|, vol_of_vol]

    # Emission parameters (Gaussian) — prior ที่สมเหตุผลสำหรับ SPY
    # row = regime, col = emission dim
    emission_means: np.ndarray = field(default=None)
    emission_vars:  np.ndarray = field(default=None)

    # Transition matrix ระหว่าง regime (เมื่อ duration หมด)
    # ห้าม self-transition (duration จัดการแทน) → diagonal = 0
    trans: np.ndarray = field(default=None)

    # Duration model (mean duration ต่อ regime, หน่วย = bars)
    duration_mean: np.ndarray = field(default=None)

    # online posterior belief over regimes
    belief: np.ndarray = field(default=None)

    # online EM accumulators
    _em_count:   np.ndarray = field(default=None)
    _em_sum:     np.ndarray = field(default=None)
    _em_sq_sum:  np.ndarray = field(default=None)
    _n_updates:  int = 0
    em_warmup:   int = 100     # เริ่ม online EM หลังเก็บได้ 100 obs

    def __post_init__(self):
        if self.emission_means is None:
            # latent_vol(annualized), |trend|, vol_of_vol
            self.emission_means = np.array([
                [0.10, 0.0005, 0.02],   # Risk-on: vol ต่ำ, trend นิ่ง
                [0.18, 0.0015, 0.06],   # Transition: vol กลาง, choppy
                [0.35, 0.0040, 0.15],   # Crisis: vol สูงมาก
            ])
        if self.emission_vars is None:
            self.emission_vars = np.array([
                [0.003, 1e-6, 0.001],
                [0.010, 4e-6, 0.004],
                [0.050, 2e-5, 0.020],
            ])
        if self.trans is None:
            # เมื่อ regime จบ duration จะไปไหนต่อ
            self.trans = np.array([
                [0.00, 0.85, 0.15],   # Risk-on → มัก Transition
                [0.55, 0.00, 0.45],   # Transition → กลับ Risk-on หรือไป Crisis
                [0.30, 0.70, 0.00],   # Crisis → มัก Transition ก่อนกลับ
            ])
        if self.duration_mean is None:
            self.duration_mean = np.array([60.0, 15.0, 20.0])  # bars
        if self.belief is None:
            self.belief = np.array([0.6, 0.25, 0.15])  # prior

        self._em_count  = np.zeros(self.n_regimes)
        self._em_sum    = np.zeros((self.n_regimes, self.n_emission_dims))
        self._em_sq_sum = np.zeros((self.n_regimes, self.n_emission_dims))

    def _emission_loglik(self, obs: np.ndarray) -> np.ndarray:
        """log P(obs | regime) ต่อแต่ละ regime (diagonal Gaussian)"""
        diff = obs[None, :] - self.emission_means          # [R, D]
        ll = -0.5 * (np.log(2 * np.pi * self.emission_vars)
                     + diff ** 2 / self.emission_vars)
        return ll.sum(axis=1)                              # [R]

    def update(self, obs: np.ndarray, change_point_prob: float = 0.0) -> dict:
        """
        Online forward filtering 1 step
        obs: [latent_vol, |trend|, vol_of_vol]
        change_point_prob: จาก BOCPD — boost การ transition เมื่อสูง

        คืน posterior belief over 3 regimes
        """
        # ── 1. effective transition (blend self-persist กับ trans-on-changepoint) ──
        # ปกติ regime persist (duration); ถ้า change point → ใช้ trans matrix มากขึ้น
        persist = np.exp(-1.0 / self.duration_mean)        # prob คงอยู่ต่อ bar
        cp = np.clip(change_point_prob * 3.0, 0.0, 1.0)    # amplify CP signal

        # transition operator: T[i,j] = P(regime j ตอนนี้ | regime i เมื่อกี้)
        T = np.zeros((self.n_regimes, self.n_regimes))
        for i in range(self.n_regimes):
            stay = persist[i] * (1 - cp)
            T[i, i] = stay
            leave = 1 - stay
            T[i] += leave * self.trans[i]
            T[i, i] += leave * 0.0  # trans diagonal = 0 already
        # normalize
        T /= T.sum(axis=1, keepdims=True)

        # ── 2. predict ──
        pred = self.belief @ T

        # ── 3. update ด้วย emission likelihood ──
        ll = self._emission_loglik(obs)
        ll -= ll.max()                  # stabilize
        lik = np.exp(ll)
        post = pred * lik
        post /= (post.sum() + 1e-12)
        self.belief = post

        # ── 4. online EM accumulation (soft assignment) ──
        self._accumulate_em(obs, post)

        return {
            "regime_probs": post.copy(),
            "map_regime":   int(np.argmax(post)),
            "regime_name":  REGIME_NAMES[int(np.argmax(post))],
            "entropy":      float(-np.sum(post * np.log(post + 1e-12))),
        }

    def _accumulate_em(self, obs: np.ndarray, resp: np.ndarray):
        """สะสม sufficient stats สำหรับ online EM (responsibility-weighted)"""
        self._em_count  += resp
        self._em_sum    += resp[:, None] * obs[None, :]
        self._em_sq_sum += resp[:, None] * (obs[None, :] ** 2)
        self._n_updates += 1

        # อัปเดต emission params เป็นระยะ (decay ช้าๆ — ไม่ให้ drift แรง)
        if self._n_updates >= self.em_warmup and self._n_updates % 50 == 0:
            self._partial_m_step(lr=0.10)

    def _partial_m_step(self, lr: float = 0.1):
        """
        Partial M-step: ขยับ emission means/vars เข้าหา data จริง
        ด้วย learning rate ต่ำ (online EM / stochastic approximation)
        คงลำดับ regime (sort by latent_vol) เพื่อไม่ให้ label สลับ
        """
        for k in range(self.n_regimes):
            n = self._em_count[k]
            if n < 5:
                continue
            new_mean = self._em_sum[k] / n
            new_var = self._em_sq_sum[k] / n - new_mean ** 2
            new_var = np.clip(new_var, 1e-7, None)
            self.emission_means[k] = (1 - lr) * self.emission_means[k] + lr * new_mean
            self.emission_vars[k]  = (1 - lr) * self.emission_vars[k]  + lr * new_var

        # บังคับ label ordering: regime 0 vol ต่ำสุด, 2 สูงสุด
        order = np.argsort(self.emission_means[:, 0])
        if not np.array_equal(order, np.arange(self.n_regimes)):
            self.emission_means = self.emission_means[order]
            self.emission_vars  = self.emission_vars[order]
            self.belief         = self.belief[order]

        # decay accumulator (forget old data slowly → adaptive)
        self._em_count  *= 0.5
        self._em_sum    *= 0.5
        self._em_sq_sum *= 0.5

    def state_dict(self) -> dict:
        return {
            "emission_means": self.emission_means.tolist(),
            "emission_vars":  self.emission_vars.tolist(),
            "trans":          self.trans.tolist(),
            "duration_mean":  self.duration_mean.tolist(),
            "belief":         self.belief.tolist(),
            "n_updates":      self._n_updates,
        }

    def load_state_dict(self, d: dict):
        self.emission_means = np.array(d["emission_means"])
        self.emission_vars  = np.array(d["emission_vars"])
        self.trans          = np.array(d["trans"])
        self.duration_mean  = np.array(d["duration_mean"])
        self.belief         = np.array(d["belief"])
        self._n_updates     = d["n_updates"]
