"""
core/regime/encoder.py
======================
Deep Sequential Encoder (GRU) — Learned Representation Layer
-----------------------------------------------------------
เรียนรู้ representation จาก sequence ของ features ที่จับ pattern
ระยะยาว/non-linear ที่ statistical models จับไม่ได้

สถาปัตยกรรม:
  features[t-L:t]  →  GRU  →  embedding h_t  →  2 heads:
    (a) vol-prediction head  : ทำนาย realized_vol step ถัดไป (self-supervised)
    (b) regime-logit head    : ทำนาย regime posterior (distilled จาก HSMM)

การเทรน (self-improving):
  - self-supervised: ทำนาย next-bar vol → ไม่ต้องมี label
  - knowledge distillation: ใช้ HSMM posterior เป็น soft target
  → encoder ค่อยๆ เก่งขึ้นเมื่อ data สะสมมากขึ้น โดยไม่ต้อง label มือ

output regime_logits ถูก blend เข้า ensemble (ดู ensemble.py)
รันได้แม้ยังไม่เทรน (จะให้ near-uniform prior จนกว่าจะ fit)
"""

from __future__ import annotations
import numpy as np
from collections import deque
from typing import Optional

import torch
import torch.nn as nn

from core.regime.features import N_FEATURES
from core.regime.hsmm import N_REGIMES


class SequentialRegimeEncoder(nn.Module):
    def __init__(self, n_features: int = N_FEATURES,
                 hidden: int = 32, n_regimes: int = N_REGIMES,
                 n_layers: int = 1):
        super().__init__()
        self.gru = nn.GRU(n_features, hidden, n_layers, batch_first=True)
        self.norm = nn.LayerNorm(hidden)
        # head (a): predict next-step vol (self-supervised)
        self.vol_head = nn.Sequential(nn.Linear(hidden, 16), nn.ReLU(), nn.Linear(16, 1))
        # head (b): regime logits (distilled from HSMM)
        self.regime_head = nn.Sequential(nn.Linear(hidden, 16), nn.ReLU(), nn.Linear(16, n_regimes))

    def forward(self, x):
        # x: [B, L, F]
        out, h = self.gru(x)
        emb = self.norm(out[:, -1, :])      # last timestep embedding
        return self.vol_head(emb), self.regime_head(emb), emb


class EncoderManager:
    """
    จัดการ inference + online training ของ encoder
    เก็บ rolling buffer ของ (sequence, next_vol, hsmm_target) สำหรับ replay training
    """
    def __init__(self, seq_len: int = 32, device: str = "cpu",
                 train_every: int = 64, buffer_size: int = 2000):
        self.seq_len = seq_len
        self.device = device
        self.model = SequentialRegimeEncoder().to(device)
        self.opt = torch.optim.Adam(self.model.parameters(), lr=1e-3)

        self._seq_buf: deque = deque(maxlen=seq_len)        # rolling feature window
        # replay buffer: (seq[L,F], next_vol, hsmm_post[R])
        self._replay: deque = deque(maxlen=buffer_size)
        self.train_every = train_every
        self._step = 0
        self._trained_batches = 0
        self.is_warmed = False

    def push_and_infer(self, feat_norm: np.ndarray,
                       next_vol_target: Optional[float],
                       hsmm_post: np.ndarray) -> dict:
        """
        ป้อน normalized feature 1 step
        next_vol_target: realized_vol ของ bar นี้ (เป็น target ของ "ก่อนหน้า")
        hsmm_post: posterior จาก HSMM (เป็น distillation target)
        คืน encoder regime probs + embedding
        """
        self._seq_buf.append(feat_norm.astype(np.float32))
        self._step += 1

        if len(self._seq_buf) < self.seq_len:
            # warmup ยังไม่พอ → คืน uniform
            return {"encoder_probs": np.ones(N_REGIMES) / N_REGIMES,
                    "embedding": None, "ready": False}

        seq = np.stack(self._seq_buf)                       # [L, F]

        # เก็บ training example (sequence → ทำนาย vol + distill regime)
        if next_vol_target is not None:
            self._replay.append((seq.copy(), float(next_vol_target), hsmm_post.copy()))

        # inference
        self.model.eval()
        with torch.no_grad():
            x = torch.from_numpy(seq).unsqueeze(0).to(self.device)  # [1,L,F]
            _, logits, emb = self.model(x)
            probs = torch.softmax(logits, dim=-1).cpu().numpy()[0]

        # online training trigger
        if len(self._replay) >= max(64, self.train_every) and self._step % self.train_every == 0:
            self._train_step()

        return {"encoder_probs": probs,
                "embedding": emb.cpu().numpy()[0],
                "ready": self.is_warmed}

    def _train_step(self, batch_size: int = 32, epochs: int = 1):
        """
        Mini-batch training:
          loss = MSE(vol_pred, next_vol) + KL(encoder_probs || hsmm_post)
        """
        self.model.train()
        replay = list(self._replay)
        for _ in range(epochs):
            idx = np.random.choice(len(replay), size=min(batch_size, len(replay)), replace=False)
            seqs   = torch.from_numpy(np.stack([replay[i][0] for i in idx])).to(self.device)
            vols   = torch.tensor([replay[i][1] for i in idx], dtype=torch.float32, device=self.device).unsqueeze(1)
            tgts   = torch.from_numpy(np.stack([replay[i][2] for i in idx]).astype(np.float32)).to(self.device)

            vol_pred, logits, _ = self.model(seqs)
            log_probs = torch.log_softmax(logits, dim=-1)

            vol_loss = nn.functional.mse_loss(vol_pred, vols)
            # distillation: KL(hsmm || encoder)
            kl_loss = nn.functional.kl_div(log_probs, tgts, reduction="batchmean")
            loss = vol_loss + 0.5 * kl_loss

            self.opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
            self.opt.step()

        self._trained_batches += 1
        if self._trained_batches >= 10:
            self.is_warmed = True

    def save(self, path: str):
        torch.save({"model": self.model.state_dict(),
                    "opt": self.opt.state_dict(),
                    "trained_batches": self._trained_batches,
                    "is_warmed": self.is_warmed}, path)

    def load(self, path: str):
        ckpt = torch.load(path, map_location=self.device)
        self.model.load_state_dict(ckpt["model"])
        self.opt.load_state_dict(ckpt["opt"])
        self._trained_batches = ckpt.get("trained_batches", 0)
        self.is_warmed = ckpt.get("is_warmed", False)
