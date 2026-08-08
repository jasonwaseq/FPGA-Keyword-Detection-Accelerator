#!/usr/bin/env python3
"""
train_np.py - Self-contained training + quantization for the KWS accelerator
(NumPy only; no ML framework).

Consumes feature records produced by the C featurizer (host/src/featurize.c),
i.e. INT8 features from the EXACT deployment MFCC front end with frozen
corpus normalization - the model trains on precisely what the hardware sees.

Pipeline:
  1. train the float twin of the hardware (conv1d 40->CONV_OUT_CH k3, ReLU,
     maxpool 2, dense DENSE_IN->NUM_CLASSES, time-major flatten) with Adam
     on cross-entropy;
  2. post-training INT8 quantization with correct cross-layer bias scaling:
       s_w1 = max|W1|/127          W1q = round(W1/s_w1)   b1q = round(b1/s_w1)
       (M1,S1) calibrated on real windows;  k1 = (M1/2^S1)/s_w1
       s_w2 = max|W2|/127          W2q = round(W2/s_w2)   b2q = round(b2*k1/s_w2)
       (M2,S2) calibrated with the conv stage in place
     (the INT8 activation a8 ~= a_float*k1, so scaling b2 by k1 keeps the
      dense pre-activation proportional to the float model's - argmax and
      relative confidences are preserved);
  3. bit-exact INT8 evaluation via model/kws_quant.py (the same arithmetic
     the RTL implements);
  4. self-test stream selection: a held-out 'yes' utterance whose feature
     stream provably fires the full pipeline (verified under the clean and
     the fault-injection patterns used by tb_kws_core / hwtest);
  5. emission of the complete deployment artifact set into weights/.

Usage (see docs/training.md for the full flow):
  python3 training/train_np.py --work ~/kws_data/work --weights-out weights
"""

import argparse
import os
import struct
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                "..", "model"))
import kws_quant as q

LABELS = q.LABELS


# --- data -------------------------------------------------------------------
def load_kwsf(path):
    with open(path, "rb") as f:
        magic = f.read(4)
        assert magic == b"KWSF", f"bad magic in {path}"
        n, frames = struct.unpack("<II", f.read(8))
        rec = 1 + frames * q.NUM_MFCC
        raw = np.frombuffer(f.read(n * rec), dtype=np.uint8).reshape(n, rec)
    y = raw[:, 0].astype(np.int64)
    x = raw[:, 1:].view(np.int8).reshape(n, frames, q.NUM_MFCC)
    return x, y


# --- float model -------------------------------------------------------------
class Net:
    def __init__(self, rng):
        ch = q.CONV_OUT_CH
        k_flat = q.CONV_K * q.NUM_MFCC          # 120
        self.W1 = rng.normal(0, 0.05, (ch, q.CONV_K, q.NUM_MFCC)).astype(np.float64)
        self.b1 = np.zeros(ch)
        self.W2 = rng.normal(0, 0.05, (q.NUM_CLASSES, q.DENSE_IN)).astype(np.float64)
        self.b2 = np.zeros(q.NUM_CLASSES)
        self.params = [self.W1, self.b1, self.W2, self.b2]
        self.m = [np.zeros_like(p) for p in self.params]
        self.v = [np.zeros_like(p) for p in self.params]
        self.t = 0
        self._k_flat = k_flat

    @staticmethod
    def im2col(x):
        # x [N,WINDOW_LEN,NUM_MFCC] -> [N,CONV_OUT_LEN,CONV_K*NUM_MFCC]
        return np.concatenate(
            [x[:, 0:q.CONV_OUT_LEN],
             x[:, 1:q.CONV_OUT_LEN + 1],
             x[:, 2:q.CONV_OUT_LEN + 2]], axis=2)

    def forward(self, x):
        ch = q.CONV_OUT_CH
        xw = self.im2col(x)                              # [N,30,k_flat]
        w1f = self.W1.reshape(ch, self._k_flat)
        a = xw @ w1f.T + self.b1                         # [N,30,ch]
        r = np.maximum(a, 0.0)
        p = r.reshape(-1, q.POOL_OUT_LEN, q.POOL_SIZE, ch)
        pooled = p.max(axis=2)                           # [N,15,ch]
        flat = pooled.reshape(-1, q.DENSE_IN)            # time-major
        logits = flat @