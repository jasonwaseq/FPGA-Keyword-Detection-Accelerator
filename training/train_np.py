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
        logits = flat @ self.W2.T + self.b2
        cache = (xw, a, r, p, pooled, flat)
        return logits, cache

    # Keyword classes weighted 2x: sharpens keyword-vs-unknown margins the
    # streaming decision layer thresholds on (silence/unknown recall has
    # plenty of slack in the confusion matrix).
    CLASS_W = np.array([1.0, 1.0] + [2.0] * (q.NUM_CLASSES - 2))

    def backward(self, x, y, logits, cache, wd=1e-4):
        ch = q.CONV_OUT_CH
        xw, a, r, p, pooled, flat = cache
        n = x.shape[0]
        e = np.exp(logits - logits.max(axis=1, keepdims=True))
        prob = e / e.sum(axis=1, keepdims=True)
        d = prob
        d[np.arange(n), y] -= 1.0
        w = self.CLASS_W[y]
        d *= w[:, None]
        d /= w.sum()

        gW2 = d.T @ flat + wd * self.W2
        gb2 = d.sum(axis=0)
        dflat = d @ self.W2                              # [N,DENSE_IN]
        dpooled = dflat.reshape(-1, q.POOL_OUT_LEN, ch)
        # max-pool routing
        dp = np.zeros_like(p)
        mx = p.max(axis=2, keepdims=True)
        mask = (p == mx)
        mask = mask / np.maximum(mask.sum(axis=2, keepdims=True), 1)
        dp = mask * dpooled[:, :, None, :]
        dr = dp.reshape(-1, q.CONV_OUT_LEN, ch)
        da = dr * (a > 0)
        gW1 = np.tensordot(da, xw, axes=([0, 1], [0, 1])).reshape(
            ch, q.CONV_K, q.NUM_MFCC) + wd * self.W1
        gb1 = da.sum(axis=(0, 1))
        return [gW1, gb1, gW2, gb2]

    def adam(self, grads, lr):
        self.t += 1
        b1, b2, eps = 0.9, 0.999, 1e-8
        for i, (p_, g) in enumerate(zip(self.params, grads)):
            self.m[i] = b1 * self.m[i] + (1 - b1) * g
            self.v[i] = b2 * self.v[i] + (1 - b2) * g * g
            mh = self.m[i] / (1 - b1 ** self.t)
            vh = self.v[i] / (1 - b2 ** self.t)
            p_ -= lr * mh / (np.sqrt(vh) + eps)

    def accuracy(self, x, y, batch=4096):
        correct = 0
        for i in range(0, len(x), batch):
            logits, _ = self.forward(x[i:i + batch])
            correct += (logits.argmax(axis=1) == y[i:i + batch]).sum()
        return correct / len(x)


# --- quantization -------------------------------------------------------------
def quantize(net, cal_windows):
    s_w1 = float(np.abs(net.W1).max()) / 127.0
    w1q = np.clip(np.round(net.W1 / s_w1), -128, 127).astype(int)
    b1q = np.round(net.b1 / s_w1).astype(int)

    conv_w = w1q.transpose(0, 1, 2).tolist()             # [oc][k][ic]
    conv_b = b1q.tolist()

    max_conv = 1
    for w in cal_windows:
        _, acc = q.conv1d_int(w, conv_w, conv_b, q.M_MIN, 20, relu=True)
        max_conv = max(max_conv, acc)
    m1, sh1 = q._pick_m_s(max_conv, target=110)
    k1 = (m1 / (1 << sh1)) / s_w1

    s_w2 = float(np.abs(net.W2).max()) / 127.0
    w2q = np.clip(np.round(net.W2 / s_w2), -128, 127).astype(int)
    b2q = np.round(net.b2 * k1 / s_w2).astype(int)

    dense_w = w2q.tolist()
    dense_b = b2q.tolist()

    # Dense requant scale is calibrated against the maximum POSITIVE
    # accumulator, not max|acc|: cross-entropy training drives the silence
    # logit deeply negative while keeping positive-class logits small, and
    # scaling by the absolute max would compress the decision margins into
    # quantization noise. Large negatives simply saturate at -128, which is
    # semantically lossless (they are never the winner and only deflate the
    # smoothed averages of losing classes).
    max_dense = 1     # 24-bit accumulator bound check (absolute)
    max_pos = 1       # requant scaling target (positive side)
    for w in cal_windows:
        act, _ = q.conv1d_int(w, conv_w, conv_b, m1, sh1, relu=True)
        pooled = q.maxpool_int(act)
        logits_acc = []
        for cls in range(q.NUM_CLASSES):
            acc = dense_b[cls]
            flat = [pooled[t][c] for t in range(q.POOL_OUT_LEN)
                    for c in range(q.CONV_OUT_CH)]
            for i in range(q.DENSE_IN):
                acc += flat[i] * dense_w[cls][i]
            logits_acc.append(acc)
        max_dense = max(max_dense, max(abs(a) for a in logits_acc))
        max_pos = max(max_pos, max(logits_acc))
    m2, sh2 = q._pick_m_s(max_pos, target=110)

    for name, mx in (("conv", max_conv), ("dense", max_dense)):
        if mx >= q.ACC_LIMIT:
            raise ValueError(f"{name} accumulator {mx} exceeds 24-bit bound")
    print(f"dense calibration: max|acc|={max_dense} max_pos={max_pos}")

    return {
        "conv_w": conv_w, "conv_b": conv_b, "m_conv": m1, "s_conv": sh1,
        "dense_w": dense_w, "dense_b": dense_b,
        "m_dense": m2, "s_dense": sh2,
        "max_acc_conv": max_conv, "max_acc_dense": max_dense,
    }


def int8_eval(model, x, y, limit=None):
    """Bit-exact INT8 accuracy. limit=None evaluates the full set (gate metric)."""
    idx = np.arange(len(x))
    if limit is not None and len(idx) > limit:
        idx = idx[:: len(idx) // limit + 1]
    conf = np.zeros((q.NUM_CLASSES, q.NUM_CLASSES), dtype=int)
    for i in idx:
        logits = q.infer_int(x[i].tolist(), model)
        conf[y[i], int(np.argmax(logits))] += 1
    acc = np.trace(conf) / conf.sum()
    return acc, conf


def cosine_lr(base_lr, epoch, epochs, min_lr=1e-5):
    """Cosine decay from base_lr -> min_lr over epochs."""
    import math
    t = epoch / max(epochs - 1, 1)
    return min_lr + 0.5 * (base_lr - min_lr) * (1.0 + math.cos(math.pi * t))


# --- self-test stream ----------------------------------------------------------
def pick_selftest(model, full_x, full_y):
    """Choose a held-out 'yes' stream that fires under both bench patterns."""
    yes = np.where(full_y == 2)[0]
    for i in yes:
        stream = full_x[i][4:4 + q.SELFTEST_FRAMES].tolist()
        ok = True
        for corrupt in ((), (12, 55)):
            ev = q.stream_events(stream, model, corrupt=corrupt)
            if not ev or any(e["cls"] != 2 for e in ev):
                ok = False
                break
        if ok:
            return stream, int(i)
    raise RuntimeError("no single-clip self-test stream fires; "
                       "inspect detection thresholds")


def stream_metrics(model, full_x, full_y, per_class=120):
    """Detection statistics over held-out full-clip streams."""
    out = {}
    for cls in range(q.NUM_CLASSES):
        idx = np.where(full_y == cls)[0][:per_class]
        fired = {c: 0 for c in range(q.NUM_CLASSES)}
        none = 0
        for i in idx:
            ev = q.stream_events(full_x[i][4:4 + q.SELFTEST_FRAMES].tolist(),
                                 model)
            if not ev:
                none += 1
            else:
                for e in ev:
                    fired[e["cls"]] += 1
        out[cls] = (len(idx), fired, none)
    return out


# --- main -----------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--work", required=True, help="featurizer output dir")
    ap.add_argument("--weights-out", default="weights")
    ap.add_argument("--epochs", type=int, default=100)
    ap.add_argument("--batch", type=int, default=512)
    ap.add_argument("--lr", type=float, default=2e-3)
    ap.add_argument("--seed", type=int, default=1)
    args = ap.parse_args()
    work = os.path.expanduser(args.work)
    rng = np.random.default_rng(args.seed)

    xt, yt = load_kwsf(os.path.join(work, "train.bin"))
    xv, yv = load_kwsf(os.path.join(work, "val.bin"))
    xe, ye = load_kwsf(os.path.join(work, "test.bin"))
    print(f"geometry: CONV_OUT_CH={q.CONV_OUT_CH} DENSE_IN={q.DENSE_IN} "
          f"NUM_CLASSES={q.NUM_CLASSES}")
    print(f"train {len(xt)}  val {len(xv)}  test {len(xe)}")

    xtf = xt.astype(np.float64)
    xvf = xv.astype(np.float64)
    xef = xe.astype(np.float64)
    net = Net(rng)
    best_val, best = 0.0, None
    for ep in range(args.epochs):
        order = rng.permutation(len(xtf))
        lr = cosine_lr(args.lr, ep, args.epochs)
        for i in range(0, len(order), args.batch):
            b = order[i:i + args.batch]
            logits, cache = net.forward(xtf[b])
            grads = net.backward(xtf[b], yt[b], logits, cache)
            net.adam(grads, lr)
        va = net.accuracy(xvf, yv)
        print(f"epoch {ep + 1:3d}/{args.epochs}  lr {lr:.5f}  val_acc {va:.4f}")
        if va > best_val:
            best_val = va
            best = [p.copy() for p in net.params]
    net.W1, net.b1, net.W2, net.b2 = best
    net.params = [net.W1, net.b1, net.W2, net.b2]

    test_acc = net.accuracy(xef, ye)
    print(f"float: best val {best_val:.4f}  test {test_acc:.4f}")

    # --- quantize on real calibration windows -------------------------------
    cal_idx = rng.permutation(len(xt))[:500]
    cal = [xt[i].tolist() for i in cal_idx]
    model = quantize(net, cal)
    print(f"requant: conv M={model['m_conv']} S={model['s_conv']}, "
          f"dense M={model['m_dense']} S={model['s_dense']}")

    int_acc, conf = int8_eval(model, xe, ye)   # full test set (92% gate)
    print(f"int8 (bit-exact) test accuracy: {int_acc:.4f}")
    print(f"confusion (rows=truth {'/'.join(LABELS)}):")
    print(conf)

    # --- persist the quantized model for tuning + emission --------------------
    # (training is decoupled from smoothing-threshold selection: see
    #  tune_detect.py for the operating-point sweep and emit_weights.py for
    #  the final artifact emission)
    import json
    model_out = dict(model)
    model_out["float_test_acc"] = float(test_acc)
    model_out["int8_test_acc"] = float(int_acc)
    model_out["seed"] = args.seed
    with open(os.path.join(work, "model_int8.json"), "w") as f:
        json.dump(model_out, f)
    np.savez(os.path.join(work, "float_params.npz"),
             W1=net.W1, b1=net.b1, W2=net.W2, b2=net.b2)
    print(f"saved quantized model -> {work}/model_int8.json (+ float_params.npz)")
    print("next: python3 training/tune_detect.py --work", args.work)


if __name__ == "__main__":
    main()
