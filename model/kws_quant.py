#!/usr/bin/env python3
"""
kws_quant.py - Quantization / export library for the KWS accelerator.

Single source of truth for:
  * model dimensions (must match rtl/kws_pkg.sv),
  * the bit-exact INT8 inference arithmetic (mirrors the C reference model
    and the RTL datapath),
  * requantization-parameter calibration,
  * emission of $readmemh .mem files, the C weight header and JSON metadata.

Used by:
  * model/gen_weights.py        - deterministic bring-up weights
  * training/export_weights.py  - export of a trained PyTorch checkpoint

Pure standard library; no numpy/torch dependency.
"""

import json
import os

# ----------------------------------------------------------------------------
# Model dimensions - keep in sync with rtl/kws_pkg.sv (there are assertions on
# both sides: the RTL checks .mem sizes at elaboration, the C reference model
# checks the JSON metadata at startup).
# ----------------------------------------------------------------------------
NUM_MFCC     = 40   # input channels (MFCC coefficients per frame)
WINDOW_LEN   = 32   # frames per inference window
CONV_K       = 3    # temporal kernel size
CONV_OUT_CH  = 8    # convolution output channels
CONV_OUT_LEN = WINDOW_LEN - CONV_K + 1          # 30 (valid convolution)
POOL_SIZE    = 2    # temporal pooling factor
POOL_OUT_LEN = CONV_OUT_LEN // POOL_SIZE        # 15
DENSE_IN     = POOL_OUT_LEN * CONV_OUT_CH       # 120
NUM_CLASSES  = 4    # 0=silence 1=unknown 2..N = keywords

# Requantization multiplier precision: M is a positive integer < 2**15 so the
# RTL multiply is a signed 24x16 product (maps onto one SB_MAC16).
M_MIN, M_MAX = 1 << 14, (1 << 15) - 1
S_MAX        = 31

# RTL accumulators are checked (assertions) to stay within signed 24 bits.
ACC_LIMIT = 1 << 23


# ----------------------------------------------------------------------------
# Bit-exact integer arithmetic (specification of the hardware behaviour)
# ----------------------------------------------------------------------------
def sat8(x: int) -> int:
    """Saturate to signed 8-bit."""
    return max(-128, min(127, x))


def requant(acc: int, m: int, s: int) -> int:
    """Requantize an INT32 accumulator to INT8.

    y = sat8( (acc * M + 2^(S-1)) >> S )   (arithmetic shift, round half up)

    Python's >> on ints is an arithmetic (floor) shift, identical to the RTL
    '>>>' on a signed value and to the C reference implementation.
    """
    assert 1 <= s <= S_MAX and 0 < m <= M_MAX
    return sat8((acc * m + (1 << (s - 1))) >> s)


def conv1d_int(feat, w, b, m, s, relu=True):
    """Reference INT8 valid Conv1D over time.

    feat : [WINDOW_LEN][NUM_MFCC] int8
    w    : [CONV_OUT_CH][CONV_K][NUM_MFCC] int8
    b    : [CONV_OUT_CH] int32
    Returns ([CONV_OUT_LEN][CONV_OUT_CH] int8, max_abs_acc)
    """
    out = [[0] * CONV_OUT_CH for _ in range(CONV_OUT_LEN)]
    max_acc = 0
    for t in range(CONV_OUT_LEN):
        for oc in range(CONV_OUT_CH):
            acc = b[oc]
            for k in range(CONV_K):
                for ic in range(NUM_MFCC):
                    acc += feat[t + k][ic] * w[oc][k][ic]
            max_acc = max(max_acc, abs(acc))
            y = requant(acc, m, s)
            out[t][oc] = max(0, y) if relu else y
    return out, max_acc


def maxpool_int(act):
    """Temporal max pool, factor POOL_SIZE. act: [CONV_OUT_LEN][CONV_OUT_CH]."""
    return [[max(act[p * POOL_SIZE + q][c] for q in range(POOL_SIZE))
             for c in range(CONV_OUT_CH)] for p in range(POOL_OUT_LEN)]


def avgpool_int(act):
    """Temporal average pool (truncating shift, POOL_SIZE must be 2**n)."""
    sh = POOL_SIZE.bit_length() - 1
    assert (1 << sh) == POOL_SIZE
    return [[sum(act[p * POOL_SIZE + q][c] for q in range(POOL_SIZE)) >> sh
             for c in range(CONV_OUT_CH)] for p in range(POOL_OUT_LEN)]


def dense_int(pooled, w, b, m, s):
    """Reference INT8 dense layer.

    pooled : [POOL_OUT_LEN][CONV_OUT_CH] int8 (flattened i = t*CONV_OUT_CH+ch)
    w      : [NUM_CLASSES][DENSE_IN] int8
    Returns ([NUM_CLASSES] int8 logits, max_abs_acc)
    """
    flat = [pooled[t][c] for t in range(POOL_OUT_LEN) for c in range(CONV_OUT_CH)]
    logits, max_acc = [], 0
    for cls in range(NUM_CLASSES):
        acc = b[cls]
        for i in range(DENSE_IN):
            acc += flat[i] * w[cls][i]
        max_acc = max(max_acc, abs(acc))
        logits.append(requant(acc, m, s))
    return logits, max_acc


def infer_int(feat, model, pool_mode="max"):
    """Full reference inference. model is the dict produced by calibrate()."""
    act, _ = conv1d_int(feat, model["conv_w"], model["conv_b"],
                        model["m_conv"], model["s_conv"], relu=True)
    pooled = maxpool_int(act) if pool_mode == "max" else avgpool_int(act)
    logits, _ = dense_int(pooled, model["dense_w"], model["dense_b"],
                          model["m_dense"], model["s_dense"])
    return logits


# ----------------------------------------------------------------------------
# Calibration - choose (M, S) per layer from observed accumulator range
# ----------------------------------------------------------------------------
def _pick_m_s(max_acc: int, target: int):
    """Choose M, S so that max_acc * M / 2^S ~= target, with M in [2^14, 2^15)."""
    assert max_acc > 0
    ratio = target / max_acc
    s = 1
    while s < S_MAX and round(ratio * (1 << s)) < M_MIN:
        s += 1
    m = min(M_MAX, max(1, round(ratio * (1 << s))))
    return m, s


def calibrate(conv_w, conv_b, dense_w, dense_b, rng, n_windows=64,
              windows=None):
    """Derive requant parameters from calibration windows.

    windows: optional list of real INT8 feature windows
             [n][WINDOW_LEN][NUM_MFCC] (trained exports pass real data);
             falls back to rng-generated uniform windows (bring-up weights).
    Returns the complete model dict.
    """
    if windows is None:
        windows = [[[rng.randint(-64, 63) for _ in range(NUM_MFCC)]
                    for _ in range(WINDOW_LEN)] for _ in range(n_windows)]

    # Pass 1: convolution accumulator range (requant params irrelevant here).
    max_conv = 1
    for w_ in windows:
        _, acc = conv1d_int(w_, conv_w, conv_b, M_MIN, 20, relu=True)
        max_conv = max(max_conv, acc)
    m_conv, s_conv = _pick_m_s(max_conv, target=110)

    # Pass 2: dense accumulator range with final conv params in place.
    max_dense = 1
    for w_ in windows:
        act, _ = conv1d_int(w_, conv_w, conv_b, m_conv, s_conv, relu=True)
        _, acc = dense_int(maxpool_int(act), dense_w, dense_b, M_MIN, 20)
        max_dense = max(max_dense, acc)
    m_dense, s_dense = _pick_m_s(max_dense, target=100)

    for name, mx in (("conv", max_conv), ("dense", max_dense)):
        if mx >= ACC_LIMIT:
            raise ValueError(f"{name} accumulator {mx} exceeds RTL 24-bit bound")

    return {
        "conv_w": conv_w, "conv_b": conv_b, "m_conv": m_conv, "s_conv": s_conv,
        "dense_w": dense_w, "dense_b": dense_b,
        "m_dense": m_dense, "s_dense": s_dense,
        "max_acc_conv": max_conv, "max_acc_dense": max_dense,
    }


# ----------------------------------------------------------------------------
# Temporal smoothing simulator (mirror of rtl/temporal_smoothing.sv and
# ref_model.c kws_smooth_step) - used to validate self-test streams and to
# evaluate detection behaviour offline.
# ----------------------------------------------------------------------------
# smoothing defaults - keep synchronized with rtl/kws_pkg.sv and
# host/src/ref_model.c (tuned operating point, see training/tune_detect.py)
SMOOTH_DEPTH = 4
SMOOTH_DEFAULTS = dict(thresh=25, vote_min=2, min_consec=1, debounce=12,
                       target_mask=0x0C, enable=1)


class SmoothSim:
    """Bit-exact Python mirror of the hardware decision layer.

    depth overrides SMOOTH_DEPTH for design-space exploration; the shipped
    configuration must match kws_pkg::SMOOTH_DEPTH / KWS_SMOOTH_DEPTH."""

    def __init__(self, depth=None, **cfg):
        self.depth = depth or SMOOTH_DEPTH
        self.cfg = dict(SMOOTH_DEFAULTS, **cfg)
        self.hist = [[0] * NUM_CLASSES for _ in range(self.depth)]
        self.sums = [0] * NUM_CLASSES
        self.win_hist = [0] * self.depth
        self.head = self.whead = self.fill = 0
        self.consec = 0
        self.last_cand = -1
        self.debounce = 0

    def step(self, logits, winner):
        for c in range(NUM_CLASSES):
            self.sums[c] += logits[c] - self.hist[self.head][c]
            self.hist[self.head][c] = logits[c]
        self.head = (self.head + 1) % self.depth
        self.win_hist[self.whead] = winner
        self.whead = (self.whead + 1) % self.depth
        if self.fill < self.depth:
            self.fill += 1

        sh = self.depth.bit_length() - 1
        avg = [self.sums[c] >> sh for c in range(NUM_CLASSES)]
        sm_idx = avg.index(max(avg))        # lowest-index tie-break
        sm_val = avg[sm_idx]
        votes = sum(1 for d in range(self.fill) if self.win_hist[d] == sm_idx)

        c = self.cfg
        cand = (c["enable"] and ((c["target_mask"] >> sm_idx) & 1)
                and sm_val >= c["thresh"] and votes >= c["vote_min"])
        if not cand:
            run = 0
        elif self.consec != 0 and self.last_cand == sm_idx:
            run = self.consec if self.consec >= 15 else self.consec + 1
        else:
            run = 1
        fire = cand and self.debounce == 0 and run >= c["min_consec"]
        if cand:
            self.last_cand = sm_idx
        if fire:
            self.debounce = c["debounce"]
            self.consec = 0
            return dict(cls=sm_idx, conf=max(0, sm_val), votes=votes)
        self.consec = run
        if self.debounce > 0:
            self.debounce -= 1
        return None


def stream_events(frames, model, corrupt=(), pool_mode="max", **cfg):
    """Simulate the full FPGA pipeline over a frame stream (the scheduler's
    window law included): frames [n][NUM_MFCC] int8, corrupt = indices whose
    frames are dropped (CRC-rejected). Returns list of event dicts with the
    commit index of the newest frame in the detecting window."""
    sim = SmoothSim(**cfg)
    commits = []
    events = []
    for i, fr in enumerate(frames):
        if i in corrupt:
            continue
        commits.append(fr)
        n = len(commits)
        if n < WINDOW_LEN or (n - WINDOW_LEN) % 8 != 0:
            continue
        win = commits[n - WINDOW_LEN:n]
        logits = infer_int(win, model, pool_mode)
        winner = logits.index(max(logits))  # lowest-index tie-break
        evt = sim.step(logits, winner)
        if evt:
            evt["frame_idx"] = i
            events.append(evt)
    return events


# ----------------------------------------------------------------------------
# Emission
# ----------------------------------------------------------------------------
def _hex8(v: int) -> str:
    return f"{v & 0xFF:02x}"


def _hex32(v: int) -> str:
    return f"{v & 0xFFFFFFFF:08x}"


SELFTEST_FRAMES = 90   # fixed stream length consumed by tb_kws_core / hwtest


def emit(model, out_dir, origin: str, feat_stats=None, selftest=None):
    """Write all export products into out_dir.

    origin     : provenance string.
    feat_stats : (mean[NUM_MFCC], std[NUM_MFCC], frozen) MFCC normalization
                 statistics baked into the host front end; None = adaptive
                 defaults (mean 0, std 1, frozen 0).
    selftest   : SELFTEST_FRAMES INT8 frames known to trigger >= 1 keyword
                 event through this model - the weight-agnostic stimulus for
                 the full-system bench and the hardware acceptance test.
                 Verified here under both the clean and the corrupted
                 (frames 12, 55 dropped) streaming patterns.
    """
    os.makedirs(out_dir, exist_ok=True)
    cw, cb = model["conv_w"], model["conv_b"]
    dw, db = model["dense_w"], model["dense_b"]

    if feat_stats is None:
        feat_stats = ([0.0] * NUM_MFCC, [1.0] * NUM_MFCC, 0)
    f_mean, f_std, f_frozen = feat_stats

    if selftest is None:
        raise ValueError("emit() requires a selftest stream (see gen_weights/"
                         "train_np for construction)")
    if len(selftest) != SELFTEST_FRAMES:
        raise ValueError(f"selftest must be {SELFTEST_FRAMES} frames")
    for pattern, corrupt in (("clean", ()), ("corrupted", (12, 55))):
        ev = stream_events(selftest, model, corrupt=corrupt)
        if not ev:
            raise ValueError(f"selftest stream fires no event ({pattern} "
                             "pattern) - pick stronger stimulus")

    # conv_weights.mem : index = (oc*CONV_K + k)*NUM_MFCC + ic
    with open(os.path.join(out_dir, "conv_weights.mem"), "w", newline="\n") as f:
        f.write(f"// conv1d weights int8 [{CONV_OUT_CH}][{CONV_K}][{NUM_MFCC}]"
                f" flattened oc-major ({origin})\n")
        for oc in range(CONV_OUT_CH):
            for k in range(CONV_K):
                for ic in range(NUM_MFCC):
                    f.write(_hex8(cw[oc][k][ic]) + "\n")

    # dense_weights.mem : index = cls*DENSE_IN + (t*CONV_OUT_CH + ch)
    with open(os.path.join(out_dir, "dense_weights.mem"), "w", newline="\n") as f:
        f.write(f"// dense weights int8 [{NUM_CLASSES}][{DENSE_IN}]"
                f" flattened class-major ({origin})\n")
        for cls in range(NUM_CLASSES):
            for i in range(DENSE_IN):
                f.write(_hex8(dw[cls][i]) + "\n")

    # Per-layer parameter files: biases then requant multiplier and shift.
    with open(os.path.join(out_dir, "conv_bias.mem"), "w", newline="\n") as f:
        f.write(f"// conv bias int32 [{CONV_OUT_CH}], then M, S ({origin})\n")
        for v in cb:
            f.write(_hex32(v) + "\n")
        f.write(_hex32(model["m_conv"]) + "\n")
        f.write(_hex32(model["s_conv"]) + "\n")

    with open(os.path.join(out_dir, "dense_bias.mem"), "w", newline="\n") as f:
        f.write(f"// dense bias int32 [{NUM_CLASSES}], then M, S ({origin})\n")
        for v in db:
            f.write(_hex32(v) + "\n")
        f.write(_hex32(model["m_dense"]) + "\n")
        f.write(_hex32(model["s_dense"]) + "\n")

    # C header consumed by the reference model (host + Verilator benches).
    with open(os.path.join(out_dir, "kws_weights.h"), "w", newline="\n") as f:
        f.write("/* Auto-generated by model/kws_quant.py - do not edit.\n"
                f" * Origin: {origin}\n */\n"
                "#ifndef KWS_WEIGHTS_H\n#define KWS_WEIGHTS_H\n\n"
                "#include <stdint.h>\n\n")
        f.write(f"#define KWS_NUM_MFCC     {NUM_MFCC}\n")
        f.write(f"#define KWS_WINDOW_LEN   {WINDOW_LEN}\n")
        f.write(f"#define KWS_CONV_K       {CONV_K}\n")
        f.write(f"#define KWS_CONV_OUT_CH  {CONV_OUT_CH}\n")
        f.write(f"#define KWS_CONV_OUT_LEN {CONV_OUT_LEN}\n")
        f.write(f"#define KWS_POOL_SIZE    {POOL_SIZE}\n")
        f.write(f"#define KWS_POOL_OUT_LEN {POOL_OUT_LEN}\n")
        f.write(f"#define KWS_DENSE_IN     {DENSE_IN}\n")
        f.write(f"#define KWS_NUM_CLASSES  {NUM_CLASSES}\n\n")
        f.write(f"#define KWS_M_CONV  {model['m_conv']}\n")
        f.write(f"#define KWS_S_CONV  {model['s_conv']}\n")
        f.write(f"#define KWS_M_DENSE {model['m_dense']}\n")
        f.write(f"#define KWS_S_DENSE {model['s_dense']}\n\n")

        f.write("static const int8_t kws_conv_w"
                "[KWS_CONV_OUT_CH][KWS_CONV_K][KWS_NUM_MFCC] = {\n")
        for oc in range(CONV_OUT_CH):
            f.write("  {\n")
            for k in range(CONV_K):
                row = ", ".join(str(cw[oc][k][ic]) for ic in range(NUM_MFCC))
                f.write(f"    {{ {row} }},\n")
            f.write("  },\n")
        f.write("};\n\n")

        f.write("static const int32_t kws_conv_b[KWS_CONV_OUT_CH] = { "
                + ", ".join(str(v) for v in cb) + " };\n\n")

        f.write("static const int8_t kws_dense_w"
                "[KWS_NUM_CLASSES][KWS_DENSE_IN] = {\n")
        for cls in range(NUM_CLASSES):
            row = ", ".join(str(dw[cls][i]) for i in range(DENSE_IN))
            f.write(f"  {{ {row} }},\n")
        f.write("};\n\n")

        f.write("static const int32_t kws_dense_b[KWS_NUM_CLASSES] = { "
                + ", ".join(str(v) for v in db) + " };\n\n")

        # MFCC normalization statistics (frozen for trained deployments)
        f.write(f"#define KWS_FEAT_FROZEN {f_frozen}\n")
        f.write("static const float kws_feat_mean[KWS_NUM_MFCC] = {\n  "
                + ", ".join(f"{v:.6f}f" for v in f_mean) + "\n};\n")
        f.write("static const float kws_feat_std[KWS_NUM_MFCC] = {\n  "
                + ", ".join(f"{v:.6f}f" for v in f_std) + "\n};\n\n")

        f.write(f"#define KWS_SELFTEST_FRAMES {SELFTEST_FRAMES}\n\n")
        f.write("#endif /* KWS_WEIGHTS_H */\n")

    # Self-test stimulus: frame-major hex bytes, loaded at run time by the
    # full-system bench and the hardware acceptance test.
    with open(os.path.join(out_dir, "selftest_frames.mem"), "w",
              newline="\n") as f:
        f.write(f"// self-test stimulus: {SELFTEST_FRAMES} frames x "
                f"{NUM_MFCC} int8 ({origin})\n")
        for fr in selftest:
            for v in fr:
                f.write(_hex8(v) + "\n")

    with open(os.path.join(out_dir, "model_params.json"), "w", newline="\n") as f:
        json.dump({
            "origin": origin,
            "num_mfcc": NUM_MFCC, "window_len": WINDOW_LEN,
            "conv_k": CONV_K, "conv_out_ch": CONV_OUT_CH,
            "conv_out_len": CONV_OUT_LEN, "pool_size": POOL_SIZE,
            "pool_out_len": POOL_OUT_LEN, "dense_in": DENSE_IN,
            "num_classes": NUM_CLASSES,
            "m_conv": model["m_conv"], "s_conv": model["s_conv"],
            "m_dense": model["m_dense"], "s_dense": model["s_dense"],
            "max_acc_conv": model["max_acc_conv"],
            "max_acc_dense": model["max_acc_dense"],
            "feat_frozen": f_frozen,
            "feat_mean": [round(v, 6) for v in f_mean],
            "feat_std": [round(v, 6) for v in f_std],
        }, f, indent=2)
        f.write("\n")
