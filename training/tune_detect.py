#!/usr/bin/env python3
"""
tune_detect.py - Select the temporal-smoothing operating point from data.

The smoothing gates (threshold / votes / consecutive) trade detection rate
against false accepts. This script caches per-window logit sequences for the
held-out full-clip streams once (the expensive bit-exact part), then sweeps
the smoothing configuration over the cache and reports, per config:

    det_yes / det_no : fraction of 'yes'/'no' streams firing the right class
    xtalk            : keyword streams firing the WRONG keyword
    fa               : events fired on silence/unknown streams (false accepts)

The chosen operating point must then be reflected in THREE synchronized
places (grep for 'smoothing defaults'):
    rtl/kws_pkg.sv          - hardware CSR reset values
    host/src/ref_model.c    - kws_smooth_init defaults
    model/kws_quant.py      - SMOOTH_DEFAULTS (emission-time verification)

Usage: python3 training/tune_detect.py --work ~/kws_data/work [--per-class 200]
"""

import argparse
import itertools
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                "..", "model"))
import kws_quant as q
from train_np import load_kwsf

LABELS = ["silence", "unknown", "yes", "no"]


def cache_logits(model, fx, fy, per_class):
    """Per-stream window logit sequences (the window-schedule mirror)."""
    streams = []
    for cls in range(4):
        for i in np.where(fy == cls)[0][:per_class]:
            frames = fx[i][4:4 + q.SELFTEST_FRAMES].tolist()
            seq = []
            for n in range(q.WINDOW_LEN, len(frames) + 1):
                if (n - q.WINDOW_LEN) % 8 != 0:
                    continue
                logits = q.infer_int(frames[n - q.WINDOW_LEN:n], model)
                seq.append(logits)
            streams.append((cls, seq))
    return streams


def run_cfg(streams, **cfg):
    det = {2: 0, 3: 0}
    n_kw = {2: 0, 3: 0}
    xtalk = 0
    fa = 0
    for cls, seq in streams:
        sim = q.SmoothSim(**cfg)
        fired = set()
        for logits in seq:
            winner = logits.index(max(logits))
            ev = sim.step(logits, winner)
            if ev:
                fired.add(ev["cls"])
        if cls in (2, 3):
            n_kw[cls] += 1
            if cls in fired:
                det[cls] += 1
            if ({2, 3} - {cls}) & fired:
                xtalk += 1
        else:
            fa += len(fired)
    return (det[2] / max(n_kw[2], 1), det[3] / max(n_kw[3], 1), xtalk, fa)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--work", required=True)
    ap.add_argument("--per-class", type=int, default=200)
    args = ap.parse_args()
    work = os.path.expanduser(args.work)

    with open(os.path.join(work, "model_int8.json")) as f:
        model = json.load(f)
    fx, fy = load_kwsf(os.path.join(work, "test_full.bin"))

    print("caching stream logits (bit-exact, one-time cost)...")
    streams = cache_logits(model, fx, fy, args.per_class)
    n_bg = sum(1 for c, _ in streams if c in (0, 1))
    print(f"{len(streams)} streams cached ({n_bg} background)")

    print(f"{'dep':>3} {'thr':>4} {'vote':>4} {'consec':>6} | {'det_yes':>7} "
          f"{'det_no':>7} {'xtalk':>5} {'fa':>4}")
    results = []
    for depth, thr, vote, consec in itertools.product(
            (4, 8), (10, 15, 20, 25, 30, 35), (2, 3, 4), (1, 2)):
        dy, dn, xt, fa = run_cfg(streams, depth=depth, thresh=thr,
                                 vote_min=vote, min_consec=consec)
        results.append((depth, thr, vote, consec, dy, dn, xt, fa))
        print(f"{depth:>3} {thr:>4} {vote:>4} {consec:>6} | {dy:>7.3f} "
              f"{dn:>7.3f} {xt:>5} {fa:>4}")

    # Recommend: zero false accepts, then max mean detection
    ok = [r for r in results if r[7] == 0]
    best = max(ok, key=lambda r: r[4] + r[5]) if ok else None
    if best:
        print(f"\nrecommended (fa=0): depth={best[0]} thresh={best[1]} "
              f"vote_min={best[2]} min_consec={best[3]}  "
              f"det_yes={best[4]:.3f} det_no={best[5]:.3f} xtalk={best[6]}")


if __name__ == "__main__":
    main()
