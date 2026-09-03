#!/usr/bin/env python3
"""
emit_weights.py - Final deployment-artifact emission for a trained model.

Loads the quantized model saved by train_np.py, selects a held-out 'yes'
self-test stream that provably fires through the CURRENT smoothing defaults
(kws_quant.SMOOTH_DEFAULTS - keep in sync with rtl/kws_pkg.sv and
ref_model.c), and emits the complete artifact set into weights/.

Run AFTER tune_detect.py and after the chosen smoothing defaults have been
applied to the three synchronized locations.

Usage: python3 training/emit_weights.py --work ~/kws_data/work --weights-out weights
"""

import argparse
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                "..", "model"))
import kws_quant as q
from train_np import load_kwsf, pick_selftest, stream_metrics, LABELS


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--work", required=True)
    ap.add_argument("--weights-out", default="weights")
    args = ap.parse_args()
    work = os.path.expanduser(args.work)

    with open(os.path.join(work, "model_int8.json")) as f:
        model = json.load(f)

    fx, fy = load_kwsf(os.path.join(work, "test_full.bin"))

    print("streaming metrics at current smoothing defaults "
          f"({q.SMOOTH_DEFAULTS}):")
    met = stream_metrics(model, fx, fy)
    for cls in range(4):
        n, fired, none = met[cls]
        print(f"  streams[{LABELS[cls]:8s}] n={n:3d} -> events "
              f"sil={fired[0]} unk={fired[1]} yes={fired[2]} no={fired[3]} "
              f"(no event: {none})")

    stream, picked = pick_selftest(model, fx, fy)
    print(f"selftest stream: held-out test clip #{picked} ('yes')")

    mean, std = [], []
    with open(os.path.join(work, "stats.txt")) as f:
        for line in f:
            a, b = line.split()
            mean.append(float(a))
            std.append(float(b))

    origin = (f"train_np.py speech_commands_v2 "
              f"float={model.get('float_test_acc', 0):.4f} "
              f"int8={model.get('int8_test_acc', 0):.4f} "
              f"seed={model.get('seed', '?')}")
    q.emit(model, args.weights_out, origin=origin,
           feat_stats=(mean, std, 1), selftest=stream)
    print(f"emitted deployment artifacts to {args.weights_out}/ ({origin})")


if __name__ == "__main__":
    main()
