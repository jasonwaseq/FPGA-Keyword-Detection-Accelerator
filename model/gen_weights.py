#!/usr/bin/env python3
"""
gen_weights.py - Deterministic bring-up weight set for the KWS accelerator.

Generates a functional, fully calibrated INT8 parameter set with a fixed seed
so that the repository builds and verifies bit-exactly out of the box, with no
training dependency. These weights exercise the complete datapath (all
verification is bit-accurate against them) but are NOT trained: for real
keyword-spotting accuracy, train with training/train.py and export with
training/export_weights.py, which emits the identical file set.

The class-2 and class-3 kernels are seeded with low-frequency structure so
that the host's --selftest stimulus can provoke genuine detection events on
untrained weights (useful for end-to-end hardware bring-up).

Usage: python model/gen_weights.py [--out weights] [--seed 20260712]
"""

import argparse
import random
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import kws_quant as q


def synth_weights(seed: int):
    rng = random.Random(seed)

    # Small-magnitude random weights: keeps accumulators comfortably inside
    # the 24-bit RTL bound while using the full INT8 feature range.
    conv_w = [[[rng.randint(-16, 16) for _ in range(q.NUM_MFCC)]
               for _ in range(q.CONV_K)] for _ in range(q.CONV_OUT_CH)]
    conv_b = [rng.randint(-4096, 4096) for _ in range(q.CONV_OUT_CH)]

    # Give the keyword classes matched-filter structure: channel oc responds
    # to a cosine-shaped MFCC envelope. This makes bring-up detections
    # reproducible (host --selftest synthesizes the matching envelope).
    import math
    for oc in (2, 3):
        for k in range(q.CONV_K):
            for ic in range(q.NUM_MFCC):
                v = 20.0 * math.cos(2.0 * math.pi * (oc - 1) * ic / q.NUM_MFCC)
                conv_w[oc][k][ic] = int(round(v))

    dense_w = [[rng.randint(-16, 16) for _ in range(q.DENSE_IN)]
               for _ in range(q.NUM_CLASSES)]
    dense_b = [rng.randint(-2048, 2048) for _ in range(q.NUM_CLASSES)]

    # Bias each keyword class toward its matched conv channel so that a strong
    # channel-oc response wins the argmax.
    for cls in (2, 3):
        for t in range(q.POOL_OUT_LEN):
            dense_w[cls][t * q.CONV_OUT_CH + cls] = 24

    return conv_w, conv_b, dense_w, dense_b, rng


def make_selftest(rng):
    """Deterministic self-test stream: noise with a long class-2 cosine burst
    (frames 30..69) matching the seeded matched-filter kernels."""
    import math
    frames = []
    for i in range(q.SELFTEST_FRAMES):
        fr = []
        for ic in range(q.NUM_MFCC):
            v = rng.randint(-12, 12)
            if 30 <= i < 70:
                v += int(round(48.0 * math.cos(2.0 * math.pi * ic / q.NUM_MFCC)))
            fr.append(max(-128, min(127, v)))
        frames.append(fr)
    return frames


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[1])
    ap.add_argument("--out", default="weights", help="output directory")
    ap.add_argument("--seed", type=int, default=20260712)
    args = ap.parse_args()

    conv_w, conv_b, dense_w, dense_b, rng = synth_weights(args.seed)
    model = q.calibrate(conv_w, conv_b, dense_w, dense_b, rng)
    q.emit(model, args.out, origin=f"gen_weights.py seed={args.seed}",
           selftest=make_selftest(rng))

    print(f"wrote {args.out}/: conv_weights.mem dense_weights.mem "
          f"conv_bias.mem dense_bias.mem kws_weights.h model_params.json")
    print(f"requant: conv M={model['m_conv']} S={model['s_conv']} "
          f"(max|acc|={model['max_acc_conv']}), "
          f"dense M={model['m_dense']} S={model['s_dense']} "
          f"(max|acc|={model['max_acc_dense']})")


if __name__ == "__main__":
    main()
