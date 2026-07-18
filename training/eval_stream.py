#!/usr/bin/env python3
"""
eval_stream.py - Evaluate smoothing operating points on a continuous
featurized stream (e.g. the validation WAV), reporting events against an
expected schedule. This is the exact decision pipeline the FPGA runs
(window law + bit-exact inference + smoothing), so results here predict
hardware behaviour on the same audio.

Usage:
  python3 training/eval_stream.py --work ~/kws_data/work \
      --stream ~/kws_data/work/validate.bin \
      --schedule 1.50:yes 6.50:no 11.38:yes 16.07:no 21.07:yes 26.07:no \
                 31.07:yes 36.05:no \
      [--sweep]
"""

import argparse
import itertools
import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                "..", "model"))
import kws_quant as q
from train_np import load_kwsf

LABELS = ["silence", "unknown", "yes", "no"]
LAB_IDX = {"yes": 2, "no": 3}


def stream_logit_seq(model, frames):
    seq = []   # (commit_index, logits)
    for n in range(q.WINDOW_LEN, len(frames) + 1):
        if (n - q.WINDOW_LEN) % 8:
            continue
        lg = q.infer_int(frames[n - q.WINDOW_LEN:n], model)
        seq.append((n - 1, lg))
    return seq


def run(seq, schedule, tol_s=1.6, **cfg):
    """Events vs schedule. A hit = right class within tol of the utterance
    start; anything else is a false accept. Returns (hits, misses, fa, log)."""
    sim = q.SmoothSim(**cfg)
    events = []
    for idx, lg in seq:
        winner = lg.index(max(lg))
        ev = sim.step(lg, winner)
        if ev:
            events.append((idx / 100.0, ev["cls"], ev["conf"]))
    hits, fa = 0, 0
    used = [False] * len(schedule)
    log = []
    for t, cls, conf in events:
        matched = False
        for i, (ts, lab) in enumerate(schedule):
            if not used[i] and LAB_IDX[lab] == cls and ts <= t <= ts + tol_s:
                used[i] = True
                hits += 1
                matched = True
                log.append(f"  {t:6.2f}s  {LABELS[cls]:4s} conf={conf:3d}  HIT ({lab} @ {ts:.2f}s)")
                break
        if not matched:
            fa += 1
            log.append(f"  {t:6.2f}s  {LABELS[cls]:4s} conf={conf:3d}  FALSE-ACCEPT")
    misses = used.count(False)
    for i, (ts, lab) in enumerate(schedule):
        if not used[i]:
            log.append(f"  {ts:6.2f}s  {lab:4s}            MISS")
    return hits, misses, fa, log


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--work", required=True)
    ap.add_argument("--stream", required=True)
    ap.add_argument("--schedule", nargs="+", required=True,
                    help="time:label entries")
    ap.add_argument("--sweep", action="store_true")
    args = ap.parse_args()
    work = os.path.expanduser(args.work)

    with open(os.path.join(work, "model_int8.json")) as f:
        model = json.load(f)
    fx, _ = load_kwsf(os.path.expanduser(args.stream))
    frames = fx[0].tolist()
    print(f"stream: {len(frames)} frames ({len(frames) / 100.0:.1f} s)")

    schedule = []
    for ent in args.schedule:
        ts, lab = ent.split(":")
        schedule.append((float(ts), lab))

    seq = stream_logit_seq(model, frames)

    if args.sweep:
        print(f"{'dep':>3} {'thr':>4} {'vote':>4} {'consec':>6} | "
              f"{'hits':>4} {'miss':>4} {'fa':>3}")
        for depth, thr, vote, consec in itertools.product(
                (4, 8), (15, 20, 25, 30), (2, 3, 4), (1, 2)):
            h, m, fa, _ = run(seq, schedule, depth=depth, thresh=thr,
                              vote_min=vote, min_consec=consec)
            print(f"{depth:>3} {thr:>4} {vote:>4} {consec:>6} | "
                  f"{h:>4} {m:>4} {fa:>3}")
    else:
        h, m, fa, log = run(seq, schedule)
        print("\n".join(log))
        print(f"hits={h} misses={m} false_accepts={fa} "
              f"(defaults {q.SMOOTH_DEFAULTS})")


if __name__ == "__main__":
    main()
