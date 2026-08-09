#!/usr/bin/env python3
"""
check_unknown_diversity.py - Audit source-word diversity inside the
'unknown' class of a prepare_manifests.py manifest split.

Usage:
    python3 training/check_unknown_diversity.py --manifest ~/kws_data/work/test.txt
"""

import argparse
import os
from collections import Counter


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", required=True,
                    help="path to a manifest .txt from prepare_manifests.py "
                         "(e.g. train.txt / val.txt / test.txt)")
    ap.add_argument("--unknown-label", type=int, default=1,
                    help="label ID for the unknown class (default: 1)")
    args = ap.parse_args()

    path = os.path.expanduser(args.manifest)
    counts = Counter()
    total_lines = 0

    with open(path) as f:
        for line in f:
            parts = line.split()
            if not parts:
                continue
            total_lines += 1
            label = int(parts[0])
            if label != args.unknown_label:
                continue
            wav_path = parts[1]
            word = os.path.basename(os.path.dirname(wav_path))
            counts[word] += 1

    total_unknown = sum(counts.values())
    print(f"manifest: {path}")
    print(f"total lines: {total_lines}, unknown-labeled: {total_unknown}, "
          f"distinct source words: {len(counts)}\n")

    if not counts:
        print("no unknown-labeled entries found - check --unknown-label")
        return

    print(f"{'word':<15} {'count':>6} {'% of unknown':>14}")
    for word, c in counts.most_common():
        pct = 100.0 * c / total_unknown
        print(f"{word:<15} {c:>6} {pct:>13.1f}%")

    top5 = sum(c for _, c in counts.most_common(5))
    print(f"\ntop 5 words account for {100.0 * top5 / total_unknown:.1f}% "
          f"of all unknown clips")


if __name__ == "__main__":
    main()
