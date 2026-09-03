#!/usr/bin/env python3
"""
prepare_manifests.py - Build featurizer manifests from Google Speech
Commands v2 using the official validation/testing splits.

Classes: 0=silence (crops of _background_noise_), 1=unknown (all non-target
words, subsampled), 2=yes, 3=no.

Usage:
    python prepare_manifests.py --data ~/kws_data/sc2 --out ~/kws_data/work
Outputs (in --out): train.txt, val.txt, test.txt, selftest_yes.txt
Manifest line: <label> <wav path> [offset_samples]
"""

import argparse
import os
import random

KEYWORDS = {"yes": 2, "no": 3}
UNKNOWN_TRAIN = 8000
UNKNOWN_EVAL = 1200
SIL_HOP = 2000          # samples between silence crops (heavy overlap)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--seed", type=int, default=7)
    args = ap.parse_args()
    rng = random.Random(args.seed)
    data = os.path.expanduser(args.data)
    out = os.path.expanduser(args.out)
    os.makedirs(out, exist_ok=True)

    def load_list(name):
        with open(os.path.join(data, name)) as f:
            return set(line.strip() for line in f if line.strip())

    val_set = load_list("validation_list.txt")
    test_set = load_list("testing_list.txt")

    split = {"train": [], "val": [], "test": []}

    words = [d for d in sorted(os.listdir(data))
             if os.path.isdir(os.path.join(data, d))
             and d != "_background_noise_"]
    unknown = {"train": [], "val": [], "test": []}

    for w in words:
        label = KEYWORDS.get(w, 1)
        for fn in sorted(os.listdir(os.path.join(data, w))):
            if not fn.endswith(".wav"):
                continue
            rel = f"{w}/{fn}"
            s = "val" if rel in val_set else "test" if rel in test_set else "train"
            entry = (label, os.path.join(data, rel))
            if label == 1:
                unknown[s].append(entry)
            else:
                split[s].append(entry)

    # Subsample the unknown pool (it is ~10x the keyword classes)
    for s, cap in (("train", UNKNOWN_TRAIN), ("val", UNKNOWN_EVAL),
                   ("test", UNKNOWN_EVAL)):
        rng.shuffle(unknown[s])
        split[s].extend(unknown[s][:cap])

    # Silence: overlapping 1 s crops from the long background recordings,
    # deterministically partitioned 80/10/10.
    noise_dir = os.path.join(data, "_background_noise_")
    crops = []
    for fn in sorted(os.listdir(noise_dir)):
        if not fn.endswith(".wav"):
            continue
        path = os.path.join(noise_dir, fn)
        n_samples = (os.path.getsize(path) - 44) // 2   # PCM16 mono
        for off in range(0, max(1, n_samples - 16000), SIL_HOP):
            crops.append((0, path, off))
    rng.shuffle(crops)
    n = len(crops)
    split["train"].extend(crops[: int(0.8 * n)])
    split["val"].extend(crops[int(0.8 * n): int(0.9 * n)])
    split["test"].extend(crops[int(0.9 * n):])

    for s in ("train", "val", "test"):
        rng.shuffle(split[s])
        with open(os.path.join(out, f"{s}.txt"), "w") as f:
            for e in split[s]:
                f.write(" ".join(str(x) for x in e) + "\n")
        counts = [sum(1 for e in split[s] if e[0] == c) for c in range(4)]
        print(f"{s}: {len(split[s])} clips  (sil/unk/yes/no = {counts})")

    # Candidate clips for self-test stream selection: held-out test 'yes'
    with open(os.path.join(out, "selftest_yes.txt"), "w") as f:
        cands = [e for e in split["test"] if e[0] == 2][:200]
        for e in cands:
            f.write(" ".join(str(x) for x in e) + "\n")
    print(f"selftest candidates: {len(cands)}")


if __name__ == "__main__":
    main()
