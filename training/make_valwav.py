#!/usr/bin/env python3
"""
make_valwav.py - Build a continuous validation WAV from held-out test clips.

Stitches real Speech Commands utterances (never seen in training) into one
16 kHz mono stream with low background noise between them, and prints the
schedule of expected detections. Streaming this file through the live system
(kws_host --input wav:...) demonstrates genuine keyword recognition on the
FPGA: 'yes'/'no' events must land at the scheduled positions and nothing may
fire on the unknown-word or noise segments.

Usage:
  python3 training/make_valwav.py --data ~/kws_data/sc2 --work ~/kws_data/work \
      --out ~/kws_data/work/validate.wav
"""

import argparse
import os
import random
import struct
import wave


def read_wav_mono16(path, want_rate=16000):
    with wave.open(path, "rb") as w:
        assert w.getframerate() == want_rate and w.getsampwidth() == 2
        n = w.getnframes()
        raw = w.readframes(n)
        data = list(struct.unpack(f"<{n * w.getnchannels()}h", raw))
        if w.getnchannels() == 2:
            data = [(data[i] + data[i + 1]) // 2 for i in range(0, len(data), 2)]
    return data


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--work", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--n-each", type=int, default=4, help="clips per keyword")
    args = ap.parse_args()
    rng = random.Random(args.seed)
    data_dir = os.path.expanduser(args.data)
    work = os.path.expanduser(args.work)
    out_path = os.path.expanduser(args.out)

    # held-out clips from the official test split
    test = {}
    with open(os.path.join(work, "test.txt")) as f:
        for line in f:
            parts = line.split()
            if len(parts) < 2 or len(parts) > 2:
                continue                       # skip silence entries (offsets)
            lab = int(parts[0])
            test.setdefault(lab, []).append(parts[1])
    for v in test.values():
        rng.shuffle(v)

    # background noise bed
    noise_dir = os.path.join(data_dir, "_background_noise_")
    noise_file = os.path.join(noise_dir, sorted(
        f for f in os.listdir(noise_dir) if f.endswith(".wav"))[0])
    noise = read_wav_mono16(noise_file)

    def noise_seg(n):
        off = rng.randint(0, len(noise) - n - 1)
        return [int(s * 0.08) for s in noise[off:off + n]]

    # schedule: noise gaps between utterances; unknown words interleaved
    sequence = []
    for i in range(args.n_each):
        sequence.append(("yes", test[2][i]))
        sequence.append(("unknown", test[1][2 * i]))
        sequence.append(("no", test[3][i]))
        sequence.append(("unknown", test[1][2 * i + 1]))

    samples = noise_seg(24000)                 # 1.5 s lead-in
    schedule = []
    for label, path in sequence:
        t0 = len(samples) / 16000.0
        clip = read_wav_mono16(path)
        samples.extend(clip)
        schedule.append((t0, label, os.path.relpath(path, data_dir)))
        samples.extend(noise_seg(24000))       # 1.5 s gap

    with wave.open(out_path, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(16000)
        w.writeframes(struct.pack(f"<{len(samples)}h",
                                  *[max(-32768, min(32767, s)) for s in samples]))

    print(f"wrote {out_path}: {len(samples) / 16000.0:.1f} s")
    print("expected detections (time  label  source clip):")
    for t0, label, rel in schedule:
        mark = ">>" if label in ("yes", "no") else "  "
        print(f"  {mark} {t0:6.2f}s  {label:8s} {rel}")


if __name__ == "__main__":
    main()
