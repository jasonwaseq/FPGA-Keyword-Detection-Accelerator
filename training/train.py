#!/usr/bin/env python3
"""
train.py - Train the KWS model on Google Speech Commands v2.

Offline tooling only (PyTorch + torchaudio required); the deployed system has
no Python anywhere. The network mirrors the hardware exactly:

    input   [B, 40 MFCC, 32 frames]           (10 ms hop, 320 ms window)
    Conv1d  (40 -> 8 channels, k=3, valid) + ReLU
    MaxPool1d(2)
    Flatten (time-major to match the accelerator's t*CH+ch layout)
    Linear  (120 -> 4)

Classes: 0=silence, 1=unknown, 2..N = target keywords (default: yes, no).

Usage:
    python training/train.py --data-dir training/data --epochs 30
    python training/export_weights.py --ckpt training/runs/kws_best.pt

The feature front end intentionally matches host/src/mfcc.c: 40 mel bands
20..7600 Hz, log, orthonormal DCT-II, per-coefficient normalization. Small
front-end mismatches degrade accuracy gracefully; the bit-exactness contract
of this project applies from INT8 features onward, not to feature extraction.
"""

import argparse
import os
import random

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchaudio


KEYWORDS = ["yes", "no"]          # class ids 2, 3
SAMPLE_RATE = 16000
N_MFCC = 40
N_FRAMES = 32                     # 320 ms window
HOP = 160


class KwsNet(nn.Module):
    """Float twin of the accelerator datapath."""

    def __init__(self, n_classes: int):
        super().__init__()
        self.conv = nn.Conv1d(N_MFCC, 8, kernel_size=3)
        self.fc = nn.Linear(15 * 8, n_classes)

    def forward(self, x):                       # x: [B, 40, 32]
        y = F.relu(self.conv(x))                # [B, 8, 30]
        y = F.max_pool1d(y, 2)                  # [B, 8, 15]
        # Hardware flatten order is time-major (i = t*CH + ch):
        y = y.permute(0, 2, 1).reshape(y.size(0), -1)   # [B, 15*8]
        return self.fc(y)


class SpeechCommandsKws(torch.utils.data.Dataset):
    """Wraps torchaudio SPEECHCOMMANDS into the 4-class KWS task."""

    def __init__(self, root: str, subset: str):
        os.makedirs(root, exist_ok=True)
        self.ds = torchaudio.datasets.SPEECHCOMMANDS(
            root, download=True, subset=subset)
        self.mfcc = torchaudio.transforms.MFCC(
            sample_rate=SAMPLE_RATE, n_mfcc=N_MFCC,
            melkwargs=dict(n_fft=512, hop_length=HOP, win_length=400,
                           n_mels=N_MFCC, f_min=20.0, f_max=7600.0))
        self.rng = random.Random(1234)

    def __len__(self):
        return len(self.ds)

    def __getitem__(self, idx):
        wav, sr, label, *_ = self.ds[idx]
        assert sr == SAMPLE_RATE
        wav = wav[0]

        # class mapping
        if label in KEYWORDS:
            cls = 2 + KEYWORDS.index(label)
        else:
            cls = 1                                     # unknown word
        # synthesize silence examples from low-amplitude noise
        if self.rng.random() < 0.1:
            wav = torch.randn_like(wav) * 0.001
            cls = 0

        feat = self.mfcc(wav.unsqueeze(0))[0]           # [40, T]
        # random 32-frame crop (pad if short)
        if feat.shape[1] < N_FRAMES:
            feat = F.pad(feat, (0, N_FRAMES - feat.shape[1]))
        t0 = self.rng.randint(0, feat.shape[1] - N_FRAMES)
        feat = feat[:, t0:t0 + N_FRAMES]

        # per-utterance normalization approximates the host's running stats
        feat = (feat - feat.mean(dim=1, keepdim=True)) \
             / (feat.std(dim=1, keepdim=True) + 1e-6)
        return feat, cls


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[1])
    ap.add_argument("--data-dir", default="training/data")
    ap.add_argument("--out-dir", default="training/runs")
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--batch", type=int, default=256)
    ap.add_argument("--lr", type=float, default=3e-3)
    ap.add_argument("--seed", type=int, default=1)
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    os.makedirs(args.out_dir, exist_ok=True)

    train_ds = SpeechCommandsKws(args.data_dir, "training")
    val_ds = SpeechCommandsKws(args.data_dir, "validation")
    train_ld = torch.utils.data.DataLoader(
        train_ds, batch_size=args.batch, shuffle=True, num_workers=4)
    val_ld = torch.utils.data.DataLoader(
        val_ds, batch_size=args.batch, num_workers=4)

    net = KwsNet(2 + len(KEYWORDS)).to(dev)
    opt = torch.optim.AdamW(net.parameters(), lr=args.lr)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, args.epochs)

    best_acc = 0.0
    for epoch in range(args.epochs):
        net.train()
        for feat, cls in train_ld:
            feat, cls = feat.to(dev), cls.to(dev)
            opt.zero_grad()
            # Weight-range penalty keeps post-training INT8 quantization tame.
            loss = F.cross_entropy(net(feat), cls) \
                 + 1e-4 * sum(p.abs().max() for p in net.parameters())
            loss.backward()
            opt.step()
        sched.step()

        net.eval()
        correct = total = 0
        with torch.no_grad():
            for feat, cls in val_ld:
                pred = net(feat.to(dev)).argmax(dim=1).cpu()
                correct += (pred == cls).sum().item()
                total += cls.numel()
        acc = correct / max(total, 1)
        print(f"epoch {epoch + 1:3d}/{args.epochs}  val_acc {acc:.4f}")

        if acc > best_acc:
            best_acc = acc
            torch.save({"state_dict": net.state_dict(),
                        "keywords": KEYWORDS, "val_acc": acc},
                       os.path.join(args.out_dir, "kws_best.pt"))

    print(f"best validation accuracy: {best_acc:.4f}")
    print(f"export with: python training/export_weights.py "
          f"--ckpt {args.out_dir}/kws_best.pt")


if __name__ == "__main__":
    main()
