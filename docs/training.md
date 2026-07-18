# Training and Model Export

The deployed system contains no Python; everything here is offline tooling.
The repository ships with a **trained** model (provenance and metrics in
`weights/model_params.json` and [performance.md](performance.md)); this page
documents how it was produced and how to reproduce or retrain it.

## The pipeline (NumPy, no ML framework)

The model is ~1.5 k parameters — a framework buys nothing. The decisive
design choice is that **training features come from the exact deployment
front end**: the C MFCC code (`host/src/mfcc.c`) is compiled into a
featurizer tool and every training window passes through it, with
per-coefficient normalization statistics computed over the corpus and then
**frozen** into the weight artifacts. Training-time and run-time features
match by construction — there is no "training used librosa" gap.

```
Google Speech Commands v2  (~2.4 GB, 105 k utterances)
      │  training/prepare_manifests.py     official val/test splits;
      │                                    silence crops; unknown subsample
      ▼
host/build/kws_featurize   (C, deployment MFCC)
      │  pass 1: corpus mean/std per coefficient  -> stats.txt (frozen)
      │  pass 2: INT8 windows (energy-centered, ±4-frame jitter,
      │          background-noise mixing at 5..20 dB SNR for training)
      ▼
training/train_np.py       float twin of the datapath, Adam + weighted CE
      │                    (keywords 2x), then post-training INT8 quantization
      │                    with margin-preserving dense calibration
      ▼
training/tune_detect.py    smoothing operating-point sweep over held-out
      │                    full-clip streams (detection vs false accepts)
      ▼
training/emit_weights.py   artifact emission incl. frozen stats and a
      │                    verified self-test stream from a held-out clip
      ▼
weights/*.mem  kws_weights.h  selftest_frames.mem  model_params.json
      │
make sim bit host          regression re-proves HW == reference bit-exactly,
                           ROMs are baked at synthesis, host picks up stats
```

## Reproducing (WSL/Linux, Python 3 + NumPy)

```sh
# 1. dataset
mkdir -p ~/kws_data/sc2 && cd ~/kws_data
curl -L -o sc.tar.gz http://download.tensorflow.org/data/speech_commands_v0.02.tar.gz
tar -xzf sc.tar.gz -C sc2

# 2. features (repo root)
make -C host featurize
python3 training/prepare_manifests.py --data ~/kws_data/sc2 --out ~/kws_data/work
host/build/kws_featurize --manifest ~/kws_data/work/train.txt --stats-out ~/kws_data/work/stats.txt
host/build/kws_featurize --manifest ~/kws_data/work/train.txt --stats-in ~/kws_data/work/stats.txt \
    --out ~/kws_data/work/train.bin --jitter --mix-dir ~/kws_data/sc2/_background_noise_
host/build/kws_featurize --manifest ~/kws_data/work/val.txt  --stats-in ~/kws_data/work/stats.txt --out ~/kws_data/work/val.bin
host/build/kws_featurize --manifest ~/kws_data/work/test.txt --stats-in ~/kws_data/work/stats.txt --out ~/kws_data/work/test.bin
host/build/kws_featurize --manifest ~/kws_data/work/test.txt --stats-in ~/kws_data/work/stats.txt --out ~/kws_data/work/test_full.bin --all-frames

# 3. train + tune + emit
python3 training/train_np.py    --work ~/kws_data/work
python3 training/tune_detect.py --work ~/kws_data/work
#   -> if you change the smoothing operating point, apply it to the THREE
#      synchronized defaults (grep 'smoothing defaults'):
#      rtl/kws_pkg.sv, host/src/ref_model.c, model/kws_quant.py
python3 training/emit_weights.py --work ~/kws_data/work --weights-out weights

# 4. rebuild + reverify + reflash
make sim bit host
# hardware acceptance: make -C host hwtest && host/build/kws_hwtest /dev/ttyUSB1
```

## Quantization scheme (matches the hardware by construction)

* Features: INT8, `quant_scale` units per (frozen, corpus-wide) standard
  deviation.
* Weights: symmetric per-tensor INT8; biases INT32 in the accumulator
  domain, bounded to ±2²³ (RTL assertion).
* Requantization `y = sat8((acc·M + 2^(S−1)) ≫ S)` per layer, M < 2¹⁵.
* Cross-layer bias scaling: with `k1 = (M1/2^S1)/s_w1`, dense biases are
  `round(b·k1/s_w2)`, keeping INT8 pre-activations proportional to float.
* **Dense calibration against the maximum positive accumulator**, not
  max|acc|: cross-entropy drives the silence logit deeply negative while
  the decision margins live on the positive side; absolute-max scaling
  would compress them into quantization noise (deep negatives saturate at
  −128 harmlessly). This detail is what makes streaming thresholds usable.

## The decision layer is part of the model

Single-window accuracy is not the product metric — event behaviour is.
`tune_detect.py` sweeps smoothing depth / threshold / votes / consecutive
over held-out full-clip streams and reports detection rate, cross-talk and
false accepts per configuration; the shipped defaults are the fa-conscious
operating point from that sweep (numbers in
[performance.md](performance.md)). The same simulator (`kws_quant.SmoothSim`)
is bit-exact against the RTL smoother, so the sweep predicts hardware
behaviour exactly.

## Self-test stream

`emit_weights.py` selects a held-out test 'yes' utterance whose feature
stream provably fires the pipeline under both the clean and fault-injection
patterns used by `tb_kws_core` and `kws_hwtest`, and ships it as
`weights/selftest_frames.mem`. Retraining regenerates it, keeping the
regression and the hardware acceptance test weight-agnostic.

## Bring-up weights

`model/gen_weights.py` still produces the deterministic untrained set
(cosine matched-filter kernels + synthetic self-test stream) for hardware
bring-up with zero ML dependencies; CI checks it stays deterministic.

## Changing the keyword set

1. Edit `KEYWORDS` in `training/prepare_manifests.py`.
2. If the class count changes: update `NUM_CLASSES` in `rtl/kws_pkg.sv` and
   `model/kws_quant.py`, and `labels` in `host/kws.ini`.
3. Re-run the pipeline above.
