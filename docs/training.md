# Training and Model Export

The deployed system contains no Python; training is offline tooling.

## Pipeline

```
training/train.py            PyTorch + torchaudio, Google Speech Commands v2
      │   float model, geometry-locked to the hardware (40×32 → conv 8×3 →
      │   maxpool 2 → dense 120→4, time-major flatten)
      ▼
training/runs/kws_best.pt
      │
training/export_weights.py   symmetric per-tensor INT8 quantization,
      │                      requant (M,S) calibration via model/kws_quant.py
      ▼
weights/*.mem  weights/kws_weights.h  weights/model_params.json
      │
make bit host sim            ROMs baked at synthesis; reference model and
                             regression rebuild against the same header
```

## Commands

```sh
pip install torch torchaudio
python training/train.py --data-dir training/data --epochs 30
python training/export_weights.py --ckpt training/runs/kws_best.pt
make sim          # regression re-proves HW == reference on the new weights
make bit host
```

## Quantization scheme (must match the hardware — it does by construction)

* Features: INT8, `quant_scale` units per standard deviation (host kws.ini,
  default 20) after per-coefficient running normalization.
* Weights: symmetric per-tensor INT8 (`round(w / (max|w|/127))`).
* Biases: INT32 in the accumulator domain, bounded to ±2²³ (RTL assertion).
* Requantization: `y = sat8((acc·M + 2^(S−1)) ≫ S)`, M < 2¹⁵, S ∈ [1,31],
  chosen by `kws_quant.calibrate()` so calibration accumulator maxima map
  near full INT8 scale. The export path calls the *same* calibration used
  for the bring-up weights — hardware, reference model and export cannot
  drift apart.

A max-magnitude penalty during training keeps post-training quantization
loss small at this model size; migrate to full QAT if a larger model or
tighter accuracy target demands it.

## Bring-up weights

`model/gen_weights.py` emits the checked-in deterministic set (fixed seed,
calibrated like a real export, class-2/3 kernels seeded with cosine
structure so bring-up stimulus can trigger genuine detections). Purpose:
the repository builds, verifies bit-exactly and demos end-to-end with zero
ML dependencies. CI regenerates the set and diffs it byte-for-byte.

## Changing the keyword set

1. Edit `KEYWORDS` in `training/train.py` (class ids 2..N).
2. If the class count changes: update `NUM_CLASSES` in `rtl/kws_pkg.sv`
   and `model/kws_quant.py` (assertions on both sides catch mismatches),
   and the `labels` line in `host/kws.ini`.
3. Retrain, export, `make sim bit host`.
