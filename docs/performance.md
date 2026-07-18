# Performance and Resource Report

Numbers from the checked-in implementation run (Yosys `synth_ice40 -dsp`,
nextpnr-ice40 `--up5k --freq 12 --seed 7`, trained weight ROMs); regenerate
with `make bit time` and inspect `build/nextpnr.log`, `build/icetime.rpt`,
`build/yosys.log`.

## Model quality (Google Speech Commands v2, official test split)

Trained via `training/train_np.py` (provenance: `weights/model_params.json`).

| Metric | Value |
|---|---|
| 4-class window accuracy (float) | 84.6 % |
| 4-class window accuracy (INT8, bit-exact) | 85.4 % |
| Held-out single-utterance streams firing correctly (cold start) | yes 60 %, no 66 % |
| Silence streams firing | 0 of 120 |
| Continuous validation stream (41 s, 8 keywords + 8 distractors) | **7/8 detected, 1 false accept** |

The streaming numbers are the product metric: they include the full window
schedule and the tuned decision layer (depth 4, threshold 25, votes 2/4,
consec 1, debounce 12 — selected by `tune_detect.py` / `eval_stream.py` on
held-out data). Offline prediction and live-hardware behaviour on the same
audio agreed exactly (see below).

## Hardware validation (iCEBreaker v1.1a, measured, trained bitstream)

1. **Acceptance test** (`host/build/kws_hwtest`) — the full-system bench
   scenario over the real UART: command plane, fault injection (2 corrupted
   packets), statistics ground truth, STOP/RESET semantics, error
   responses, session restart. The self-test stimulus is a held-out spoken
   "yes" whose events matched the reference model exactly. **PASS.**
2. **Real-speech validation** — a 41 s WAV of held-out test-split
   utterances (4 yes, 4 no, 8 distractor words over background noise)
   streamed through the live system: **7 of 8 keywords detected** at the
   scheduled times with correct labels, 1 false accept (a distractor word),
   0 events on noise — identical to the offline reference-model prediction,
   and all 8 hardware events agreed with the live software cross-check
   (`agree 8/8`).
3. **Live microphone** — end-to-end mic → MFCC → FPGA → event path verified
   on Windows (COM) and WSL (ttyUSB).

Measured on silicon: inference latency **16 384 cycles = 1.37 ms**
(constant), conv utilization 1.5 % at the full 100 fps feature rate,
host-measured end-to-end event latency 20–30 ms (dominated by the two UART
transfers). Opening the host serial port can glitch a few framing errors;
the host's RESET-on-connect zeroes the counters after open.

## Utilization (iCE40UP5K-SG48)

| Resource | Used | Available | % |
|---|---|---|---|
| Logic cells (ICESTORM_LC) | 4607 | 5280 | 87 % |
| Block RAM (EBR) | 19 | 30 | 63 % |
| DSP (SB_MAC16) | 7 | 8 | 87 % |
| SPRAM | 0 | 4 | 0 % |
| Global buffers | 8 | 8 | 100 % |
| I/O | 7 | 96 | 7 % |

## Timing

| Analysis | Result | Requirement |
|---|---|---|
| nextpnr (worst of two passes) | **14.98 MHz** | 12 MHz — PASS |
| icetime (independent STA) | 14.13 MHz (70.76 ns) | 12 MHz — PASS |

Critical-path / area history (kept because the fixes are instructive):

1. 32-bit latency subtraction chained into the statistics adder — fixed by
   registering the subtraction (±1 cycle measurement error).
2. Requantization saturation via magnitude comparators — replaced with
   sign-extension checks (LUT trees instead of carry chains).
3. Statistics snapshot: 512-FF shadow bank → 2-EBR sequential copy
   (105 % → <90 % LC).
4. Smoothing depth 8 → 4 (data-driven; also freed ~150 LCs and improved
   short-keyword response).

## Quantization/decision findings worth keeping

* **Calibrate the classifier rescale on the positive accumulator side.**
  Cross-entropy drives the silence logit hugely negative; scaling by
  max|acc| compressed decision margins into quantization noise and produced
  zero streaming detections despite 87 % window accuracy. Positive-side
  calibration (deep negatives saturate harmlessly) restored usable margins.
* **Smoothing depth must match keyword dwell time.** At stride 8 a short
  utterance fully covers only ~3–4 windows; an 8-deep average halves its
  effective confidence. Depth 4 tripled detection at equal false-accept
  levels.
* **Cold-start clip metrics understate warm-stream behaviour** — final
  threshold selection used a continuous featurized stream
  (`eval_stream.py`), which predicted the live hardware run exactly.

## Throughput (unchanged by training)

100 fps features, 12.5 inferences/s, 1.37 ms/inference, 29 280 MACs per
window, 58× compute headroom at P=2 — see [pipeline.md](pipeline.md).

## Power (qualitative)

Not measured; see [future_work.md](future_work.md). No PLL, 12 MHz fabric,
engines idle 98.5 % of cycles, sub-10 mW-class configuration.
