# Performance and Resource Report

Geometry: **8-channel / 10-class** (silence/unknown + yes/no/up/down/left/right/on/off).
Trained via `training/train_np.py` (provenance: `weights/model_params.json`).
Implementation: Yosys `synth_ice40 -dsp`, nextpnr-ice40 `--up5k --freq 12 --seed 7`
(oss-cad-suite); regenerate with `make bit time`.

## Model quality (Google Speech Commands v2, official test split)

| Metric | Value |
|---|---|
| 10-class window accuracy (float) | 65.7 % |
| 10-class window accuracy (INT8, bit-exact) | 65.3 % |
| Streaming detection (held-out clips, defaults thresh=25 / votes=2 / consec=1) | yes 84 %, no 62 %, up 39 %, down 41 %, left 41 %, right 68 %, on 42 %, off 45 % |
| Silence streams firing | 0 of 120 |
| Unknown streams with any keyword event | 51 of 120 (false accepts) |

Window accuracy is well below the old 2-keyword ~85 % point: eight-way classification
with an 8-channel TinyML net is capacity-limited. Streaming detection is higher
than window accuracy for several keywords because of temporal smoothing.
A 16-channel attempt trained to ~71.5 % INT8 but did not fit the UP5K LC budget.

## Hardware implementation (iCEBreaker v1.1a target)

| Resource | Used | Available | % |
|---|---|---|---|
| Logic cells (ICESTORM_LC) | 5094 | 5280 | 96 % |
| Block RAM (EBR) | 22 | 30 | 73 % |
| DSP (SB_MAC16) | 7 | 8 | 87 % |
| SPRAM | 0 | 4 | 0 % |

| Analysis | Result | Requirement |
|---|---|---|
| nextpnr (seed 7) | **16.13 MHz** | 12 MHz — PASS |
| icetime | 15.26 MHz (65.53 ns) | 12 MHz — PASS |

Bitstream: `build/kws.bin`. Full Verilator regression (bit-exact vs C reference)
passes with the trained weight ROMs.

## Fit notes (8 keywords on UP5K)

* Parallel FF history for `NUM_CLASSES=10` pushed LC over 110 %. History now
  lives in EBR with a one-class-per-cycle fold (`confidence_accumulator`).
* Combinatorial argmax over 10 logits failed timing (~7–8 MHz). Classifier and
  smoother use sequential scans; classifier→smoother handoff is registered.
* `CONV_OUT_CH=16` fitted EBR (~27/30) but not LC; stay at 8 channels unless
  the decision path is further slimmed.

## Throughput

100 fps features, 12.5 inferences/s, ~1.5 ms/inference at P=2, 30 000 MACs per
window — see [pipeline.md](pipeline.md).

## Power (qualitative)

Not measured; see [future_work.md](future_work.md). No PLL, 12 MHz fabric,
engines idle most cycles.
