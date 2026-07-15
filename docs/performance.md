# Performance and Resource Report

Numbers from the checked-in implementation run (Yosys `synth_ice40 -dsp`,
nextpnr-ice40 `--up5k --freq 12 --seed 7`); regenerate with `make bit time`
and inspect `build/nextpnr.log`, `build/icetime.rpt`, `build/yosys.log`.

## Utilization (iCE40UP5K-SG48)

| Resource | Used | Available | % |
|---|---|---|---|
| Logic cells (ICESTORM_LC) | 4951 | 5280 | 93 % |
| Block RAM (EBR) | 19 | 30 | 63 % |
| DSP (SB_MAC16) | 7 | 8 | 87 % |
| SPRAM | 0 | 4 | 0 % |
| Global buffers | 8 | 8 | 100 % |
| I/O | 7 | 96 | 7 % |

Post-synthesis cell mix: 3778 SB_LUT4, ~2410 flip-flops, 1077 SB_CARRY.

DSP allocation: 2 conv MAC lanes + 1 dense MAC + 2 requantization
multipliers (+2 from inference splitting). EBR: see
[memory_map.md](memory_map.md).

## Timing

| Analysis | Result | Requirement |
|---|---|---|
| nextpnr (worst of two passes) | **13.16 MHz** | 12 MHz — PASS |
| icetime (independent STA) | 12.58 MHz (79.51 ns) | 12 MHz — PASS |

Critical-path history (documented because the fixes are instructive):

1. *Initial*: 32-bit latency subtraction (`cycles − lat_start`) chained
   combinationally into the statistics accumulator adder — two 32-bit carry
   chains plus routing. Fixed by registering the subtraction (measurement
   becomes one cycle stale: 83 ns error on millisecond quantities).
2. *Current*: requantization path (accumulator mux → SB_MAC16 → 41-bit
   round/shift/saturate), comfortably inside the 83 ns budget. The next
   frequency step would pipeline the product register — the engines already
   treat requantization as a separate state, so it is a localized change.

Area history: an early build was 105 % LC. Recovered by (a) moving the
statistics snapshot from a 512-FF shadow bank into 2 EBRs and (b) replacing
saturation magnitude comparators with sign-extension checks.

## Throughput / latency (measured in the full-system bench and live stats)

| Metric | Value |
|---|---|
| Feature rate | 100 frames/s (40 B + framing @ 115200 baud, 49 % line use) |
| Inference rate | 12.5 windows/s (stride 8) |
| Inference latency (window issue → decision) | ~16.6 k cycles = 1.38 ms |
| Compute occupancy | 1.7 % (58× headroom at P=2, 12 MHz) |
| Hardware share of detection latency | < 9 ms including both UART transfers |
| Sustained MAC rate | 366 k MAC/s (24 MMAC/s peak at P=2) |

Detection latency end-to-end is dominated by the configured temporal
smoothing policy (≥ 2 consecutive windows at 80 ms stride), not hardware —
see [pipeline.md](pipeline.md) for the full budget.

## Power (qualitative)

Not measured on hardware. Structural choices that matter at this node: no
PLL, 12 MHz fabric clock, compute engines idle (no switching in the MAC
array) 98 % of the time, EBRs clock-enabled by their FSMs. The UP5K is a
sub-10 mW-class device in this configuration; publishing a measured figure
requires the board instrumentation listed in future_work.md.
