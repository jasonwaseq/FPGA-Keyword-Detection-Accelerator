# Pipeline, Latency and Throughput Analysis

## Stage inventory

| # | Stage | Mechanism | Cost (defaults, cycles) |
|---|---|---|---|
| 1 | UART reception | continuous, 115200 baud | 1 byte / ~1042 |
| 2 | Packet decode | 1 byte/cycle from FIFO | 57 (hidden under 1) |
| 3 | Feature commit | pointer update | 1 |
| 4 | Window schedule | commit counting | 1 |
| 5 | Conv1D (P=2) | 1 MAC/lane/cycle streaming | ~15.2 k |
| 6 | Pooling | 3·POOL+1 cycles/output | ~848 |
| 7 | Classifier | 1 MAC/cycle streaming | ~509 |
| 8 | Smoothing decision | fold + evaluate | 2 |
| 9 | Event encode + TX | 23-byte packet | ~70 + 23 bytes wire |

Stages 1–4 run continuously; stages 5–8 run once per scheduled window;
stage 9 only on detection.

## Fine-grained pipelining inside the engines

Every memory in the compute path has a 2-cycle read latency (registered
address, registered data). The conv and dense engines issue one operand
fetch per cycle and consume through a 2-stage valid shifter, so the MAC
throughput is 1 MAC/cycle/lane with a fixed 2-cycle drain per output row —
no stalls, no wait states in steady state:

```
cycle:      n      n+1        n+2
issue:   addr(j) addr(j+1)  addr(j+2) ...
fetch:            mem[j]     mem[j+1] ...
mac:                         acc+=j   ...
```

## Cycle budget per inference (P = 2)

```
conv:   (8/2) groups × [ 3·2+3 bias/param + 30 rows × (3·40+3+2) ] ≈ 15 236
pool:   15×8 outputs × (3·2+1)                                     ≈    848
dense:  5 + 4 × (3 + 120 + 2 + 1)                                  ≈    509
smooth:                                                                   2
total:                                                             ≈ 16 595
```

Measured in the full-system bench (STAT_LAT_MAX): ~16.6 k cycles = **1.38 ms
@ 12 MHz**, in agreement.

## Throughput and utilization

* Feature rate: 100 frames/s → one window every 8 frames = **12.5
  inferences/s**.
* Compute occupancy: 16.6 k / 960 k cycles = **1.7 %** of the machine.
* MACs per inference: 28 800 (conv) + 480 (dense) = 29 280 → 366 k MAC/s
  sustained, headroom ≈ 58× at this clock with P=2 (bounded by the feature
  link, not the datapath).
* UART RX occupancy: 5 700 B/s of 11 520 B/s = 49 %.

The design never stalls end-to-end: the RX FIFO absorbs a full packet while
the decoder streams at 1 byte/cycle (104× faster than the line), and window
overrun would require inference to exceed the 80 ms stride — margin 58×.
`windows_drop` and `fifo_overflows` counters prove this in live statistics.

## Latency budget (keyword spoken → host log line)

| Component | Typical |
|---|---|
| Audio accumulation (window fill after word onset) | 100–300 ms (algorithmic) |
| Smoothing (min_consec=2 windows × 80 ms stride) | 160 ms (algorithmic) |
| Host MFCC + packetization | < 1 ms |
| UART frame transfer (57 B) | 5 ms |
| FPGA inference + decision | 1.4 ms |
| Event packet (23 B) | 2 ms |
| **Hardware round-trip share** | **< 9 ms** |

The detection latency is dominated by the *deliberate* temporal-smoothing
policy, not by the hardware. The `EVT_KEYWORD` latency field isolates the
FPGA share (window issue → decision) per event; the host additionally
measures end-to-end latency by frame-number attribution.

## Scaling levers

| Lever | Effect | Cost |
|---|---|---|
| `PARALLEL_OUT_CH` 2→4 | conv 15.2 k→8.1 k cycles | +2 kernel-ROM EBR, +2 DSP |
| Baud 115200→921600 | 8× feature bandwidth (frames or channels) | FTDI supports it; CLK_HZ may need PLL for RX margin |
| Stride 8→4 | 25 inferences/s, finer temporal resolution | compute occupancy still <4 % |
| Larger model | limited by EBR for weights and LCs for control | see future_work.md |
