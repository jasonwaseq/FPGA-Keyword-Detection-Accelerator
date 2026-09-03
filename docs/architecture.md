# System Architecture

Streaming keyword-spotting (KWS) coprocessor for the Lattice iCE40UP5K
(iCEBreaker v1.1a). The host PC owns audio capture and MFCC extraction; the
FPGA owns **all** remaining computation: buffering, window scheduling, INT8
inference, temporal decision logic and event generation. There is no
request/response inference — the host streams features forever, the FPGA
processes forever and speaks only when it detects a keyword.

```
 Microphone ─ Host PC ──────────────────────────────────────────────────────┐
              audio 16 kHz → MFCC 40×INT8 @100 fps → packets ──UART 115200──┤
                                                                            ▼
 ┌─────────────────────────── iCE40UP5K (kws_core) ──────────────────────────┐
 │                                                                           │
 │ uart_rx → rx_fifo → packet_decoder ─┬─ commands ─→ register_file ──────┐  │
 │              (EBR)   CRC16, resync  │                 CSR + responses  │  │
 │                                     ▼                                 │  │
 │                        feature_buffer (64×40 INT8, EBR)               │  │
 │                        speculative write / CRC commit                 │  │
 │                                     │                                 │  │
 │                            window_scheduler                           │  │
 │                     32-frame window every 8 commits                   │  │
 │                                     ▼                                 │  │
 │  kernel_memory ──► conv1d_engine (P×MAC lanes, INT8→INT32)            │  │
 │  bias_memory   ──►   requant+ReLU → activation RAM (EBR)              │  │
 │                                     ▼                                 │  │
 │                            pooling_engine (max/avg)                   │  │
 │                              → pooled RAM (EBR)                       │  │
 │  weight_memory ──► classifier (dense 120→4 + argmax)                  │  │
 │  bias_memory   ──►                  ▼                                 │  │
 │                        temporal_smoothing (avg, vote,                 │  │
 │                          consec, debounce)                            │  │
 │                                     ▼                                 ▼  │
 │                          keyword_detector ──► [arbiter] → packet_encoder │
 │                                                              │           │
 │  statistics_counters ◄── every subsystem                     ▼           │
 │  interrupt_controller → IRQ pin, LEDs             tx_fifo → uart_tx ──►  │
 └───────────────────────────────────────────────────────────────────────────┘
                                                                UART → Host
```

## Model geometry (defaults)

| Stage | Shape | Arithmetic |
|---|---|---|
| Input window | 32 frames × 40 MFCC | INT8 |
| Conv1D (temporal, valid) | K=3, 40→8 ch → 30×8 | INT8×INT8 → INT24 acc |
| Requant + ReLU | 30×8 | (acc·M + 2^(S−1)) ≫ S, sat8 |
| Temporal pool (max/avg) | /2 → 15×8 | INT8 |
| Dense | 120 → 4 logits | INT8×INT8 → INT24 acc |
| Requant | 4 | same rescale, no ReLU |
| Smoothing | 4-deep history | moving avg + majority vote |

Windows overlap: stride is 8 frames (80 ms), so each frame participates in
four windows. All geometry is parameterized (`rtl/kws_pkg.sv`) and echoed at
run time in the `RSP_VERSION` payload so host and bitstream can verify they
agree.

## Key design decisions and tradeoffs

### Clocking — 12 MHz, no PLL
One inference costs ~15.4 k cycles (P=2) against an 80 ms (960 k-cycle)
stride budget: **1.6 % utilization**. The UART is the real bottleneck. A PLL
would buy nothing, cost a hard-macro dependency and add a timing variable;
the design closes at 14.98 MHz worst-case as-is. `CLK_HZ` is a parameter if
a faster link ever demands it.

### Feature buffer — commit pointers instead of double buffering
The packet decoder writes payload bytes **speculatively** into the write
slot while the packet is still arriving; only a verified CRC advances the
write pointer (`circular_buffer_controller`). Consequences:

* a corrupted packet costs zero cleanup — the next packet overwrites it;
* readers only ever see committed frames, writers only ever touch the one
  uncommitted slot — reception and inference are concurrent by construction
  on the two ports of one EBR, which is why classic double buffering
  (2× memory + swap control) was rejected;
* `HIST_DEPTH = 2×WINDOW_LEN` gives the writer a ≥32-frame (320 ms) head
  start over any in-flight window, versus a 1.3 ms inference: overrun is
  impossible by >200× margin (still monitored via `windows_drop`).

### Address layout — power-of-two frame stride
Feature RAM addresses are `{frame[5:0], coef[5:0]}`: 40 coefficients padded
to a 64-slot stride. This wastes 3 of 30 EBRs but makes every address
computation in the hot loop pure bit concatenation and small adds — no
multiplier, no long carry chain in the address path. The dense alternative
(`frame*40+coef`) was rejected: EBR is not the scarce resource, LCs are.

### Conv engine — parallel output-channel lanes, shared feature fetch
Loop nest `oc_group → t → (k, ic)` with `PARALLEL` MAC lanes. Each lane owns
a kernel-ROM bank; all lanes share one feature fetch per cycle (feature RAM
has a single read port — sharing it is what scales). Cycle count per window:

    OUT_CH/P × ( 3P + 3 + OUT_LEN × (K·IC + 3 + P) ) + 5

| P | cycles/window | @12 MHz | DSPs |
|---|---|---|---|
| 1 | ~29.3 k | 2.44 ms | 3 |
| 2 (default) | ~15.2 k | 1.27 ms | 5 |
| 4 | ~8.1 k | 0.68 ms | 7* |

Kernel banks replicate the full 960-byte array (2 EBR per lane) instead of
slicing: one canonical `.mem` file works for every P. Beyond P=4, slice the
initialization images instead (build-system change, not an RTL change).

*at P=4 the four requant/lane multipliers begin competing for the 8 SB_MAC16.

### Layer sequencing — daisy chain, not inter-layer pipeline
`conv.done → pool.start → classifier.start → smoothing.update`. Overlapping
layers across windows would need double-buffered activation RAMs to win back
latency that is already <2 % of the stride period. The pipeline that
matters — UART reception vs. inference — **is** concurrent.

### FIFO design — synchronous read, no fall-through
`uart_fifo` exposes a 1-cycle-latency read (`rd_en` → `rd_valid`), mapping
directly onto EBR. A first-word-fall-through wrapper (prefetch + output
register) was evaluated and rejected: both consumers are paced by the
115200-baud line (~104 clocks/byte), so FWFT's extra logic buys nothing.

### Statistics — sequential EBR snapshot
`READ_STATS` copies the 16-counter bank into a snapshot RAM one word per
cycle (per-word atomic, ≤16-cycle skew ≈ 1.3 µs, invisible next to the
5.6 ms wire time). This replaced a 512-FF shadow bank — the change that
brought the design from 105 % to under 90 % LC utilization.

### Requantization — TFLite-Micro style fixed-point rescale
`y = sat8((acc·M + 2^(S−1)) ≫ S)` with M < 2^15, S ∈ [1,31], per layer, read
from the bias `.mem` image. The 24×16 product maps onto one SB_MAC16.
Saturation is detected by a sign-extension equality check (LUT tree) rather
than magnitude comparators (carry chains) — worth ~140 LCs.

## Memory hierarchy

See [memory_map.md](memory_map.md) for layouts and address formulas.

| Memory | Geometry | Ports | Backing |
|---|---|---|---|
| RX FIFO | 256×8 | 1W/1R | 1 EBR |
| TX FIFO | 128×8 | 1W/1R | 1 EBR |
| Feature buffer | 4096×8 (64 fr × 64 stride) | 1W (decoder) / 1R (conv) | 8 EBR |
| Kernel ROM | P × 960×8 | 1R per lane | 2P EBR |
| Conv bias/param ROM | 10×32 | 1R | LUTs |
| Activation RAM | 240×8 | 1W (conv) / 1R (pool) | 1 EBR |
| Pooled RAM | 120×8 | 1W (pool) / 1R (classifier) | 1 EBR |
| Dense weight ROM | 480×8 | 1R | 1 EBR |
| Dense bias/param ROM | 6×32 | 1R | LUTs |
| Stats snapshot | 16×32 | 1W/1R | 2 EBR |
| Smoothing history | 8×4×8 + sums | all-parallel update | FFs (by necessity) |

Total: 19/30 EBR, 0/4 SPRAM (SPRAM's 16-bit single port fits nothing here
better than EBR does; it remains free for a future deeper feature history).

## Reset domains

* `rst_ni` — full asynchronous-assert/synchronous-release reset.
* `rst_eng_n` — `rst_ni` gated by a registered 1-cycle pulse from RESET or
  START commands: layer engines, smoothing and the event queue reset so a
  command never strands an FSM mid-inference, while the protocol path stays
  alive to deliver the ACK.

## Correctness contract

Everything from INT8 features to keyword events is **bit-exact** against the
C reference model (`host/src/ref_model.c`), which is compiled into every
Verilator bench *and* into the host application (`--check` runs it live
against the hardware). One arithmetic specification, three consumers.
