# Verification Test Plan

Requirement-to-test traceability for the iCE40 KWS accelerator. All tests
are self-checking Verilator benches under `tb/`, built and executed by
`make sim` (also in CI). Golden model: `host/src/ref_model.c`,
`host/src/protocol.c` — the same sources compiled into the host application.

Status column reflects the checked-in regression: **13/13 PASS**.

Hardware execution: the same scenario runs against the real iCEBreaker via
`host/src/hwtest.c` (build `make -C host hwtest`, run
`./host/build/kws_hwtest /dev/ttyUSB1`). Result on the reference board with
the shipped **trained** bitstream: **PASS** — V-08..V-31 re-confirmed on
silicon, keyword events (from a held-out spoken "yes" self-test stream)
bit-exact against the reference model, measured inference latency 1.37 ms.
Additionally, a 41 s continuous stream of held-out test-split utterances
produced 7/8 correct detections + 1 false accept on the live board,
matching the offline reference prediction exactly (8/8 event agreement).
See docs/performance.md, "Hardware validation".

| ID | Requirement | Test(s) | Method | Status |
|----|-------------|---------|--------|--------|
| V-01 | CRC16-CCITT-FALSE equivalence HW/SW | tb_crc | 200 random streams vs C | PASS |
| V-02 | UART RX byte integrity, arbitrary gaps | tb_uart_rx | BFM, 300 random bytes | PASS |
| V-03 | UART RX glitch immunity | tb_uart_rx | 3-clock low pulses ×20 | PASS |
| V-04 | UART RX break → framing error + recovery | tb_uart_rx | 15-bit-time break | PASS |
| V-05 | UART TX wire format | tb_uart_tx | BFM monitor, 300 bytes | PASS |
| V-06 | FIFO data/flag/level correctness | tb_fifo | 200 k random ops vs deque | PASS |
| V-07 | FIFO overflow-drop accounting | tb_fifo | forced-full phases | PASS |
| V-08 | Packet parse: features, metadata, speculative writes | tb_decoder | 20 frames byte-compared | PASS |
| V-09 | Packet parse: all 6 commands | tb_decoder | directed | PASS |
| V-10 | CRC rejection: single-bit corruption anywhere | tb_decoder | 30 random flips; no commit ever | PASS |
| V-11 | Resync through garbage / oversized LEN / timeout | tb_decoder | directed, TIMEOUT=2000 build | PASS |
| V-12 | Semantic errors (bad ver / bad len / unknown type) | tb_decoder, tb_kws_core | directed | PASS |
| V-13 | Encoder wire bytes = C builder, len 0..64 | tb_encoder | exhaustive lengths + random | PASS |
| V-14 | Encoder under TX back-pressure | tb_encoder | 40 % random full | PASS |
| V-15 | Conv1D bit-exact (requant+ReLU) | tb_conv1d_p1/p2/p4 | 12 windows/config vs ref | PASS |
| V-16 | Conv parallelism P ∈ {1,2,4} | tb_conv1d_p* | parameter overrides | PASS |
| V-17 | Circular window wrap-around | tb_conv1d_*, tb_kws_core | random bases; >64 streamed frames | PASS |
| V-18 | Max pool bit-exact | tb_pooling | 8 runs vs ref | PASS |
| V-19 | Avg pool bit-exact (floor shift) | tb_pooling | 8 runs vs ref | PASS |
| V-20 | Dense layer + argmax bit-exact | tb_classifier | 24 runs vs ref | PASS |
| V-21 | Smoothing decision equivalence | tb_smoothing | 5 configs × 600 steps | PASS |
| V-22 | Smoothing gates: threshold/votes/consec/debounce/mask/enable | tb_smoothing | config sweep incl. degenerate | PASS |
| V-23 | Window schedule = reference schedule | tb_kws_core | event frame attribution | PASS |
| V-24 | End-to-end event equality (class/conf/votes/frame) | tb_kws_core | 2 sessions, every event | PASS |
| V-25 | Corrupted frame not committed; ref mirror skips | tb_kws_core | fault injection ×2 | PASS |
| V-26 | Statistics counters vs ground truth | tb_kws_core | 8 counters checked exactly | PASS |
| V-27 | STOP drops frames; RESET clears stats | tb_kws_core | directed | PASS |
| V-28 | Session restart = fresh boot | tb_kws_core | second START session | PASS |
| V-29 | VERSION geometry contract | tb_kws_core | payload vs compiled header | PASS |
| V-30 | No overrun under continuous streaming | tb_kws_core | windows_drop==0, fifo_ovf==0 | PASS |
| V-31 | Latency measurement plausibility | tb_kws_core | 0 < lat < 100 k cycles | PASS |
| V-32 | Structural invariants | all | immediate assertions (--assert) | PASS |

## Known gaps / future verification work

* PARALLEL=8 is covered by lint and elaboration assertions only (the packed
  port exceeds the 64-bit fast path of the bench harness); extend the
  harness with VlWide accessors before enabling P=8 in production.
* Average-pool mode is verified at unit level and via the pool-mode CSR
  default path; a full-system run with `cfg_pool_mode=1` requires the
  WRITE_REG protocol extension.
* No gate-level simulation (post-PnR); icetime static timing plus generous
  margin at 12 MHz stands in.
* MFCC front end is validated by construction (standard algorithm), not
  bit-locked — deliberate: the bit-exactness contract starts at INT8
  features (see docs/architecture.md).

## Debug utilities

`sim/probe_main.cpp` — internal-visibility harness (build the model with
`--public-flat-rw`) used during bring-up to isolate the RX chain; kept as a
worked example for future integration debug.
