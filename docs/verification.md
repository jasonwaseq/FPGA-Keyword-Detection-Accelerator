# Verification Strategy

Full plan and per-test detail: [verification/testplan.md](../verification/testplan.md).

## Principles

1. **One golden model.** The C reference (`host/src/ref_model.c` +
   `protocol.c`) specifies the arithmetic and decision behaviour bit-exactly.
   It is linked into every Verilator bench *and* shipped inside the host
   application (`--check` cross-checks live hardware). Any RTL/model
   divergence is a bug by definition — there is no "close enough".
2. **Self-checking, deterministic, CI-gated.** Every bench seeds its own
   xorshift RNG, checks its results and exits nonzero on failure; the
   regression is a `make sim` away and gates every push.
3. **Unit benches verify timing contracts; the system bench verifies the
   product.** Unit benches emulate each engine's memory interfaces with
   exact rom_sync/ram_dp_sync timing (registered address + registered data),
   so integration cannot be broken by latency assumptions. The full-system
   bench then drives only the real external interface — bit-level UART — and
   observes only real outputs.

## Bench inventory (13 builds, `make sim`)

| Bench | DUT | Oracle | Highlights |
|---|---|---|---|
| tb_crc | crc16 | kws_crc16 | 200 random streams, re-seed, idle gaps |
| tb_uart_rx | uart_rx | UART BFM | 300 bytes random gaps, glitch reject, break/framing, recovery |
| tb_uart_tx | uart_tx | UART BFM | 300 bytes, back-to-back and gapped |
| tb_fifo | uart_fifo | C++ deque | 200 k random ops, overflow-drop, wrap, level/flags |
| tb_decoder | packet_decoder | C parser + builder | corruption anywhere in packet, resync through garbage, oversized LEN, semantic errors, timeout |
| tb_encoder | packet_encoder | kws_pkt_build | every payload length 0..64, byte-exact wire, random back-pressure |
| tb_conv1d_p1/p2/p4 | conv1d_engine | kws_ref_conv1d | 12 windows each at PARALLEL=1/2/4, random bases incl. wrap, bit-exact |
| tb_pooling | pooling_engine | kws_ref_pool | max+avg, full signed range, 16 runs |
| tb_classifier | classifier | kws_ref_dense/argmax | 24 runs, logits + winner bit-exact |
| tb_smoothing | temporal_smoothing | kws_smooth_step | 5 configs × 600 inferences, fire timing + payload equality, disabled-mode |
| tb_kws_core | kws_core | full reference mirror | see below |

## Full-system bench (tb_kws_core)

Real 8N1 UART bit timing on both directions (baud raised to CLK/16 by
parameter override purely for wall-clock speed — the same override is *not*
used for the UART unit benches, which run at the shipping 104 clocks/bit).

Covered end to end, in one continuous scenario:

* command plane: PING/ACK with frame echo, READ_VERSION geometry contract
  against the compiled weight header, START/STOP/RESET;
* streaming: 90 frames with keyword bursts; **every** EVT_KEYWORD compared
  to the reference event stream in order, class, confidence, votes, frame
  attribution and latency plausibility;
* fault injection: two CRC-corrupted feature packets → counted, not
  committed, reference mirror skips them identically;
* statistics: frames_rx / crc_errors / inferences / windows_sched /
  keywords / windows_drop / frames_dropped / fifo_overflows cross-checked
  against ground truth; STOP-then-feature increments frames_dropped;
  RESET zeroes the bank;
* protocol errors: unknown type, bad length, bad version → correct
  RSP_ERROR payloads;
* session restart: a second START must behave exactly like a fresh boot
  (window alignment and smoothing history cleared).

## Assertions

Simulation-only immediate assertions (`` `ifndef SYNTHESIS ``, enabled with
`--assert`) guard structural invariants: FIFO read-while-empty, TX FIFO
overflow, accumulator overflow vs. the 24-bit bound, feature index range,
encoder request length, event-queue overflow, elaboration-time parameter
legality (power-of-two depths, divisibility, stride bounds).

## What found real bugs during development

Recorded because a verification chapter that never failed is a red flag:

1. **UART stop-bit re-arm margin** — full-system bench at CLKS_PER_BIT=16
   showed accumulating start-bit misalignment on back-to-back bytes; fixed
   by mid-stop exit. Unit test at 104 clocks/bit had never provoked it.
2. **Window scheduler stride credit** — the reference-model schedule mirror
   exposed 4× duplicate windows at stream start (credit accumulated during
   buffer fill); fixed with first-window semantics.
3. **Encoder CRC timing** — registered outputs lagging the state machine
   folded the SOF into the CRC and read the CRC one fold early; fixed with
   scheduling-time `crc_en_q` + settle state (caught by tb_encoder byte
   comparison).
4. **Pooling write-address race** — registered enable with post-incremented
   address wrote every result one slot high (caught by tb_pooling).

## Running

```
make sim                # everything
make -C sim run-tb_kws_core SIMARGS=+trace   # one bench with VCD
gtkwave sim/out/tb_kws_core.vcd
```
