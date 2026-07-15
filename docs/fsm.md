# Finite State Machines

Every FSM in the design, with transition tables and corner-case behaviour.
All FSMs use `typedef enum` states, single `always_ff` processes with async
reset, and `unique case` (checked at simulation time via `--assert`).

---

## uart_rx (`rtl/uart_rx.sv`)

```
        rxd falling edge            cell end / maj==0
 IDLE ────────────────► START ─────────────────────► DATA ──(8 bits)──► STOP
   ▲                      │ cell end / maj==1 (glitch)                    │
   │                      ▼                                               │
   └──────────────────── IDLE ◄──────────────── mid-stop sample ──────────┘
```

| State | Exit condition | Action |
|---|---|---|
| IDLE | synchronized `rxd==0` | arm bit counter |
| START | cell end | majority==0 → DATA; majority==1 → IDLE (glitch reject) |
| DATA | cell end ×8 | shift majority sample, LSB first |
| STOP | count == MID+2 | majority==1 → `valid_o`; ==0 → `frame_err_o`; → IDLE |

Corner cases:
* **Glitch** shorter than the majority window: rejected in START.
* **Break** (line held low): first frame ends with a framing error; the
  receiver re-arms inside the break and may assemble one garbage byte as the
  line returns high — standard UART semantics, tolerated and flushed by the
  packet layer (CRC).
* **Back-to-back bytes at low CLKS_PER_BIT**: STOP exits at the stop-bit
  midpoint, restoring half a bit cell of re-arm margin (the receiver starts
  each frame 2–3 clocks late through the synchronizer). Verified at
  CLKS_PER_BIT=16 in simulation; at 104 (12 MHz/115200) margin is ~46 clocks.

## uart_tx (`rtl/uart_tx.sv`)

IDLE → START → DATA(×8) → STOP → IDLE. `ready_o` = IDLE. No corner cases:
the line idles high and a transfer is unabortable once accepted.

---

## packet_decoder (`rtl/packet_decoder.sv`)

```
 HUNT ─SOF→ VER → TYPE → LEN_L → LEN_H ─ok→ TS(×4) → FN(×4) ─len>0→ PAYLOAD(×LEN)
   ▲                            │LEN>MAX          │len==0            │
   │                            ▼                 ▼                  ▼
   ├────────── proto_err ◄──────┘               CRC_L ◄──────────────┘
   │                                              │
   └───────────────── dispatch/crc_err ◄──────── CRC_H
```

| Transition | Condition | Notes |
|---|---|---|
| HUNT→VER | byte == 0xA5 | CRC re-seeded |
| LEN_H→HUNT | LEN > 64 | `proto_err`; LEN cannot be trusted, resync |
| FN→CRC_L | LEN == 0 | commands |
| CRC_H→HUNT | always | CRC pass → dispatch (commit / cmd / semantic error); fail → `crc_err` |
| any→HUNT | 100 ms inter-byte timeout | `proto_err` |

Corner cases:
* **Speculative feature writes** stream during PAYLOAD; only the CRC-pass
  dispatch commits the frame (see architecture.md).
* **Semantic errors** (bad VER, wrong LEN for known type, unknown type) are
  recorded during parsing and reported *after* CRC verification.
* Byte pump sustains 1 byte/cycle: a new FIFO read issues on the same cycle
  the previous byte lands.

## packet_encoder (`rtl/packet_encoder.sv`)

```
 IDLE ─req→ HDR(×13) ─len>0→ PL_ADDR → PL_WAIT → PL_PUSH ─more→ PL_ADDR
              │len==0                                 │last
              ▼                                       ▼
           CRC_WAIT ◄─────────────────────────────────┘
              ▼
            CRC_L → CRC_H → IDLE (pkt_done)
```

* Outputs are registered; the CRC enable for each byte is decided at
  scheduling time (`crc_en_q`) so the SOF exclusion and payload folds line
  up with the delayed write strobe.
* `CRC_WAIT` guarantees the final fold settles before the CRC bytes are read.
* Every push waits on the TX FIFO **almost-full** flag (margin 2): the
  registered write lands one cycle after the check, and the encoder is the
  FIFO's only writer.
* Payload fetch is index→data with a 2-cycle contract, absorbed by
  PL_ADDR/PL_WAIT — sources may be register muxes or synchronous RAM.

---

## conv1d_engine (`rtl/conv1d_engine.sv`)

```
 IDLE ─start→ LD_M_A→LD_M_W→LD_M_C→LD_S_W→LD_S_C     (layer M, S)
                 ┌──────────────────────────┘
                 ▼
         ┌─ LD_B_A→LD_B_W→LD_B_C ─┐   (×PARALLEL lanes)
         │                        ▼
    next oc_group            MAC (K·IC issues + 2 drain)
         ▲                        ▼
         │                      REQ (×PARALLEL: requant+ReLU+write)
         └── last t ──────────────┤
                                  └─ next t → MAC
        last t & last group → done → IDLE
```

Corner cases: accumulator overflow is impossible by weight-export bounds
(asserted in simulation); the last lane's bias load seeds its accumulator
directly in LD_B_C (the bias register writes the same edge).

## pooling_engine (`rtl/pooling_engine.sv`)

IDLE → {RD_A → RD_W → RD_C}×POOL_SIZE → WR → … → IDLE.
`red_q` holds the running max or sum; on `q==0` the arriving sample loads
unconditionally (no initialization value needed for either mode). Write
address/data/enable register together — the RAM sees a consistent triple.

## classifier (`rtl/classifier.sv`)

Same param-load prologue as the conv engine, then per class:
LD_B_A/W/C → MAC(120 issues + 2 drain) → REQ (logit registered). The weight
address increments monotonically across the whole layer (ROM layout matches
iteration order). `done_o` pulses with the last logit write; the argmax
outputs are combinational over the completed logit register file and are
valid in the same cycle `done_o` is observed.

## temporal_smoothing (`rtl/temporal_smoothing.sv`)

Two-phase, not an explicit FSM: `update_i` folds the sample (cycle 0),
`eval_q` decides (cycle 1). The decision procedure (candidate → run length →
fire) is specified in ref_model.c `kws_smooth_step` and locked by
tb_smoothing. Corner cases: run length saturates at 15; a fired event zeroes
the run (a full fresh run is required, plus debounce); `clear_i`/engine
reset drops all history including the vote fill counter (cold-start voting
counts only real samples).

## window_scheduler (`rtl/window_scheduler.sv`)

Not a state machine but a scheduling law (see module header): first window
at commit #WINDOW_LEN, then every WINDOW_STRIDE commits, stride credit
carried across busy-engine delays, one-stride debt cap with `drop_o`.
Corner case fixed during development: stride credit must **not** accumulate
during the initial buffer fill, or the first window would issue 4× on
identical data (caught by the reference-model window schedule mirror).

## Arbiter in kws_core

A_IDLE → A_REQ → A_WAIT → A_IDLE. Events outrank responses; both requesters
hold their lines until acknowledged, so priority never loses a packet.

## statistics snapshot copier

idle / busy with 4-bit index; triggered by `snapshot_i`, copies one counter
per cycle for 16 cycles. Simulation asserts a snapshot is never re-triggered
mid-copy (the response protocol makes it impossible: header emission alone
takes longer).
