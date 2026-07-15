# UART Packet Protocol (v1)

Binary framing over 8N1 UART, 115200 baud default. Little-endian multi-byte
fields. Same definition in `rtl/kws_pkg.sv` and `host/include/kws_protocol.h`;
the full-system bench cross-checks the two through `RSP_VERSION`.

## Frame format

```
offset  size  field
0       1     SOF        0xA5
1       1     VER        protocol version, 0x01
2       1     TYPE       packet type (below)
3       2     LEN        payload length, 0..64
5       4     TIMESTAMP  sender time, ms (host monotonic / FPGA uptime)
9       4     FRAME_NUM  feature frame counter / echoed command frame
13      LEN   PAYLOAD
13+LEN  2     CRC16      CCITT-FALSE (poly 0x1021, init 0xFFFF, MSB-first,
                         no reflection) over bytes 1 .. 12+LEN
```

Maximum packet: 13 + 64 + 2 = 79 bytes (~6.9 ms on the wire at 115200).

## Packet types

### Host → FPGA

| Type | Value | LEN | Semantics |
|---|---|---|---|
| `CMD_PING` | 0x01 | 0 | liveness check → ACK |
| `CMD_RESET` | 0x02 | 0 | soft reset: pipeline, engines, statistics; stream disabled → ACK |
| `CMD_START_STREAM` | 0x03 | 0 | enable feature commits; clears window/smoothing history (fresh session) → ACK |
| `CMD_STOP_STREAM` | 0x04 | 0 | disable commits (features then count as `frames_dropped`) → ACK |
| `CMD_READ_STATS` | 0x05 | 0 | atomic counter snapshot → RSP_STATS |
| `CMD_READ_VERSION` | 0x06 | 0 | identity/geometry → RSP_VERSION |
| `DATA_FEATURE` | 0x10 | 40 | one MFCC frame, 40×INT8; FRAME_NUM = monotonically increasing frame counter |

The host sends **one command at a time** and awaits the response (the
response queue on the FPGA is one deep). Feature packets may be pipelined
freely — they generate no response.

### FPGA → Host

| Type | Value | LEN | Payload |
|---|---|---|---|
| `RSP_ACK` | 0x81 | 1 | `[0]` echoed command type |
| `RSP_ERROR` | 0x82 | 2 | `[0]` error code, `[1]` offending TYPE byte |
| `RSP_STATS` | 0x83 | 64 | 16 × u32 counters (map below) |
| `RSP_VERSION` | 0x84 | 12 | identity/geometry (below) |
| `EVT_KEYWORD` | 0x90 | 8 | detection event (below) |

FPGA→host header fields: TIMESTAMP = FPGA uptime in ms; FRAME_NUM = echoed
command frame (responses) or the newest host frame in the detected window
(events).

Error codes: `0x01 BAD_LENGTH` (LEN inconsistent with TYPE), `0x02
UNKNOWN_TYPE`, `0x03 BAD_VERSION`. Semantic errors are only reported for
packets that pass CRC; CRC failures are counted silently (a corrupt packet
cannot be attributed reliably) and the parser re-synchronizes on the next
SOF. An inter-byte timeout (100 ms) also returns the parser to hunt mode.

### EVT_KEYWORD payload

| Offset | Field |
|---|---|
| 0 | class id |
| 1 | confidence (smoothed winner score, 0..127) |
| 2 | votes (winner count in the 8-deep history) |
| 3 | reserved (0) |
| 4..7 | u32 latency in clock cycles, window issue → decision (÷ clk MHz for µs) |

### RSP_VERSION payload

| Offset | Field | Offset | Field |
|---|---|---|---|
| 0 | major | 6 | WINDOW_STRIDE |
| 1 | minor | 7 | CONV_K |
| 2 | patch | 8 | CONV_OUT_CH |
| 3 | protocol version | 9 | NUM_CLASSES |
| 4 | NUM_MFCC | 10 | PARALLEL_OUT_CH |
| 5 | WINDOW_LEN | 11 | clock MHz |

The host refuses to stream if bytes 4..9 disagree with its compiled weight
header — the bit-exactness contract requires one model on both sides.

### RSP_STATS counter map (16 × u32, little-endian)

| Idx | Counter | Idx | Counter |
|---|---|---|---|
| 0 | frames_rx (committed) | 8 | keywords detected |
| 1 | frames_dropped (stream off) | 9 | last inference latency, cycles |
| 2 | pkts_rx (CRC-valid, any type) | 10 | max inference latency, cycles |
| 3 | crc_errors | 11 | accumulated latency (avg = 11/7) |
| 4 | framing/protocol errors | 12 | conv busy cycles (wraps ~358 s) |
| 5 | RX FIFO overflows | 13 | uptime cycles (wraps ~358 s) |
| 6 | windows scheduled | 14 | windows dropped (engine overrun) |
| 7 | inferences completed | 15 | packets transmitted |

Counters wrap mod 2³²; hosts difference successive snapshots.

## Extension points

* `CMD_WRITE_REG / CMD_READ_REG` for runtime smoothing thresholds: the
  configuration registers already exist in `register_file` with parameter
  defaults; only command plumbing is missing.
* Additional keyword classes: `NUM_CLASSES` is parameterized end to end;
  the payload formats need no change up to 255 classes.
