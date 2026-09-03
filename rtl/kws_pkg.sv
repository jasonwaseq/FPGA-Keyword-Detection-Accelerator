// -----------------------------------------------------------------------------
// Project : iCE40 KWS Accelerator
// Module  : kws_pkg
// Purpose : Global constants: model geometry defaults, UART protocol encoding,
//           statistics counter map, version identity.
//
// This package is the RTL-side single source of truth. The Python export
// library (model/kws_quant.py) and the C reference model / host application
// (host/include/kws_protocol.h, weights/kws_weights.h) carry the same
// constants; the full-system testbench cross-checks them at run time via the
// VERSION response payload.
//
// Reusable leaf modules (UART, FIFO, CRC, ROM/RAM primitives) deliberately do
// NOT import this package - they stay context-free. Application modules
// default their parameters from here and remain overridable per instance.
// -----------------------------------------------------------------------------
`default_nettype none

package kws_pkg;

  // ---------------------------------------------------------------------------
  // Clocking / UART
  // ---------------------------------------------------------------------------
  localparam int unsigned CLK_FREQ_HZ = 12_000_000;  // iCEBreaker 12 MHz osc
  localparam int unsigned UART_BAUD   = 115_200;

  // ---------------------------------------------------------------------------
  // Model geometry (must match weights/model_params.json)
  // ---------------------------------------------------------------------------
  localparam int unsigned NUM_MFCC     = 40;  // coefficients per frame
  localparam int unsigned WINDOW_LEN   = 32;  // frames per inference window
  localparam int unsigned WINDOW_STRIDE = 8;  // frames between windows
  localparam int unsigned HIST_DEPTH   = 64;  // circular buffer frames (2**n)
  localparam int unsigned CONV_K       = 3;   // temporal kernel size
  localparam int unsigned CONV_OUT_CH  = 8;   // conv output channels
  localparam int unsigned CONV_OUT_LEN = WINDOW_LEN - CONV_K + 1;  // 30
  localparam int unsigned POOL_SIZE    = 2;   // temporal pool factor (2**n)
  localparam int unsigned POOL_OUT_LEN = CONV_OUT_LEN / POOL_SIZE; // 15
  localparam int unsigned DENSE_IN     = POOL_OUT_LEN * CONV_OUT_CH; // 120
  localparam int unsigned NUM_CLASSES  = 4;   // 0=silence 1=unknown 2.. keywords

  // Datapath widths
  localparam int unsigned DATA_W = 8;   // INT8 features / weights / activations
  localparam int unsigned ACC_W  = 24;  // accumulator (worst case |acc| < 2**21)
  localparam int unsigned MULT_W = 16;  // requant multiplier width (M < 2**15)

  // Parallel conv MAC lanes (output channels computed concurrently).
  // Must divide CONV_OUT_CH. 1/2/4/8 supported; see docs/pipeline.md.
  localparam int unsigned PARALLEL_OUT_CH = 2;

  // ---------------------------------------------------------------------------
  // Temporal smoothing defaults ("smoothing defaults" - keep synchronized with
  // host/src/ref_model.c kws_smooth_init and model/kws_quant.py
  // SMOOTH_DEFAULTS). Operating point selected by training/tune_detect.py +
  // eval_stream.py on held-out Speech Commands streams: depth 4 matches the
  // ~3-4 windows a short spoken keyword fully covers at stride 8.
  // ---------------------------------------------------------------------------
  localparam int unsigned SMOOTH_DEPTH   = 4;      // history length (2**n)
  localparam logic signed [7:0] CONF_THRESH = 8'sd25; // smoothed score threshold
  localparam int unsigned VOTE_MIN       = 2;      // majority votes required
  localparam int unsigned MIN_CONSEC     = 1;      // consecutive candidates
  localparam int unsigned DEBOUNCE_INFER = 12;     // refractory inferences
  localparam logic [NUM_CLASSES-1:0] TARGET_MASK = 4'b1100; // classes 2,3 trigger

  // ---------------------------------------------------------------------------
  // UART packet protocol (see docs/protocol.md)
  //   [SOF][VER][TYPE][LEN_L][LEN_H][TS0..TS3][FN0..FN3][PAYLOAD...][CRC_L][CRC_H]
  //   CRC16-CCITT-FALSE over VER..end-of-payload (SOF and CRC excluded)
  // ---------------------------------------------------------------------------
  localparam logic [7:0] PROTO_SOF     = 8'hA5;
  localparam logic [7:0] PROTO_VERSION = 8'h01;
  localparam int unsigned PROTO_MAX_PAYLOAD = 64;  // header is 13 bytes SOF..FN3

  // Host -> FPGA
  localparam logic [7:0] PKT_CMD_PING         = 8'h01;
  localparam logic [7:0] PKT_CMD_RESET        = 8'h02;
  localparam logic [7:0] PKT_CMD_START_STREAM = 8'h03;
  localparam logic [7:0] PKT_CMD_STOP_STREAM  = 8'h04;
  localparam logic [7:0] PKT_CMD_READ_STATS   = 8'h05;
  localparam logic [7:0] PKT_CMD_READ_VERSION = 8'h06;
  localparam logic [7:0] PKT_DATA_FEATURE     = 8'h10;

  // FPGA -> Host
  localparam logic [7:0] PKT_RSP_ACK      = 8'h81;
  localparam logic [7:0] PKT_RSP_ERROR    = 8'h82;
  localparam logic [7:0] PKT_RSP_STATS    = 8'h83;
  localparam logic [7:0] PKT_RSP_VERSION  = 8'h84;
  localparam logic [7:0] PKT_EVT_KEYWORD  = 8'h90;

  // RSP_ERROR payload byte 0
  localparam logic [7:0] ERR_BAD_LENGTH   = 8'h01;
  localparam logic [7:0] ERR_UNKNOWN_TYPE = 8'h02;
  localparam logic [7:0] ERR_BAD_VERSION  = 8'h03;

  // ---------------------------------------------------------------------------
  // Statistics counter map (RSP_STATS payload: 16 x u32, little-endian)
  // ---------------------------------------------------------------------------
  localparam int unsigned STATS_NUM = 16;
  typedef enum logic [3:0] {
    STAT_FRAMES_RX      = 4'd0,   // feature frames accepted into buffer
    STAT_FRAMES_DROPPED = 4'd1,   // feature frames discarded (stream stopped)
    STAT_PKTS_RX        = 4'd2,   // packets with valid CRC (any type)
    STAT_CRC_ERRORS     = 4'd3,   // packets rejected on CRC
    STAT_FRAMING_ERRORS = 4'd4,   // UART framing + protocol format errors
    STAT_FIFO_OVERFLOWS = 4'd5,   // RX FIFO drops
    STAT_WINDOWS_SCHED  = 4'd6,   // windows issued to the engine
    STAT_INFERENCES     = 4'd7,   // inferences completed
    STAT_KEYWORDS       = 4'd8,   // keyword events emitted
    STAT_LAT_LAST       = 4'd9,   // last inference latency, cycles
    STAT_LAT_MAX        = 4'd10,  // max inference latency, cycles
    STAT_LAT_ACC        = 4'd11,  // accumulated latency (avg = acc/inferences)
    STAT_CONV_BUSY      = 4'd12,  // cycles conv engine busy (wraps)
    STAT_UPTIME_CYC     = 4'd13,  // total cycles since reset (wraps)
    STAT_WINDOWS_DROP   = 4'd14,  // windows skipped because engine was busy
    STAT_TX_PKTS        = 4'd15   // packets transmitted
  } stat_idx_e;

  // ---------------------------------------------------------------------------
  // Version identity (RSP_VERSION payload)
  // ---------------------------------------------------------------------------
  localparam logic [7:0] VER_MAJOR = 8'd1;
  localparam logic [7:0] VER_MINOR = 8'd0;
  localparam logic [7:0] VER_PATCH = 8'd0;

endpackage : kws_pkg

`default_nettype wire
