// -----------------------------------------------------------------------------
// Project : iCE40 KWS Accelerator
// Module  : register_file
// Purpose : Control/status plane: executes verified host commands, owns the
//           runtime configuration registers, and generates all command
//           responses (ACK / ERROR / STATS / VERSION) toward the packet
//           encoder.
//
// Command semantics:
//   PING         - no side effect; ACK echoes the command type.
//   RESET        - pulses soft_clear_o (streaming pipeline + statistics are
//                  cleared, engines are reset); stream disabled; ACK.
//   START_STREAM - enables feature commits; pulses stream_clear_o so a new
//                  session never inherits stale window/smoothing history; ACK.
//   STOP_STREAM  - disables feature commits (buffer contents retained); ACK.
//   READ_STATS   - pulses snapshot_o (atomic counter capture), then serves
//                  the 64-byte RSP_STATS payload from the snapshot.
//   READ_VERSION - serves the 12-byte identity/geometry payload below.
//
// RSP_VERSION payload:
//   [0] major [1] minor [2] patch [3] protocol
//   [4] NUM_MFCC [5] WINDOW_LEN [6] WINDOW_STRIDE [7] CONV_K
//   [8] CONV_OUT_CH [9] NUM_CLASSES [10] PARALLEL [11] clk MHz
//
// Response queue is one deep: the host protocol is strictly one command in
// flight (the host library enforces it). A pipelined second command still
// executes its side effects but overwrites the pending response.
//
// Configuration registers reset to the kws_pkg defaults. The v1 wire protocol
// has no WRITE_REG command; the register file is the single place one would
// be added (documented extension point, docs/protocol.md).
// -----------------------------------------------------------------------------
`default_nettype none

module register_file
  import kws_pkg::*;
#(
  parameter int unsigned CLK_HZ      = kws_pkg::CLK_FREQ_HZ,
  parameter int unsigned N_CLASSES   = kws_pkg::NUM_CLASSES,
  parameter int unsigned MAX_PAYLOAD = kws_pkg::PROTO_MAX_PAYLOAD,
  parameter int unsigned PARALLEL    = kws_pkg::PARALLEL_OUT_CH
) (
  input  wire         clk_i,
  input  wire         rst_ni,

  // Decoder interface (CRC-verified)
  input  wire         cmd_valid_i,
  input  wire  [7:0]  cmd_type_i,
  input  wire         err_valid_i,
  input  wire  [7:0]  err_code_i,
  input  wire  [7:0]  err_detail_i,
  input  wire  [31:0] cmd_frame_num_i,

  // Time base
  input  wire  [31:0] ts_ms_i,

  // Control outputs
  output logic        stream_en_o,
  output logic        soft_clear_o,     // 1-cycle: RESET command
  output logic        stream_clear_o,   // 1-cycle: START command
  output logic        snapshot_o,       // 1-cycle: READ_STATS command

  // Statistics snapshot read (combinational)
  output logic [3:0]  stats_idx_o,
  input  wire  [31:0] stats_data_i,

  // Runtime configuration (temporal smoothing / pooling)
  output logic signed [7:0]             cfg_thresh_o,
  output logic        [3:0]             cfg_vote_min_o,
  output logic        [3:0]             cfg_min_consec_o,
  output logic        [7:0]             cfg_debounce_o,
  output logic        [N_CLASSES-1:0]   cfg_target_mask_o,
  output logic                          cfg_pool_mode_o,   // 0 max, 1 avg
  output logic                          cfg_smooth_en_o,

  // Encoder request (held until ack)
  output logic        rsp_req_o,
  input  wire         rsp_ack_i,
  output logic [7:0]  rsp_type_o,
  output logic [$clog2(MAX_PAYLOAD+1)-1:0] rsp_len_o,
  output logic [31:0] rsp_timestamp_o,
  output logic [31:0] rsp_frame_o,

  // Payload byte fetch (combinational; encoder samples 2 cycles later)
  input  wire  [$clog2(MAX_PAYLOAD)-1:0] pl_idx_i,
  output logic [7:0]  pl_data_o
);

  localparam int unsigned LEN_W = $clog2(MAX_PAYLOAD + 1);

  logic [7:0] pl0_q, pl1_q;   // small payload registers (ACK echo / ERROR)

  // --- command execution and response latch ----------------------------------
  always_ff @(posedge clk_i or negedge rst_ni) begin
    if (!rst_ni) begin
      stream_en_o     <= 1'b0;
      soft_clear_o    <= 1'b0;
      stream_clear_o  <= 1'b0;
      snapshot_o      <= 1'b0;
      rsp_req_o       <= 1'b0;
      rsp_type_o      <= '0;
      rsp_len_o       <= '0;
      rsp_timestamp_o <= '0;
      rsp_frame_o     <= '0;
      pl0_q           <= '0;
      pl1_q           <= '0;
    end else begin
      soft_clear_o   <= 1'b0;
      stream_clear_o <= 1'b0;
      snapshot_o     <= 1'b0;

      if (rsp_ack_i) rsp_req_o <= 1'b0;

      if (cmd_valid_i) begin
        rsp_req_o       <= 1'b1;
        rsp_timestamp_o <= ts_ms_i;
        rsp_frame_o     <= cmd_frame_num_i;
        rsp_type_o      <= PKT_RSP_ACK;
        rsp_len_o       <= LEN_W'(1);
        pl0_q           <= cmd_type_i;

        unique case (cmd_type_i)
          PKT_CMD_PING: ;
          PKT_CMD_RESET: begin
            soft_clear_o <= 1'b1;
            stream_en_o  <= 1'b0;
          end
          PKT_CMD_START_STREAM: begin
            stream_en_o    <= 1'b1;
            stream_clear_o <= 1'b1;
          end
          PKT_CMD_STOP_STREAM: begin
            stream_en_o <= 1'b0;
          end
          PKT_CMD_READ_STATS: begin
            snapshot_o <= 1'b1;
            rsp_type_o <= PKT_RSP_STATS;
            rsp_len_o  <= LEN_W'(4 * STATS_NUM);
          end
          PKT_CMD_READ_VERSION: begin
            rsp_type_o <= PKT_RSP_VERSION;
            rsp_len_o  <= LEN_W'(12);
          end
          default: ;  // unreachable: decoder only strobes known commands
        endcase
      end else if (err_valid_i) begin
        rsp_req_o       <= 1'b1;
        rsp_timestamp_o <= ts_ms_i;
        rsp_frame_o     <= cmd_frame_num_i;
        rsp_type_o      <= PKT_RSP_ERROR;
        rsp_len_o       <= LEN_W'(2);
        pl0_q           <= err_code_i;
        pl1_q           <= err_detail_i;
      end
    end
  end

  // --- runtime configuration (defaults from kws_pkg; extension point) --------
  always_ff @(posedge clk_i or negedge rst_ni) begin
    if (!rst_ni) begin
      cfg_thresh_o      <= CONF_THRESH;
      cfg_vote_min_o    <= 4'(VOTE_MIN);
      cfg_min_consec_o  <= 4'(MIN_CONSEC);
      cfg_debounce_o    <= 8'(DEBOUNCE_INFER);
      cfg_target_mask_o <= TARGET_MASK[N_CLASSES-1:0];
      cfg_pool_mode_o   <= 1'b0;   // max pooling
      cfg_smooth_en_o   <= 1'b1;
    end
    // No runtime write path in protocol v1.
  end

  // --- payload byte server ----------------------------------------------------
  // STATS: word index = pl_idx[5:2] into the snapshot, byte lane = pl_idx[1:0]
  assign stats_idx_o = pl_idx_i[5:2];

  logic [7:0] version_byte;
  always_comb begin
    unique case (pl_idx_i[3:0])
      4'd0:    version_byte = VER_MAJOR;
      4'd1:    version_byte = VER_MINOR;
      4'd2:    version_byte = VER_PATCH;
      4'd3:    version_byte = PROTO_VERSION;
      4'd4:    version_byte = 8'(NUM_MFCC);
      4'd5:    version_byte = 8'(WINDOW_LEN);
      4'd6:    version_byte = 8'(WINDOW_STRIDE);
      4'd7:    version_byte = 8'(CONV_K);
      4'd8:    version_byte = 8'(CONV_OUT_CH);
      4'd9:    version_byte = 8'(N_CLASSES);
      4'd10:   version_byte = 8'(PARALLEL);
      4'd11:   version_byte = 8'(CLK_HZ / 1_000_000);
      default: version_byte = 8'h00;
    endcase
  end

  always_comb begin
    unique case (rsp_type_o)
      PKT_RSP_STATS:   pl_data_o = stats_data_i[8 * pl_idx_i[1:0] +: 8];
      PKT_RSP_VERSION: pl_data_o = version_byte;
      PKT_RSP_ERROR:   pl_data_o = (pl_idx_i == '0) ? pl0_q : pl1_q;
      default:         pl_data_o = pl0_q;   // ACK: echoed command type
    endcase
  end

endmodule : register_file

`default_nettype wire
