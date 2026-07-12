// -----------------------------------------------------------------------------
// Project : iCE40 KWS Accelerator
// Module  : packet_decoder
// Purpose : Byte-stream to packet parser for the host->FPGA link.
//
// Frame format (little-endian multi-byte fields):
//   [SOF][VER][TYPE][LEN_L][LEN_H][TS0..TS3][FN0..FN3][PAYLOAD*LEN][CRC_L][CRC_H]
//   CRC16-CCITT-FALSE over VER..end-of-payload.
//
// Key properties:
//  * Resynchronization: in HUNT, bytes are discarded until SOF; any structural
//    violation (oversized LEN, inter-byte timeout) returns to HUNT, so the
//    parser always relocks on the next packet boundary.
//  * Speculative feature writes: DATA_FEATURE payload bytes are streamed into
//    the feature buffer as they arrive (feat_wr_*); the buffer's write pointer
//    only advances on feat_commit_o, which fires after the CRC verifies. A
//    corrupt packet therefore costs zero buffering and zero cleanup.
//  * Semantic errors on structurally valid packets (unknown type, wrong
//    length, wrong version) are parsed to completion, CRC-checked, and
//    reported via err_valid_o so the host receives an RSP_ERROR; CRC failures
//    are counted silently (a corrupt packet cannot be attributed reliably).
//
// FSM: HUNT -> VER -> TYPE -> LEN_L -> LEN_H -> TS -> FN -> PAYLOAD
//        -> CRC_L -> CRC_H -> (dispatch) -> HUNT      (docs/fsm.md)
// -----------------------------------------------------------------------------
`default_nettype none

module packet_decoder #(
  parameter int unsigned NUM_MFCC       = kws_pkg::NUM_MFCC,
  parameter int unsigned MAX_PAYLOAD    = kws_pkg::PROTO_MAX_PAYLOAD,
  parameter int unsigned TIMEOUT_CYCLES = 1_200_000   // 100 ms @ 12 MHz
) (
  input  wire         clk_i,
  input  wire         rst_ni,

  // RX FIFO read interface (uart_fifo: 1-cycle read latency)
  input  wire         fifo_empty_i,
  output logic        fifo_rd_en_o,
  input  wire  [7:0]  fifo_rd_data_i,
  input  wire         fifo_rd_valid_i,

  // Speculative feature stream into the feature buffer
  output logic                        feat_wr_en_o,
  output logic [$clog2(NUM_MFCC)-1:0] feat_wr_idx_o,
  output logic [7:0]                  feat_wr_data_o,
  output logic                        feat_commit_o,   // CRC verified: latch frame
  output logic [31:0]                 feat_frame_num_o,
  output logic [31:0]                 feat_timestamp_o,

  // Verified command strobe (PING/RESET/START/STOP/READ_*)
  output logic        cmd_valid_o,
  output logic [7:0]  cmd_type_o,
  output logic [31:0] cmd_frame_num_o,   // header fields of the command packet,
  output logic [31:0] cmd_timestamp_o,   // valid with cmd_valid_o / err_valid_o

  // Semantic error report (structurally valid packet, CRC verified)
  output logic        err_valid_o,
  output logic [7:0]  err_code_o,
  output logic [7:0]  err_detail_o,   // offending TYPE byte

  // Statistics strobes
  output logic        pkt_ok_o,       // any packet passing CRC
  output logic        crc_err_o,      // CRC mismatch
  output logic        proto_err_o     // desync: bad LEN bound or timeout
);

  localparam int unsigned IDX_W = $clog2(MAX_PAYLOAD);
  localparam int unsigned TO_W  = $clog2(TIMEOUT_CYCLES);

  typedef enum logic [3:0] {
    ST_HUNT, ST_VER, ST_TYPE, ST_LEN_L, ST_LEN_H,
    ST_TS, ST_FN, ST_PAYLOAD, ST_CRC_L, ST_CRC_H
  } state_e;

  state_e             state_q;
  logic [7:0]         type_q;
  logic [15:0]        len_q;
  logic [1:0]         mb_cnt_q;      // multi-byte field counter (TS/FN)
  logic [IDX_W-1:0]   pl_idx_q;      // payload byte index
  logic [31:0]        ts_q, fn_q;
  logic [7:0]         crc_lo_q;
  logic [7:0]         ver_err_q;     // sticky within packet: bad VER
  logic [7:0]         len_err_q;     // sticky within packet: bad LEN
  logic [TO_W-1:0]    to_cnt_q;

  // ---------------------------------------------------------------------------
  // Byte pump: at most one FIFO read outstanding; a new read may issue on the
  // same cycle the previous byte lands, sustaining one byte per cycle.
  // ---------------------------------------------------------------------------
  logic pending_q;
  assign fifo_rd_en_o = !fifo_empty_i && (!pending_q || fifo_rd_valid_i);

  always_ff @(posedge clk_i or negedge rst_ni) begin
    if (!rst_ni)            pending_q <= 1'b0;
    else if (fifo_rd_en_o)  pending_q <= 1'b1;
    else if (fifo_rd_valid_i) pending_q <= 1'b0;
  end

  wire       byte_valid = fifo_rd_valid_i;
  wire [7:0] rx_byte    = fifo_rd_data_i;

  // ---------------------------------------------------------------------------
  // CRC engine: cleared when SOF is consumed, folds VER..payload
  // ---------------------------------------------------------------------------
  logic        crc_clear, crc_en;
  logic [15:0] crc_calc;

  assign crc_clear = byte_valid && (state_q == ST_HUNT) && (rx_byte == kws_pkg::PROTO_SOF);
  assign crc_en    = byte_valid && (state_q inside {ST_VER, ST_TYPE, ST_LEN_L,
                                                    ST_LEN_H, ST_TS, ST_FN,
                                                    ST_PAYLOAD});

  crc16 u_crc (
    .clk_i, .rst_ni,
    .clear_i (crc_clear),
    .en_i    (crc_en),
    .data_i  (rx_byte),
    .crc_o   (crc_calc)
  );

  // ---------------------------------------------------------------------------
  // Dispatch helpers
  // ---------------------------------------------------------------------------
  wire is_feature = (type_q == kws_pkg::PKT_DATA_FEATURE);
  wire is_command = (type_q inside {kws_pkg::PKT_CMD_PING, kws_pkg::PKT_CMD_RESET,
                                    kws_pkg::PKT_CMD_START_STREAM, kws_pkg::PKT_CMD_STOP_STREAM,
                                    kws_pkg::PKT_CMD_READ_STATS, kws_pkg::PKT_CMD_READ_VERSION});
  wire len_valid  = is_feature ? (len_q == 16'(NUM_MFCC))
                               : (is_command ? (len_q == 16'd0) : 1'b1);

  // ---------------------------------------------------------------------------
  // Main FSM
  // ---------------------------------------------------------------------------
  always_ff @(posedge clk_i or negedge rst_ni) begin
    if (!rst_ni) begin
      state_q          <= ST_HUNT;
      type_q           <= '0;
      len_q            <= '0;
      mb_cnt_q         <= '0;
      pl_idx_q         <= '0;
      ts_q             <= '0;
      fn_q             <= '0;
      crc_lo_q         <= '0;
      ver_err_q        <= '0;
      len_err_q        <= '0;
      to_cnt_q         <= '0;
      feat_wr_en_o     <= 1'b0;
      feat_wr_idx_o    <= '0;
      feat_wr_data_o   <= '0;
      feat_commit_o    <= 1'b0;
      feat_frame_num_o <= '0;
      feat_timestamp_o <= '0;
      cmd_valid_o      <= 1'b0;
      cmd_type_o       <= '0;
      cmd_frame_num_o  <= '0;
      cmd_timestamp_o  <= '0;
      err_valid_o      <= 1'b0;
      err_code_o       <= '0;
      err_detail_o     <= '0;
      pkt_ok_o         <= 1'b0;
      crc_err_o        <= 1'b0;
      proto_err_o      <= 1'b0;
    end else begin
      // Single-cycle strobes
      feat_wr_en_o  <= 1'b0;
      feat_commit_o <= 1'b0;
      cmd_valid_o   <= 1'b0;
      err_valid_o   <= 1'b0;
      pkt_ok_o      <= 1'b0;
      crc_err_o     <= 1'b0;
      proto_err_o   <= 1'b0;

      // Inter-byte timeout: mid-packet silence returns the parser to HUNT.
      if (byte_valid) begin
        to_cnt_q <= '0;
      end else if (state_q != ST_HUNT) begin
        if (to_cnt_q == TO_W'(TIMEOUT_CYCLES - 1)) begin
          state_q     <= ST_HUNT;
          to_cnt_q    <= '0;
          proto_err_o <= 1'b1;
        end else begin
          to_cnt_q <= to_cnt_q + 1'b1;
        end
      end

      if (byte_valid) begin
        unique case (state_q)
          ST_HUNT: begin
            if (rx_byte == kws_pkg::PROTO_SOF) begin
              state_q   <= ST_VER;
              ver_err_q <= '0;
              len_err_q <= '0;
            end
          end

          ST_VER: begin
            if (rx_byte != kws_pkg::PROTO_VERSION) ver_err_q <= 8'h01;
            state_q <= ST_TYPE;
          end

          ST_TYPE: begin
            type_q  <= rx_byte;
            state_q <= ST_LEN_L;
          end

          ST_LEN_L: begin
            len_q[7:0] <= rx_byte;
            state_q    <= ST_LEN_H;
          end

          ST_LEN_H: begin
            len_q[15:8] <= rx_byte;
            mb_cnt_q    <= '0;
            // Bound check with the full 16-bit value now available.
            if ({rx_byte, len_q[7:0]} > 16'(MAX_PAYLOAD)) begin
              state_q     <= ST_HUNT;   // cannot trust LEN: resync
              proto_err_o <= 1'b1;
            end else begin
              state_q <= ST_TS;
            end
          end

          ST_TS: begin
            ts_q[8*mb_cnt_q +: 8] <= rx_byte;
            mb_cnt_q <= mb_cnt_q + 1'b1;
            if (mb_cnt_q == 2'd3) state_q <= ST_FN;
          end

          ST_FN: begin
            fn_q[8*mb_cnt_q +: 8] <= rx_byte;
            mb_cnt_q <= mb_cnt_q + 1'b1;
            if (mb_cnt_q == 2'd3) begin
              pl_idx_q <= '0;
              // Record semantic length error once LEN and TYPE are both known.
              if (!len_valid) len_err_q <= 8'h01;
              state_q  <= (len_q == 16'd0) ? ST_CRC_L : ST_PAYLOAD;
            end
          end

          ST_PAYLOAD: begin
            // Speculative write: only for well-formed feature packets.
            if (is_feature && len_valid && (ver_err_q == '0)) begin
              feat_wr_en_o   <= 1'b1;
              feat_wr_idx_o  <= pl_idx_q[$clog2(NUM_MFCC)-1:0];
              feat_wr_data_o <= rx_byte;
            end
            pl_idx_q <= pl_idx_q + 1'b1;
            if (pl_idx_q == IDX_W'(len_q - 1)) state_q <= ST_CRC_L;
          end

          ST_CRC_L: begin
            crc_lo_q <= rx_byte;
            state_q  <= ST_CRC_H;
          end

          ST_CRC_H: begin
            state_q <= ST_HUNT;
            if ({rx_byte, crc_lo_q} == crc_calc) begin
              pkt_ok_o        <= 1'b1;
              cmd_frame_num_o <= fn_q;
              cmd_timestamp_o <= ts_q;
              if (ver_err_q != '0) begin
                err_valid_o  <= 1'b1;
                err_code_o   <= kws_pkg::ERR_BAD_VERSION;
                err_detail_o <= type_q;
              end else if (len_err_q != '0) begin
                err_valid_o  <= 1'b1;
                err_code_o   <= kws_pkg::ERR_BAD_LENGTH;
                err_detail_o <= type_q;
              end else if (is_feature) begin
                feat_commit_o    <= 1'b1;
                feat_frame_num_o <= fn_q;
                feat_timestamp_o <= ts_q;
              end else if (is_command) begin
                cmd_valid_o <= 1'b1;
                cmd_type_o  <= type_q;
              end else begin
                err_valid_o  <= 1'b1;
                err_code_o   <= kws_pkg::ERR_UNKNOWN_TYPE;
                err_detail_o <= type_q;
              end
            end else begin
              crc_err_o <= 1'b1;
            end
          end

          default: state_q <= ST_HUNT;
        endcase
      end
    end
  end

`ifndef SYNTHESIS
  always_ff @(posedge clk_i) begin
    if (feat_wr_en_o) begin
      assert (32'(feat_wr_idx_o) < NUM_MFCC)
        else $error("packet_decoder: feature index out of range");
    end
  end
`endif

endmodule : packet_decoder

`default_nettype wire
