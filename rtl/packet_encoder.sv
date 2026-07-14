// -----------------------------------------------------------------------------
// Project : iCE40 KWS Accelerator
// Module  : packet_encoder
// Purpose : Packet framer for the FPGA->host link. Serializes a response or
//           event descriptor into the wire format, computing the CRC on the
//           fly, and pushes bytes into the TX FIFO with back-pressure.
//
// Request interface: req_valid_i/req_ready_o single-cycle handshake latching
// {type, payload length, timestamp, frame number}. Payload bytes are fetched
// through a registered index interface: the encoder drives pl_idx_o and
// samples pl_data_i two cycles later, so the payload source may be a mux over
// registers, snapshot counters or a synchronous ROM. Peak emission rate of
// one byte per 3 cycles vastly outpaces the UART drain (~104 cycles/byte).
//
// CRC timing: outputs are registered, so the byte written in cycle N is
// folded into the CRC register during cycle N via crc_en_q, which is decided
// at scheduling time (one cycle earlier, when the byte's role is known).
// ST_CRC_WAIT gives the final fold one cycle to settle before the CRC bytes
// are read out.
//
// FSM: IDLE -> HDR(x13) -> [PL_ADDR -> PL_WAIT -> PL_PUSH]* -> CRC_WAIT
//        -> CRC_L -> CRC_H -> IDLE                          (docs/fsm.md)
// -----------------------------------------------------------------------------
`default_nettype none

module packet_encoder #(
  parameter int unsigned MAX_PAYLOAD = kws_pkg::PROTO_MAX_PAYLOAD
) (
  input  wire         clk_i,
  input  wire         rst_ni,

  // Packet request
  input  wire         req_valid_i,
  output logic        req_ready_o,
  input  wire  [7:0]  req_type_i,
  input  wire  [$clog2(MAX_PAYLOAD+1)-1:0] req_len_i,
  input  wire  [31:0] req_timestamp_i,
  input  wire  [31:0] req_frame_i,

  // Payload fetch (pl_data_i sampled 2 cycles after pl_idx_o updates)
  output logic [$clog2(MAX_PAYLOAD)-1:0] pl_idx_o,
  input  wire  [7:0]  pl_data_i,

  // TX FIFO write port. tx_full_i must be an ALMOST-full flag with >= 1
  // entry of margin: the write is registered, so it lands one cycle after
  // the flow-control check (kws_core provides level >= DEPTH-2).
  output logic        tx_wr_en_o,
  output logic [7:0]  tx_wr_data_o,
  input  wire         tx_full_i,

  output logic        pkt_done_o   // 1-cycle strobe: packet fully queued
);

  localparam int unsigned LEN_W = $clog2(MAX_PAYLOAD + 1);
  localparam int unsigned IDX_W = $clog2(MAX_PAYLOAD);

  typedef enum logic [2:0] {
    ST_IDLE, ST_HDR, ST_PL_ADDR, ST_PL_WAIT, ST_PL_PUSH,
    ST_CRC_WAIT, ST_CRC_L, ST_CRC_H
  } state_e;

  state_e           state_q;
  logic [3:0]       hdr_idx_q;    // 0..12
  logic [LEN_W-1:0] len_q;
  logic [IDX_W-1:0] pl_cnt_q;
  logic [7:0]       type_q;
  logic [31:0]      ts_q, fn_q;
  logic             crc_en_q;     // fold the byte being written this cycle

  assign req_ready_o = (state_q == ST_IDLE);

  // Header byte mux (index 0 = SOF .. 12 = FN3)
  logic [7:0] hdr_byte;
  always_comb begin
    unique case (hdr_idx_q)
      4'd0:    hdr_byte = kws_pkg::PROTO_SOF;
      4'd1:    hdr_byte = kws_pkg::PROTO_VERSION;
      4'd2:    hdr_byte = type_q;
      4'd3:    hdr_byte = {{(8-LEN_W){1'b0}}, len_q};
      4'd4:    hdr_byte = 8'h00;             // LEN_H (payloads are < 256)
      4'd5:    hdr_byte = ts_q[7:0];
      4'd6:    hdr_byte = ts_q[15:8];
      4'd7:    hdr_byte = ts_q[23:16];
      4'd8:    hdr_byte = ts_q[31:24];
      4'd9:    hdr_byte = fn_q[7:0];
      4'd10:   hdr_byte = fn_q[15:8];
      4'd11:   hdr_byte = fn_q[23:16];
      4'd12:   hdr_byte = fn_q[31:24];
      default: hdr_byte = 8'h00;
    endcase
  end

  logic [15:0] crc_calc;
  wire         push_ok = !tx_full_i;

  crc16 u_crc (
    .clk_i, .rst_ni,
    .clear_i (state_q == ST_IDLE),
    .en_i    (tx_wr_en_o && crc_en_q),
    .data_i  (tx_wr_data_o),
    .crc_o   (crc_calc)
  );

  always_ff @(posedge clk_i or negedge rst_ni) begin
    if (!rst_ni) begin
      state_q      <= ST_IDLE;
      hdr_idx_q    <= '0;
      len_q        <= '0;
      pl_cnt_q     <= '0;
      type_q       <= '0;
      ts_q         <= '0;
      fn_q         <= '0;
      crc_en_q     <= 1'b0;
      pl_idx_o     <= '0;
      tx_wr_en_o   <= 1'b0;
      tx_wr_data_o <= '0;
      pkt_done_o   <= 1'b0;
    end else begin
      tx_wr_en_o <= 1'b0;
      pkt_done_o <= 1'b0;

      unique case (state_q)
        ST_IDLE: begin
          hdr_idx_q <= '0;
          pl_cnt_q  <= '0;
          if (req_valid_i) begin
            type_q  <= req_type_i;
            len_q   <= req_len_i;
            ts_q    <= req_timestamp_i;
            fn_q    <= req_frame_i;
            state_q <= ST_HDR;
          end
        end

        // One header byte per cycle while the FIFO accepts. SOF (index 0) is
        // excluded from the CRC.
        ST_HDR: begin
          if (push_ok) begin
            tx_wr_en_o   <= 1'b1;
            tx_wr_data_o <= hdr_byte;
            crc_en_q     <= (hdr_idx_q != 4'd0);
            if (hdr_idx_q == 4'd12) begin
              state_q <= (len_q == '0) ? ST_CRC_WAIT : ST_PL_ADDR;
            end else begin
              hdr_idx_q <= hdr_idx_q + 1'b1;
            end
          end
        end

        // Present payload index; the source responds within two cycles.
        ST_PL_ADDR: begin
          pl_idx_o <= pl_cnt_q;
          state_q  <= ST_PL_WAIT;
        end

        ST_PL_WAIT: state_q <= ST_PL_PUSH;

        ST_PL_PUSH: begin
          if (push_ok) begin
            tx_wr_en_o   <= 1'b1;
            tx_wr_data_o <= pl_data_i;
            crc_en_q     <= 1'b1;
            if (32'(pl_cnt_q) == 32'(len_q) - 1) begin
              state_q <= ST_CRC_WAIT;
            end else begin
              pl_cnt_q <= pl_cnt_q + 1'b1;
              state_q  <= ST_PL_ADDR;
            end
          end
        end

        // The final covered byte folds into the CRC register during this
        // cycle; crc_calc is stable from the next cycle on.
        ST_CRC_WAIT: state_q <= ST_CRC_L;

        ST_CRC_L: begin
          if (push_ok) begin
            tx_wr_en_o   <= 1'b1;
            tx_wr_data_o <= crc_calc[7:0];
            crc_en_q     <= 1'b0;
            state_q      <= ST_CRC_H;
          end
        end

        ST_CRC_H: begin
          if (push_ok) begin
            tx_wr_en_o   <= 1'b1;
            tx_wr_data_o <= crc_calc[15:8];
            crc_en_q     <= 1'b0;
            pkt_done_o   <= 1'b1;
            state_q      <= ST_IDLE;
          end
        end

        default: state_q <= ST_IDLE;
      endcase
    end
  end

`ifndef SYNTHESIS
  always_ff @(posedge clk_i) begin
    if (req_valid_i && req_ready_o) begin
      assert (32'(req_len_i) <= MAX_PAYLOAD)
        else $error("packet_encoder: request length %0d exceeds MAX_PAYLOAD",
                    req_len_i);
    end
  end
`endif

endmodule : packet_encoder

`default_nettype wire
