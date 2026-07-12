// -----------------------------------------------------------------------------
// Project : iCE40 KWS Accelerator
// Module  : keyword_detector
// Purpose : Event generation. Latches a detection from the temporal smoother
//           together with its attribution (host frame number, FPGA timestamp,
//           measured latency) and holds an EVT_KEYWORD request toward the
//           packet encoder until granted.
//
// EVT_KEYWORD payload (8 bytes):
//   [0] class_id   [1] confidence (0..127)   [2] votes   [3] reserved = 0
//   [4..7] latency in clock cycles, little-endian (window issue -> decision)
//
// Queueing: depth 1. Events are separated by the smoother's debounce interval
// (hundreds of ms) while a maximal in-flight packet drains in ~7 ms, so a
// second detection while one is pending is impossible in a sane
// configuration; if it happens (debounce misconfigured to ~0), the new event
// is dropped and counted via drop_o.
// -----------------------------------------------------------------------------
`default_nettype none

module keyword_detector #(
  parameter int unsigned N           = kws_pkg::NUM_CLASSES,
  parameter int unsigned MAX_PAYLOAD = kws_pkg::PROTO_MAX_PAYLOAD
) (
  input  wire         clk_i,
  input  wire         rst_ni,

  // Detection input (temporal smoothing)
  input  wire                   detect_i,
  input  wire [$clog2(N)-1:0]   class_i,
  input  wire [7:0]             conf_i,
  input  wire [3:0]             votes_i,
  input  wire [31:0]            latency_i,     // cycles, window issue -> now
  input  wire [31:0]            frame_num_i,   // newest host frame in window
  input  wire [31:0]            ts_ms_i,       // FPGA milliseconds

  // Encoder request (held until ack)
  output logic        evt_req_o,
  input  wire         evt_ack_i,      // pulse: encoder accepted the request
  output logic [7:0]  evt_type_o,
  output logic [$clog2(MAX_PAYLOAD+1)-1:0] evt_len_o,
  output logic [31:0] evt_timestamp_o,
  output logic [31:0] evt_frame_o,

  // Payload byte fetch (combinational; encoder samples 2 cycles later)
  input  wire  [$clog2(MAX_PAYLOAD)-1:0] pl_idx_i,
  output logic [7:0]  pl_data_o,

  output logic        drop_o          // 1-cycle strobe: event lost (queue full)
);

  localparam int unsigned EVT_LEN = 8;

  logic [7:0]  class_q, conf_q, votes_q;
  logic [31:0] latency_q;

  assign evt_type_o = kws_pkg::PKT_EVT_KEYWORD;
  assign evt_len_o  = ($clog2(MAX_PAYLOAD+1))'(EVT_LEN);

  always_ff @(posedge clk_i or negedge rst_ni) begin
    if (!rst_ni) begin
      evt_req_o       <= 1'b0;
      evt_timestamp_o <= '0;
      evt_frame_o     <= '0;
      class_q         <= '0;
      conf_q          <= '0;
      votes_q         <= '0;
      latency_q       <= '0;
      drop_o          <= 1'b0;
    end else begin
      drop_o <= 1'b0;

      if (detect_i && !evt_req_o) begin
        class_q         <= 8'(class_i);
        conf_q          <= conf_i;
        votes_q         <= {4'h0, votes_i};
        latency_q       <= latency_i;
        evt_timestamp_o <= ts_ms_i;
        evt_frame_o     <= frame_num_i;
        evt_req_o       <= 1'b1;
      end else if (detect_i && evt_req_o) begin
        drop_o <= 1'b1;
      end

      if (evt_ack_i) begin
        evt_req_o <= 1'b0;
      end
    end
  end

  // Payload is 8 bytes; upper index bits only matter for larger payloads.
  wire _unused_idx_hi = &{1'b0, pl_idx_i[$clog2(MAX_PAYLOAD)-1:3]};

  always_comb begin
    unique case (pl_idx_i[2:0])
      3'd0:    pl_data_o = class_q;
      3'd1:    pl_data_o = conf_q;
      3'd2:    pl_data_o = votes_q;
      3'd3:    pl_data_o = 8'h00;
      3'd4:    pl_data_o = latency_q[7:0];
      3'd5:    pl_data_o = latency_q[15:8];
      3'd6:    pl_data_o = latency_q[23:16];
      default: pl_data_o = latency_q[31:24];
    endcase
  end

`ifndef SYNTHESIS
  always_ff @(posedge clk_i) begin
    assert (!(detect_i && evt_req_o))
      else $warning("keyword_detector: event dropped (queue full)");
  end
`endif

endmodule : keyword_detector

`default_nettype wire
