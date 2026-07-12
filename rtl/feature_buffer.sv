// -----------------------------------------------------------------------------
// Project : iCE40 KWS Accelerator
// Module  : feature_buffer
// Purpose : Circular INT8 feature store = feature_memory (data plane) +
//           circular_buffer_controller (pointer/commit plane).
//
// Write side (packet decoder): byte-granular speculative writes into the
// current slot, then a commit pulse once the frame's CRC verifies.
// Read side (conv engine): random access by absolute frame index and
// coefficient, 1-cycle latency. The reader computes wrapped frame indices
// itself ((base + offset) mod HIST_DEPTH is free with a power-of-two depth).
//
// UART reception and inference run concurrently by construction: the two
// planes share nothing but the RAM, whose write and read ports are disjoint
// EBR ports.
// -----------------------------------------------------------------------------
`default_nettype none

module feature_buffer #(
  parameter int unsigned DATA_W     = 8,
  parameter int unsigned NUM_MFCC   = 40,
  parameter int unsigned HIST_DEPTH = 64,
  parameter int unsigned WINDOW_LEN = 32
) (
  input  wire                           clk_i,
  input  wire                           rst_ni,
  input  wire                           clear_i,
  // Decoder write stream (speculative)
  input  wire                           wr_en_i,
  input  wire [$clog2(NUM_MFCC)-1:0]    wr_coef_i,
  input  wire [DATA_W-1:0]              wr_data_i,
  input  wire                           commit_i,
  input  wire [31:0]                    frame_num_i,
  input  wire [31:0]                    timestamp_i,
  // Compute read port
  input  wire [$clog2(HIST_DEPTH)-1:0]  rd_frame_i,
  input  wire [$clog2(NUM_MFCC)-1:0]    rd_coef_i,
  output logic [DATA_W-1:0]             rd_data_o,
  // State to the window scheduler
  output logic [$clog2(HIST_DEPTH)-1:0] wr_frame_o,
  output logic [$clog2(WINDOW_LEN):0]   frames_total_o,
  output logic [31:0]                   newest_frame_num_o,
  output logic [31:0]                   newest_timestamp_o
);

  circular_buffer_controller #(
    .HIST_DEPTH (HIST_DEPTH),
    .WINDOW_LEN (WINDOW_LEN)
  ) u_ctrl (
    .clk_i, .rst_ni, .clear_i,
    .commit_i,
    .frame_num_i,
    .timestamp_i,
    .wr_frame_o,
    .frames_total_o,
    .newest_frame_num_o,
    .newest_timestamp_o
  );

  feature_memory #(
    .DATA_W     (DATA_W),
    .NUM_MFCC   (NUM_MFCC),
    .HIST_DEPTH (HIST_DEPTH)
  ) u_mem (
    .clk_i,
    .wr_en_i,
    .wr_frame_i (wr_frame_o),
    .wr_coef_i,
    .wr_data_i,
    .rd_frame_i,
    .rd_coef_i,
    .rd_data_o
  );

endmodule : feature_buffer

`default_nettype wire
