// -----------------------------------------------------------------------------
// Project : iCE40 KWS Accelerator
// Module  : feature_memory
// Purpose : Frame-addressed INT8 feature store: HIST_DEPTH frames of NUM_MFCC
//           coefficients, one write port (UART side), one read port (conv
//           engine side), both synchronous.
//
// Address layout: {frame, coef} with the coefficient field padded to a power
// of two (COEF_SLOT = 2**clog2(NUM_MFCC)). For NUM_MFCC=40 this spends 64-40
// = 24 unused bytes per frame (3 extra EBRs) to make frame indexing pure bit
// concatenation - no per-access multiplier, no carry chain in the address
// path. The alternative (dense frame*40+coef addressing) saves 3 of 30 EBRs
// at the cost of a 6x6 multiply-add in both address generators; EBR is not
// the scarce resource here, address-path simplicity wins.
// -----------------------------------------------------------------------------
`default_nettype none

module feature_memory #(
  parameter int unsigned DATA_W     = 8,
  parameter int unsigned NUM_MFCC   = 40,
  parameter int unsigned HIST_DEPTH = 64   // frames, power of two
) (
  input  wire                           clk_i,
  // Write port (packet decoder)
  input  wire                           wr_en_i,
  input  wire [$clog2(HIST_DEPTH)-1:0]  wr_frame_i,
  input  wire [$clog2(NUM_MFCC)-1:0]    wr_coef_i,
  input  wire [DATA_W-1:0]              wr_data_i,
  // Read port (conv engine), 1-cycle latency
  input  wire [$clog2(HIST_DEPTH)-1:0]  rd_frame_i,
  input  wire [$clog2(NUM_MFCC)-1:0]    rd_coef_i,
  output logic [DATA_W-1:0]             rd_data_o
);

  localparam int unsigned FR_W  = $clog2(HIST_DEPTH);
  localparam int unsigned CO_W  = $clog2(NUM_MFCC);
  localparam int unsigned DEPTH = HIST_DEPTH << CO_W;

  ram_dp_sync #(
    .DATA_W (DATA_W),
    .DEPTH  (DEPTH)
  ) u_ram (
    .clk_i,
    .wr_en_i,
    .wr_addr_i ({wr_frame_i, wr_coef_i}),
    .wr_data_i,
    .rd_addr_i ({rd_frame_i, rd_coef_i}),
    .rd_data_o
  );

`ifndef SYNTHESIS
  initial begin
    assert (HIST_DEPTH == (1 << FR_W))
      else $error("feature_memory: HIST_DEPTH must be a power of two");
  end
`endif

endmodule : feature_memory

`default_nettype wire
