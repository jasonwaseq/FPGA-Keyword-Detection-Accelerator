// -----------------------------------------------------------------------------
// Project : iCE40 KWS Accelerator
// Module  : bias_memory
// Purpose : Per-layer 32-bit parameter ROM. Layout (one word per line in the
//           .mem file): bias[0..N_BIAS-1], then the layer's requantization
//           multiplier M and shift S.
//
// At 32 x (N+2) bits these tables synthesize to LUT logic rather than EBR -
// exactly right for a handful of constants read a few times per inference.
// -----------------------------------------------------------------------------
`default_nettype none

module bias_memory #(
  parameter int unsigned N_BIAS   = 8,
  parameter              MEM_FILE = "weights/conv_bias.mem"
) (
  input  wire                          clk_i,
  input  wire  [$clog2(N_BIAS+2)-1:0]  addr_i,
  output logic [31:0]                  data_o
);

  // Address N_BIAS   -> requant multiplier M
  // Address N_BIAS+1 -> requant shift S
  rom_sync #(
    .DATA_W   (32),
    .DEPTH    (N_BIAS + 2),
    .MEM_FILE (MEM_FILE)
  ) u_rom (
    .clk_i,
    .addr_i,
    .data_o
  );

endmodule : bias_memory

`default_nettype wire
