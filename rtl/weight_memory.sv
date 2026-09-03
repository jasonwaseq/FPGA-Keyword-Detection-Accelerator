// -----------------------------------------------------------------------------
// Project : iCE40 KWS Accelerator
// Module  : weight_memory
// Purpose : INT8 dense-layer weight ROM. Flattened layout:
//           index = class * IN_LEN + (t * CH + ch), matching the pooled
//           activation RAM layout so the classifier walks both memories with
//           a single incrementing address.
// -----------------------------------------------------------------------------
`default_nettype none

module weight_memory #(
  parameter int unsigned DATA_W      = 8,
  parameter int unsigned NUM_CLASSES = 4,
  parameter int unsigned IN_LEN      = 120,
  parameter              MEM_FILE    = "weights/dense_weights.mem"
) (
  input  wire                                      clk_i,
  input  wire  [$clog2(NUM_CLASSES*IN_LEN)-1:0]    addr_i,
  output logic [DATA_W-1:0]                        data_o
);

  rom_sync #(
    .DATA_W   (DATA_W),
    .DEPTH    (NUM_CLASSES * IN_LEN),
    .MEM_FILE (MEM_FILE)
  ) u_rom (
    .clk_i,
    .addr_i,
    .data_o
  );

endmodule : weight_memory

`default_nettype wire
